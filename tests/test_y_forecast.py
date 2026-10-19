"""Role-separated temporal/patch/LIFT models and information-path contracts."""
from dataclasses import asdict

import pytest
import torch

from exobase.model import ModelConfig, build_model, scalar_patch_means
from exobase.lead_estimator import estimate_fft_leads


def settings(variant='patch-lift', mode='global'):
    return ModelConfig(1, 2, 32, variant=variant, future_patches=8,
        patch_encoder='mean' if variant=='y-base' else 'ttcn', d_model=8,
        te_dim=4, n_heads=2, n_layers=1, dropout=0., leader_num=1,
        state_num=2, lag_estimator='pearson', max_lag=5, min_pairs=12,
        min_correlation=.8, refiner_mode=mode)


def batch():
    generator = torch.Generator().manual_seed(312)
    x = torch.randn(2, 32, 2, 2, generator=generator)*.1+.5
    # Scalar mean Y is an exact three-bin lag of X0.
    y = torch.roll(x[..., :1],3,1)
    history = torch.cat([y,x],-1)
    future = torch.randn(2,8,2,2,generator=generator)*.1+.5
    ht=(torch.arange(32)[None,:,None,None]+torch.tensor([.2,.8])[None,None,:,None])/40
    ft=(32+torch.arange(8)[None,:,None,None]+torch.tensor([.2,.8])[None,None,:,None])/40
    return dict(observed_data=history,observed_tp=ht.expand_as(history).clone(),
        observed_mask=torch.ones_like(history,dtype=torch.bool),
        future_exog=future,future_exog_tp=ft.expand_as(future).clone(),
        future_exog_mask=torch.ones_like(future,dtype=torch.bool),
        query_patch_indices=torch.arange(8)[None].expand(2,-1),
        tp_to_predict=(32+torch.arange(8)[None].expand(2,-1)+.4)/40,
        data_to_predict=torch.rand(2,8,1,generator=generator),
        mask_predicted_data=torch.ones(2,8,1,dtype=torch.bool))


@pytest.mark.parametrize('variant',['y-base','patch-base','patch-lift'])
def test_forward_hidden_nan_label_independence_and_reload(variant):
    torch.manual_seed(13)
    c=settings(variant); model=build_model(c).train(); b=batch()
    b['observed_mask'][:,5:7]=False
    b['future_exog_mask'][:,4:6]=False
    original,aux=model(b)
    assert original.shape==(2,8,1) and aux.item()==0
    for value,time,mask in [('observed_data','observed_tp','observed_mask'),
                             ('future_exog','future_exog_tp','future_exog_mask')]:
        b[value][~b[mask]]=float('nan'); b[time][~b[mask]]=float('nan')
    b['data_to_predict'].fill_(float('nan')); b['mask_predicted_data'].fill_(False)
    changed,_=model(b)
    torch.testing.assert_close(changed,original)
    original.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    restored=build_model(ModelConfig.from_saved(asdict(c))).eval()
    restored.load_state_dict(model.state_dict())
    model.eval()
    torch.testing.assert_close(restored(b)[0],model(b)[0])


@pytest.mark.parametrize('variant',['y-base','patch-base'])
def test_base_reads_y_only_and_has_no_exogenous_or_auxiliary_modules(variant):
    model=build_model(settings(variant)).eval(); b=batch()
    first=model(b)[0]
    b['observed_data'][...,1:]=float('nan')
    b['observed_tp'][...,1:]=float('nan')
    del b['future_exog']; del b['future_exog_tp']; del b['future_exog_mask']
    del b['data_to_predict']; del b['mask_predicted_data']
    torch.testing.assert_close(model(b)[0],first)
    assert not any(name in dict(model.named_children()) for name in
        ['x_encoder','refiner','head_x','reconstruct','cov_history','cov_future'])


def test_patch_pair_has_identical_initial_y_backbone_and_separate_x_encoder():
    torch.manual_seed(99); bare=build_model(settings('patch-base'))
    torch.manual_seed(99); full=build_model(settings('patch-lift'))
    for key,value in bare.state_dict().items():
        torch.testing.assert_close(full.state_dict()[key],value,rtol=0,atol=0)
    yp={id(p) for p in full.y_encoder.parameters()}
    assert not yp & {id(p) for p in full.x_encoder.parameters()}
    assert not full.refiner.detach_leaders


@pytest.mark.parametrize('mode',['prefix','global'])
def test_main_forecast_trains_x_encoder_mapping_and_observed_future_x(mode):
    torch.manual_seed(13); model=build_model(settings(mode=mode)).train(); b=batch()
    b['future_exog'].requires_grad_()
    pred,aux=model(b)
    loss=(pred-b['data_to_predict']).square().mean()
    loss.backward()
    assert aux.item()==0
    for module in [model.y_encoder,model.x_encoder,model.x_to_y,model.refiner]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and sum(g.abs().sum() for g in grads)>0
    assert b['future_exog'].grad is not None and b['future_exog'].grad.abs().sum()>0


def test_scalar_lags_ignore_encoder_weights_timestamp_features_and_future_fields():
    torch.manual_seed(13); model=build_model(settings()).eval(); b=batch()
    leads=model.estimate_lags(b)
    assert leads['leader_lag'].eq(3).all() and leads['leaders'].eq(0).all()
    with torch.no_grad():
        for p in model.parameters(): p.uniform_(-5,5)
    b['observed_tp']*=20
    for key in list(b):
        if not key.startswith('observed_'): del b[key]
    after=model.estimate_lags(b)
    for key in leads: torch.testing.assert_close(leads[key],after[key],rtol=0,atol=0)
    c=settings(); c.lag_estimator='fft'; fftmodel=build_model(c)
    means,mask=scalar_patch_means(b['observed_data'],b['observed_mask'])
    expected=estimate_fft_leads(means[:,:1,:,None],means[:,1:,:,None],mask[:,:1],mask[:,1:],1)
    for key,value in fftmodel.estimate_lags(b).items(): torch.testing.assert_close(value,expected[key])


def test_constant_raw_x_with_unequal_point_counts_has_no_leader():
    c=settings(); c.lag_estimator='fft'; model=build_model(c); b=batch()
    b['observed_data'][...,1:]=.1234567
    b['observed_mask'][:,::3,1,1:]=False
    assert not model.estimate_lags(b)['valid'].any()


@pytest.mark.parametrize('mode',['prefix','global'])
def test_invalid_leaders_exact_fallback_empty_y_history_and_missing_x(mode):
    model=build_model(settings(mode=mode)).eval(); b=batch()
    b['observed_data'][...,1:]=.5
    history,mask=model.encode_y(b)
    assert not model.estimate_lags(b)['valid'].any()
    torch.testing.assert_close(model.forward_grid(b)[0],model.base_prediction(history,mask))
    b['observed_mask'][:]=False; b['observed_data'][:]=float('nan'); b['observed_tp'][:]=float('nan')
    assert torch.isfinite(model(b)[0]).all()


def test_prefix_preserves_lag_alignment_and_base_excludes_x():
    torch.manual_seed(13); model=build_model(settings(mode='prefix')).eval(); b=batch()
    before=model(b)[0]
    h,m=model.encode_y(b); base=model.base_prediction(h,m)
    b['future_exog'][:,1,:,0]+=2
    after=model(b)[0]
    torch.testing.assert_close(before[:,:4],after[:,:4],atol=1e-6,rtol=0)
    assert (before[:,4:]-after[:,4:]).abs().max()>1e-7
    h,m=model.encode_y(b); torch.testing.assert_close(base,model.base_prediction(h,m))


def test_y_base_uses_fixed_means_while_main_patch_path_keeps_ttcn(monkeypatch):
    import exobase.y_forecast as module
    b=batch(); base=build_model(settings('y-base')).eval()
    first=base(b)[0]
    b['observed_data'][:,:,:,0]=b['observed_data'][:,:,:,0].flip(-1)
    torch.testing.assert_close(first,base(b)[0])
    patch=build_model(settings('patch-base')).eval()
    def forbidden(*args): raise AssertionError('scalar mean entered TTCN main path')
    monkeypatch.setattr(module,'scalar_patch_means',forbidden)
    assert torch.isfinite(patch(b)[0]).all()
