"""Behavioral and upstream parity checks for the IMTS/exogenous interfaces."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from exobase.data import (ObservationTable, TimeRecords, IrregularCollator,
    load_long_table, pack_patches, split_records, normalization_from_seen)
from exobase.metrics import error_sums, variable_mean, masked_mse
from exobase.model import ExogenousTPatchGNN, ModelConfig, pattern_windows
from exobase.patterns import fit_pattern_bank
from exobase.prepare import thin_csv


@pytest.fixture
def example():
    rng = np.random.default_rng(3)
    t = np.arange(0, 60, .5)
    v = rng.normal(size=(len(t), 3)).astype(np.float32)
    m = np.ones_like(v, dtype=bool)
    m[::5, 0] = False
    table = ObservationTable(t, v, m, ('Y', 'X1', 'X2'), 0)
    records = TimeRecords(table, 4, 2, 1)
    norm = normalization_from_seen(records, list(range(len(records))))
    collate = IrregularCollator(4, 2, 1, 1, norm)
    return records, collate


def test_time_membership_zero_and_empty_patch():
    t = np.array([0., 1., 4., 9.])
    v = np.array([[0.], [1.], [4.], [9.]])
    mask = np.ones(v.shape, bool)
    x, times, valid = pack_patches([(0,t,v,mask)], 0, 12, 4, [0])
    assert valid.sum((2,3)).tolist() == [[2,1,1]]
    assert valid[0,0,0,0] and x[0,0,0,0] == 0
    assert times[0,1,0,0] == 4
    x, times, valid = pack_patches([(0,t,v,mask)], 0, 12, 2, [0])
    assert not valid[0,1].any()
    assert valid[0,4].any()  # the physical gap was not compressed away


def test_csv_union_duplicates_dates_and_cache(tmp_path):
    path = tmp_path/'raw.csv'
    pd.DataFrame({'date':['2020/1/1 0:00','2020/1/1 0:00','2020/1/1 1:00','2020/1/1 2:00'],
        'data':[0.,2.,3.,4.], 'cols':['Y','Y','X','Y']}).to_csv(path,index=False)
    table = load_long_table(path, ['Y'], ['X'])
    assert table.time.tolist() == [0.,3600.,7200.]
    assert table.duplicate_rows == 1 and table.values[0,0] == 1
    assert table.mask.tolist() == [[True,False],[False,True],[True,False]]


def test_upstream_patch_packing_parity():
    # Execute the pinned official pure function without its unrelated imports.
    source = Path('upstream/tpatchgnn/lib/utils.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'split_and_patch_batch')
    env = {'torch':torch, 'get_device':lambda data:data.device}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream>', 'exec'), env)
    t = np.array([0.,1.,3.,4.,5.,7.])
    values = np.arange(12).reshape(6,2).astype(np.float32)
    mask = np.array([[1,0],[0,1],[1,1],[0,1],[1,0],[1,1]],bool)
    data = {'data':torch.tensor(values)[None], 'time_steps':torch.tensor(t,dtype=torch.float32),
        'mask':torch.tensor(mask,dtype=torch.float32)[None], 'tp_to_predict':torch.tensor([8.]),
        'data_to_predict':torch.zeros(1,1,2), 'mask_predicted_data':torch.ones(1,1,2)}
    original = env['split_and_patch_batch'](data,SimpleNamespace(npatch=2),len(t),
        [torch.tensor([0,1,2]),torch.tensor([3,4,5])])
    packed = pack_patches([(0,t,values,mask)],0,8,4,[0,1])
    for key, value in zip(['observed_data','observed_tp','observed_mask'],packed):
        np.testing.assert_allclose(value, original[key].numpy())


def test_history_backbone_upstream_parity(example):
    from exobase.upstream_model import tPatchGNN
    records, collate = example
    batch = collate([records[0],records[1]])
    # Valid nonempty patches: safety changes must not alter upstream results.
    batch['observed_mask'].fill_(True)
    config = ModelConfig(1,2,4,variant='tpatch',d_model=8,te_dim=3,n_heads=2)
    model = ExogenousTPatchGNN(config).eval()
    args = SimpleNamespace(device='cpu',hid_dim=8,ndim=3,npatch=4,nlayer=1,
        te_dim=3,nhead=2,tf_layer=1,node_dim=10,hop=2,outlayer='Linear')
    reference = tPatchGNN(args).eval()
    reference.load_state_dict(model.backbone.state_dict())
    with torch.no_grad():
        actual, _ = model(batch)
        expected = reference.forecasting(batch['tp_to_predict'], batch['observed_data'],
            batch['observed_tp'],batch['observed_mask'].float())[0,...,:1]
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('variant',['tpatch-y','tpatch','exo','delay'])
def test_gradients_hidden_values_and_future_y_noninterference(example, variant):
    records, collate = example
    batch = collate([records[0],records[1]])
    config = ModelConfig(1,2,4,variant=variant,d_model=8,te_dim=3,n_heads=2,delay_context=2)
    model = ExogenousTPatchGNN(config, [[-1,1],[1,-1]] if variant == 'delay' else None)
    model.eval()
    base, _ = model(batch)
    changed = {k:v.clone() for k,v in batch.items()}
    changed['data_to_predict'].fill_(float('nan'))
    changed['mask_predicted_data'].fill_(False)
    changed['observed_data'][~changed['observed_mask']] = float('nan')
    changed['observed_tp'][~changed['observed_mask']] = float('nan')
    changed['future_exog'][~changed['future_exog_mask']] = float('nan')
    changed['future_exog_tp'][~changed['future_exog_mask']] = float('nan')
    result, _ = model(changed)
    torch.testing.assert_close(base,result)
    loss = masked_mse(base,batch['data_to_predict'],batch['mask_predicted_data'])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if variant == 'delay':
        assert model.memory.value.weight.grad.abs().sum() > 0
    if variant in ('tpatch-y','tpatch'):
        changed['future_exog'].fill_(12345.)
        torch.testing.assert_close(base,model(changed)[0])


def test_future_x_changes_prediction_and_all_missing_is_finite(example):
    records, collate = example
    batch = collate([records[0]])
    model = ExogenousTPatchGNN(ModelConfig(1,2,4,variant='exo',d_model=8,te_dim=3,n_heads=2)).eval()
    base = model(batch)[0]
    batch['future_exog'] += 5
    assert not torch.allclose(base,model(batch)[0])
    batch['future_exog_mask'].fill_(False)
    batch['future_exog'].fill_(float('nan'))
    assert torch.isfinite(model(batch)[0]).all()


def test_split_independence_and_train_only_pattern_bank(example):
    records, collate = example
    split = split_records(records)
    assert not set(split['train']) & set(split['test'])
    patterns, info = fit_pattern_bank(records,split['train'],collate,1,2,2,100,seed=3)
    for i in split['val']+split['test']:
        _,a,_,b = records.records[i]
        records.table.values[a:b] = 1e5
    again, _ = fit_pattern_bank(records,split['train'],collate,1,2,2,100,seed=3)
    np.testing.assert_allclose(patterns,again)
    assert set(info['train_record_ids']) == {records.records[i][0] for i in split['train']}


def test_metrics_accumulate_counts_not_batch_means():
    p = torch.tensor([[[1.,2.]],[[3.,9.]],[[4.,9.]]])
    y = torch.zeros_like(p)
    m = torch.tensor([[[1,1]],[[1,0]],[[1,0]]],dtype=torch.bool)
    sums, counts = error_sums(p,y,m)
    pieces = [error_sums(p[:1],y[:1],m[:1]),error_sums(p[1:],y[1:],m[1:])]
    torch.testing.assert_close(variable_mean(sums,counts),
        variable_mean(sum(v[0] for v in pieces),sum(v[1] for v in pieces)))
    expected = torch.tensor(((1+9+16)/3+4)/2)
    torch.testing.assert_close(variable_mean(sums,counts),expected)


@pytest.mark.parametrize('kind',['MSE','MAE','MAPE'])
def test_official_metric_parity(kind):
    source = Path('upstream/tpatchgnn/lib/evaluation.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'compute_error')
    env = {'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<upstream>','exec'),env)
    truth = torch.tensor([[[0.,.5],[.25,1.]],[[.75,.1],[1.,.2]]])
    pred = truth+.2
    mask = torch.tensor([[[1,0],[1,1]],[[0,1],[1,1]]],dtype=torch.bool)
    expected = env['compute_error'](truth,pred,mask,kind,'mean')
    actual = variable_mean(*error_sums(pred,truth,mask,kind.lower()))
    torch.testing.assert_close(actual,expected)


def test_random_deletion_reproducible_and_sources_unchanged(tmp_path):
    source = tmp_path/'raw.csv'
    frame = pd.DataFrame({'date':np.tile(np.arange(100),2),'data':np.arange(200),
        'cols':['Y']*100+['X']*100})
    frame.to_csv(source,index=False)
    before = source.read_bytes()
    a,b = tmp_path/'a.csv',tmp_path/'b.csv'
    thin_csv(source,a,.3,2021);thin_csv(source,b,.3,2021)
    assert a.read_bytes() == b.read_bytes() and source.read_bytes() == before
    result = pd.read_csv(a)
    assert 100 < len(result) < 190
    assert not np.array_equal(result.loc[result.cols=='Y','date'].values,
                              result.loc[result.cols=='X','date'].values)


def test_pattern_windows_preserve_gaps(example):
    records, collate = example
    batch = collate([records[0]])
    batch['observed_mask'][:,2,:,1:] = False
    windows, complete = pattern_windows(batch,1,2)
    assert not complete[:,:,2:4].any()
    assert windows.shape[-1] == 2
