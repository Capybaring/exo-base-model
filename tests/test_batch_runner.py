"""Batch experiment isolation, native time units, and real no-LIFT training."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_all', ROOT/'scripts/run_all.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def prepared_be(tmp_path):
    generator = np.random.default_rng(51)
    hours = np.arange(1921, dtype=np.float64)
    x = generator.normal(size=(len(hours), 2)).astype(np.float32)
    values = np.column_stack([np.roll(x[:, 0], 3), x]).astype(np.float32)
    mask = generator.random(values.shape) > .1
    np.savez(tmp_path/'BE.npz', time=hours*3600, values=values, mask=mask,
             names=np.array(['OT', ' Generation forecast', ' System load forecast']),
             duplicate_rows=np.array(0))
    return tmp_path


def test_checkpoint_recreates_removed_directory_and_records_absolute_path(tmp_path, monkeypatch):
    prepared_be(tmp_path)
    import exobase.train as train
    original_evaluate = train.evaluate_irregular
    output = tmp_path/'removed_during_epoch'
    removed = []

    def evaluate_and_remove(*args, **kwargs):
        metrics = original_evaluate(*args, **kwargs)
        if not removed:
            shutil.rmtree(output)
            removed.append(True)
        return metrics

    monkeypatch.setattr(train, 'evaluate_irregular', evaluate_and_remove)
    monkeypatch.chdir(tmp_path)
    train.main(['--config', str(ROOT/'configs/BE.json'), '--data-path', str(tmp_path/'BE.npz'),
                '--sample-stride', '43200', '--patch-seconds', '43200', '--patch-stride', '43200',
                '--device', 'cpu', '--threads', '1', '--epochs', '1', '--max-records', '8',
                '--d-model', '8', '--n-heads', '2', '--state-num', '2',
                '--save-path', 'removed_during_epoch'])
    assert removed and (output/'best_model.pt').exists()
    saved = json.loads((output/'manifest.json').read_text())
    assert saved['args']['save_path'] == str(output.resolve())
    assert (output/'metrics.json').exists() and (output/'test_lags.csv').exists()


def test_inventory_windows_and_matched_ablation_commands(tmp_path):
    args = runner.parser().parse_args(['--variants', 'dag-lift', 'dag-history'])
    datasets, variants = runner.selections(args)
    assert len(datasets) == 13 and len(set(datasets)) == 13
    assert 'DIST' not in runner.selections(runner.parser().parse_args(['--suite', 'table3']))[0]
    for name, step in [('BE', 3600), ('Sdwpfm1', 1800), ('Colbun', 86400), ('DIST', 3600)]:
        assert runner.STEPS[name] == step
        plan = {'dataset': name, 'config': '/config.json', 'data_path': '/data.npz',
                'history_seconds': 24*step, 'forecast_seconds': 24*step, 'step_seconds': step}
        jobs = runner.build_jobs(ROOT, args, [plan], variants, tmp_path)
        commands = [job['command'] for job in jobs]
        assert [c[c.index('--save-path')+1] for c in commands] == [
            str(tmp_path/name/'dag-lift'), str(tmp_path/name/'dag-history')]
        comparable = []
        for c in commands:
            flags = dict(zip(c[4::2], c[5::2]))
            assert flags['--history'] == str(24*step)
            assert flags['--forecast'] == str(24*step)
            assert flags['--patch-seconds'] == str(step)
            assert flags['--patch-encoder'] == 'ttcn'
            assert flags['--seed'] == '1'
            flags.pop('--variant'); flags.pop('--save-path')
            comparable.append(flags)
        assert comparable[0] == comparable[1]


def test_selection_rejects_conflicts_and_duplicates():
    args = runner.parser().parse_args(['--no-lift'])
    assert runner.selections(args)[1] == ['patch-base']
    for flags in [['--no-lift', '--variants', 'dag'], ['--datasets', 'BE', 'BE'],
                  ['--variants', 'dag-lift', 'dag-lift']]:
        with pytest.raises(ValueError):
            runner.selections(runner.parser().parse_args(flags))


def test_dry_run_preflight_and_no_outputs(tmp_path, capsys):
    data = prepared_be(tmp_path)
    output = tmp_path/'output'
    assert runner.main(['--datasets', 'BE', '--data-dir', str(data), '--device', 'cpu',
        '--variants', 'dag-lift', 'dag-history', '--dry-run', '--output-dir', str(output)]) == 0
    text = capsys.readouterr().out
    assert '--history 604800' in text and '--forecast 86400' in text
    assert '--task-protocol dataset-short' in text
    assert '--variant dag-history' in text and '--variant dag-lift' in text
    assert not output.exists()
    with pytest.raises(SystemExit):
        runner.main(['--datasets', 'BE', 'FR', '--data-dir', str(data), '--output-dir', str(output)])
    assert not output.exists()  # A missing later dataset must not start BE training.


def test_no_lift_does_not_require_fft_history_length(tmp_path):
    prepared_be(tmp_path)
    flags = ['--datasets', 'BE', '--data-dir', str(tmp_path), '--protocol', 'uniform', '--history-steps', '2',
             '--dry-run', '--device', 'cpu']
    assert runner.main(flags+['--no-lift']) == 0
    with pytest.raises(SystemExit):
        runner.main(flags)


def test_real_paired_smoke_training_and_summary(tmp_path):
    data = prepared_be(tmp_path)
    output = tmp_path/'paired'
    assert runner.main(['--datasets', 'BE', '--data-dir', str(data), '--device', 'cpu',
        '--variants', 'dag-lift', 'dag-history', '--smoke', '--threads', '1',
        '--output-dir', str(output)]) == 0
    rows = list(csv.DictReader((output/'summary.csv').open()))
    assert len(rows) == 2 and all(row['status'] == 'ok' for row in rows)
    full = json.loads((output/'BE/dag-lift/manifest.json').read_text())
    bare = json.loads((output/'BE/dag-history/manifest.json').read_text())
    assert full['record_ids'] == bare['record_ids']
    assert full['normalization'] == bare['normalization']
    assert full['data_sha256'] == bare['data_sha256']
    assert full['args']['seed'] == bare['args']['seed']
    assert (output/'BE/dag-lift/test_lags.csv').exists()
    assert not (output/'BE/dag-history/test_lags.csv').exists()
    import torch
    state = torch.load(output/'BE/dag-history/best_model.pt', weights_only=True)['state_dict']
    assert not any(key.startswith('refiner.') for key in state)
    comparison = list(csv.DictReader((output/'ablation_comparison.csv').open()))
    assert len(comparison) == 1 and comparison[0]['dataset'] == 'BE'
    assert float(comparison[0]['mse_delta_with_minus_without']) == pytest.approx(
        float(rows[0]['mse'])-float(rows[1]['mse']))
    # The default output protection must never overwrite the completed run.
    before = (output/'summary.csv').read_bytes()
    with pytest.raises(SystemExit):
        runner.main(['--datasets', 'BE', '--data-dir', str(data), '--device', 'cpu',
                     '--output-dir', str(output)])
    assert before == (output/'summary.csv').read_bytes()


def test_single_dataset_no_lift_alias_trains_without_refiner(tmp_path):
    data = prepared_be(tmp_path)
    from exobase.train import main
    output = tmp_path/'alias'
    main(['--config', str(ROOT/'configs/BE.json'), '--data-path', str(data/'BE.npz'),
          '--no-lift', '--history', '86400', '--forecast', '86400',
          '--patch-seconds', '3600', '--patch-stride', '3600', '--device', 'cpu',
          '--epochs', '1', '--max-records', '8', '--d-model', '8', '--n-heads', '2',
          '--state-num', '2', '--threads', '1', '--save-path', str(output)])
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['args']['no_lift'] and manifest['args']['variant'] == 'patch-base'
    assert manifest['model_config']['variant'] == 'patch-base'
    assert manifest['lag_protocol'] is None and not (output/'test_lags.csv').exists()


def test_simple_three_variant_training_reload_and_component_comparison(tmp_path):
    data=prepared_be(tmp_path); output=tmp_path/'simple'
    assert runner.main(['--datasets','BE','--data-dir',str(data),'--device','cpu',
        '--variants','y-base','patch-base','patch-lift','--smoke','--threads','1',
        '--output-dir',str(output)])==0
    rows=list(csv.DictReader((output/'summary.csv').open()))
    assert len(rows)==3 and all(r['status']=='ok' for r in rows)
    manifests=[json.loads((output/'BE'/v/'manifest.json').read_text()) for v in
               ['y-base','patch-base','patch-lift']]
    for m in manifests:
        assert m['architecture_version']==4 and m['args']['beta']==0
        assert m['record_ids']==manifests[0]['record_ids']
        assert m['normalization']==manifests[0]['normalization']
        assert m['patch_protocol']['auxiliary_targets'] is None
        assert m['args']['history']==168*3600 and m['args']['forecast']==24*3600
        assert m['forecast_task']['protocol']=='dataset-short'
        assert m['forecast_task']['window_alignment_only']
    assert rows[0]['protocol']=='dataset-short' and int(rows[0]['history_steps'])==168
    assert manifests[0]['args']['patch_encoder']=='mean'
    assert not manifests[1]['known_future_x'] and manifests[2]['known_future_x']
    assert not manifests[2]['lag_protocol']['leader_stream_detached']
    comparisons=list(csv.DictReader((output/'component_comparison.csv').open()))
    assert len(comparisons)==2
    pair=list(csv.DictReader((output/'ablation_comparison.csv').open()))
    assert pair[0]['with_lift_variant']=='patch-lift' and pair[0]['without_lift_variant']=='patch-base'
    from exobase.evaluate import main as evaluate
    from exobase.predict import main as predict
    folder=output/'BE'/'patch-lift'
    result=evaluate(['--checkpoint',str(folder/'best_model.pt'),'--threads','1'])
    original=json.loads((folder/'metrics.json').read_text())
    assert result['test']['mse']==pytest.approx(original['test']['mse'],rel=1e-6)
    assert 'future Y only' in result['evaluation_protocol']['training_objective']
    predict(['--checkpoint',str(folder/'best_model.pt'),'--threads','1',
             '--output',str(folder/'pred.csv'),'--lag-output',str(folder/'lags.csv')])
    assert (folder/'lags.csv').read_bytes()==(folder/'test_lags.csv').read_bytes()


def test_failed_job_is_reported_and_next_variant_still_runs(tmp_path, monkeypatch):
    prepared_be(tmp_path)
    calls = []
    def failure(command, root, log_path):
        calls.append(command)
        log_path.write_text('intentional test failure')
        return 7
    monkeypatch.setattr(runner, 'run_job', failure)
    output = tmp_path/'failed'
    assert runner.main(['--datasets', 'BE', '--data-dir', str(tmp_path), '--device', 'cpu',
        '--variants', 'dag-lift', 'dag-history', '--output-dir', str(output)]) == 1
    assert len(calls) == 2
    rows = list(csv.DictReader((output/'summary.csv').open()))
    assert all(row['status'] == 'failed: training exited with code 7' for row in rows)


def test_cli_can_run_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'scripts/run_all.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and '--no-lift' in result.stdout


def test_full_ttcn_execution_smoke_reload_and_incompatible_variants(tmp_path):
    data = prepared_be(tmp_path)
    output = tmp_path/'full'
    assert runner.main(['--datasets','BE','--data-dir',str(data),'--device','cpu',
        '--variants','patch-lift','--ttcn-execution','full','--smoke','--threads','1',
        '--output-dir',str(output)]) == 0
    folder = output/'BE'/'patch-lift'
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['args']['ttcn_execution'] == manifest['model_config']['ttcn_execution'] == 'full'
    from exobase.evaluate import main as evaluate
    result = evaluate(['--checkpoint',str(folder/'best_model.pt'),'--threads','1'])
    original = json.loads((folder/'metrics.json').read_text())
    assert result['test']['mse'] == pytest.approx(original['test']['mse'],rel=1e-6)
    assert (folder/'test_lags.csv').exists()
    with pytest.raises(SystemExit):
        runner.main(['--datasets','BE','--data-dir',str(data),'--device','cpu',
            '--variants','y-base','--ttcn-execution','full','--dry-run'])


@pytest.mark.parametrize('name,history,forecast,patch',[
    ('BE',168*3600,24*3600,3600),('NP',168*3600,24*3600,3600),
    ('DIST',7200,1800,60),('Colbun',60*86400,10*86400,86400),
    ('Sdwpfm1',168*1800,24*1800,1800)])
def test_dataset_short_durations_and_units(name,history,forecast,patch):
    args=runner.parser().parse_args([])
    _,_,task=runner.task_settings(ROOT,args,name)
    assert (task['history_seconds'],task['forecast_seconds'],task['step_seconds'])==(history,forecast,patch)
    assert task['history_steps']==history//patch and task['forecast_steps']==forecast//patch


def test_long_windows_and_dist_has_no_paper_long_preset():
    args=runner.parser().parse_args(['--protocol','dataset-long'])
    for name,history,forecast,patch in [('BE',720*3600,360*3600,3600),
                                      ('NP',720*3600,360*3600,3600),
                                      ('Colbun',180*86400,30*86400,86400)]:
        path,_,task=runner.task_settings(ROOT,args,name)
        assert path.name==name+'_long.json'
        assert (task['history_seconds'],task['forecast_seconds'],task['step_seconds'])==(history,forecast,patch)
    with pytest.raises(ValueError,match='DIST is not in DAG'):
        runner.task_settings(ROOT,args,'DIST')


def test_uniform_preserves_old_task_and_dataset_overrides_are_explicit(tmp_path,capsys):
    args=runner.parser().parse_args(['--protocol','uniform'])
    for name in ['DIST','BE','NP']:
        _,_,task=runner.task_settings(ROOT,args,name)
        assert task['history_seconds']==86400 and task['forecast_seconds']==86400
        assert task['step_seconds']==3600 and task['protocol']=='uniform'
    for protocol in ['dataset-short','dataset-long']:
        args=runner.parser().parse_args(['--protocol',protocol,'--history-steps','24'])
        with pytest.raises(ValueError,match='require --protocol uniform'):
            runner.task_settings(ROOT,args,'BE')
    data=prepared_be(tmp_path)
    assert runner.main(['--datasets','BE','--data-dir',str(data),'--protocol','uniform',
                        '--variants','dag','--dry-run','--device','cpu'])==0
    assert '--history 86400' in capsys.readouterr().out
