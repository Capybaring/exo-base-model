"""Hourly origins, split isolation, unique-cell scaling and checkpoint replay."""
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exobase.data import (ObservationTable, TimeRecords, split_records,
                          records_from_settings, normalization_from_seen, observation_ranges)
from exobase.metrics import fit_evaluation_statistics

ROOT = Path(__file__).resolve().parents[1]
HOUR = 3600


def table(hours=1920):
    time = np.arange(hours, dtype=np.float64)*HOUR
    v = np.column_stack([np.arange(hours)**2, np.arange(hours)+1, np.arange(hours)+2]).astype(np.float32)
    return ObservationTable(time, v, np.ones_like(v, dtype=bool),
                            ('OT', ' Generation forecast', ' System load forecast'), 0)


def settings(**overrides):
    args = dict(history=168*HOUR, forecast=24*HOUR, targets=['OT'],
                sample_stride=HOUR, split='chronological', max_records=None)
    return SimpleNamespace(**(args | overrides))


def test_hourly_origins_preserve_clock_and_disjoint_forecast_splits():
    source = table()
    # Missing raw rows must not move any physical forecast origin.
    keep = np.arange(len(source.time)) % 11 != 3
    source.time, source.values, source.mask = source.time[keep], source.values[keep], source.mask[keep]
    records, splits = records_from_settings(source, settings())
    assert records.coverage_end == 1920*HOUR
    assert len(records) == 1920-192+1
    assert records[1][1][0] == 0  # sample starts at hour 1, not at row 192
    assert records.split_boundaries == [1152*HOUR, 1536*HOUR]
    assert records.boundary_crossing_count == 46
    target_sets = []
    for name, ids in splits.items():
        stamps = set()
        lower, upper = {'train': (0, 1152*HOUR), 'val': (1152*HOUR, 1536*HOUR),
                        'test': (1536*HOUR, 1920*HOUR)}[name]
        for i in ids:
            rid, a, h, b = records.records[i]
            assert rid*HOUR+records.history >= lower
            assert rid*HOUR+records.duration <= upper
            stamps.update(source.time[h:b])
        target_sets.append(stamps)
    assert all(not a.intersection(b) for i, a in enumerate(target_sets) for b in target_sets[i+1:])
    # Past context may cross a boundary; the labels never do.
    rid = records.records[splits['val'][0]][0]
    assert rid*HOUR < records.split_boundaries[0]
    with pytest.raises(ValueError, match='chronological'):
        split_records(records, 'official-random')


def test_unique_training_scalers_and_no_heldout_leakage():
    source = table()
    records, splits = records_from_settings(source, settings())
    stats = fit_evaluation_statistics(records, splits['train'])
    norm = normalization_from_seen(records, splits['train'])
    values = source.values[:1152, 0].astype(np.float64)
    assert stats['observed_count'] == [1152]
    assert stats['mean'][0] == pytest.approx(values.mean())
    assert stats['std'][0] == pytest.approx(values.std())
    assert observation_ranges(records, splits['train']) == [(0, 1152)]
    weighted = np.concatenate([records[i][2][:, 0] for i in splits['train']]).astype(np.float64)
    assert not np.isclose(stats['mean'][0], weighted.mean())
    source.values[1152:] = 1e10
    assert fit_evaluation_statistics(records, splits['train']) == stats
    assert normalization_from_seen(records, splits['train']) == norm


def test_legacy_default_and_rolling_smoke_partitions():
    source = table()
    old_settings = settings()
    del old_settings.sample_stride  # old saved manifests
    old, old_splits = records_from_settings(source, old_settings)
    direct = TimeRecords(source, 168*HOUR, 24*HOUR, 1)
    assert old.records == direct.records
    assert old_splits == split_records(direct, 'chronological')
    records, splits = records_from_settings(source, settings(max_records=8))
    assert [len(x) for x in splits.values()] == [4, 2, 2]
    assert records.records[splits['test'][0]][0]*HOUR+records.history == 1536*HOUR
    for stride in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='sample stride'):
            records_from_settings(source, settings(sample_stride=stride))


def test_rolling_patch_lift_train_reload_predict_and_runner(tmp_path):
    source = table()
    data = tmp_path/'BE.npz'
    np.savez(data, time=source.time, values=source.values, mask=source.mask,
             names=np.array(source.names), duplicate_rows=np.array(0))
    from exobase.train import main as train
    from exobase.evaluate import main as evaluate
    from exobase.predict import main as predict
    output = tmp_path/'trained'
    train(['--config', str(ROOT/'configs/BE.json'), '--data-path', str(data),
           '--variant', 'patch-lift', '--patch-seconds', '43200', '--patch-stride', '43200',
           '--sample-stride', '3600', '--history', '604800', '--forecast', '86400',
           '--device', 'cpu', '--epochs', '1', '--max-records', '8', '--d-model', '8',
           '--n-heads', '2', '--state-num', '2', '--threads', '1', '--save-path', str(output)])
    saved = json.loads((output/'manifest.json').read_text())
    assert saved['model_config']['history_patches'] == 14
    assert saved['model_config']['future_patches'] == 2
    assert 'sample_stride' not in saved['model_config']
    assert saved['forecast_task']['boundary_crossing_records_excluded'] == 46
    original = json.loads((output/'metrics.json').read_text())
    result = evaluate(['--checkpoint', str(output/'best_model.pt'), '--threads', '1'])
    assert result['test']['mse'] == pytest.approx(original['test']['mse'], rel=1e-6)
    predict(['--checkpoint', str(output/'best_model.pt'), '--threads', '1',
             '--output', str(output/'predictions.csv')])
    rows = list(csv.DictReader((output/'predictions.csv').open()))
    for row in rows:
        rid = int(row['record_id'])
        stamp = float(row['time_seconds'])
        assert (rid+168)*HOUR <= stamp < (rid+192)*HOUR
        assert float(row['actual']) == pytest.approx(float(source.values[int(stamp/HOUR), 0]), rel=1e-6)
    spec = importlib.util.spec_from_file_location('rolling_runner', ROOT/'scripts/run_all.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    args = runner.parser().parse_args(['--datasets', 'BE', '--data-dir', str(tmp_path),
                                      '--sample-stride', '3600', '--smoke', '--device', 'cpu'])
    plans = runner.preflight(ROOT, args, ['BE'])
    assert plans[0]['record_ids'] == saved['record_ids']
    jobs = runner.build_jobs(ROOT, args, plans, ['patch-lift'], tmp_path/'runner')
    command = jobs[0]['command']
    assert float(command[command.index('--sample-stride')+1]) == HOUR
