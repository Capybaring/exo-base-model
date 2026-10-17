"""Prepare local data without altering sources or publishing datasets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DIST_EXOG, DIST_TARGETS, load_long_table


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_cache(source, destination, targets, exog):
    table = load_long_table(source, targets, exog)
    np.savez_compressed(destination, time=table.time, values=table.values,
        mask=table.mask, names=np.asarray(table.names), duplicate_rows=table.duplicate_rows)
    return {'cache': str(destination.resolve()), 'shape': list(table.values.shape),
        'observed_by_variable': table.mask.sum(0).tolist(), 'variables': list(table.names),
        'duplicates_merged': table.duplicate_rows, 'time_range': [float(table.time[0]), float(table.time[-1])]}


def thin_csv(source, destination, drop_rate=.3, seed=2021):
    """Bernoulli deletion of individual variable observations, not whole rows of a pivot.

    Deletion applies throughout the stream, so labels also become irregular.
    Query metrics use only retained labels. Original timestamps are preserved.
    """
    if not 0 <= drop_rate < 1:
        raise ValueError('drop_rate must lie in [0,1)')
    rng = np.random.default_rng(seed)
    stats = {}
    destination = Path(destination)
    if destination.resolve() == Path(source).resolve():
        raise ValueError('Refusing to overwrite the source CSV')
    destination.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for chunk in pd.read_csv(source, chunksize=250000):
        kept = rng.random(len(chunk)) >= drop_rate
        for name, group in chunk.groupby('cols', sort=False):
            s = stats.setdefault(str(name), {'original': 0, 'retained': 0})
            s['original'] += len(group)
            s['retained'] += int(kept[group.index.to_numpy()-chunk.index[0]].sum())
        chunk.loc[kept].to_csv(destination, index=False, mode='w' if first else 'a', header=first)
        first = False
    for s in stats.values():
        s['drop_fraction'] = 1-s['retained']/s['original']
    return {'seed': seed, 'requested_drop_rate': drop_rate, 'mode': 'independent observation Bernoulli deletion',
        'scope': 'all selected variables across train/val/test; original sources untouched', 'counts': stats}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dist')
    p.add_argument('--dataset-dir', help='DAG dataset directory or its forecasting subdirectory')
    p.add_argument('--output-dir', default='data/processed')
    p.add_argument('--drop-rate', type=float, default=.3)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)
    if not args.dist and not args.dataset_dir:
        p.error('Provide --dist and/or --dataset-dir')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    planned = ([] if not args.dist else [output/'DIST.npz'])
    if args.dataset_dir:
        planned += [output/f'{n}{s}' for n in ('BE', 'NP') for s in ('.npz', '_irregular.csv')]
    if not args.overwrite and any(p.exists() for p in planned):
        p.error('Prepared outputs already exist; use another directory or --overwrite')
    report = {'datasets': {}, 'seed': args.seed, 'drop_rate': args.drop_rate}
    if args.dist:
        source = Path(args.dist)
        print('Preparing DIST observations (no additional deletion)', flush=True)
        report['datasets']['DIST'] = {'source': str(source.resolve()), 'source_sha256': sha256(source),
            **write_cache(source, output/'DIST.npz', DIST_TARGETS, DIST_EXOG)}
    if args.dataset_dir:
        directory = Path(args.dataset_dir)
        if (directory/'forecasting').is_dir():
            directory = directory/'forecasting'
        for name in ('BE', 'NP'):
            source = directory/f'{name}.csv'
            names = list(pd.read_csv(source, usecols=['cols'])['cols'].unique())
            if names[-1] != 'OT':
                raise ValueError(f'{name}: expected OT as the DAG target, found {names[-1]}')
            destination = output/f'{name}_irregular.csv'
            deletion = thin_csv(source, destination, args.drop_rate, args.seed)
            print('Preparing', name, deletion, flush=True)
            report['datasets'][name] = {'source': str(source.resolve()), 'source_sha256': sha256(source),
                'deletion': deletion, 'targets': ['OT'], 'exog': names[:-1],
                **write_cache(destination, output/f'{name}.npz', ['OT'], names[:-1])}
    (output/'preparation_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Saved:', output/'preparation_manifest.json', flush=True)
    return report


def cli():
    main()


if __name__ == '__main__':
    main()
