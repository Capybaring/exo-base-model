"""Long-table records and actual-time transformable patches.

Protocol reference: usail-hkust/t-PatchGNN, lib/{parse_datasets,physionet,utils}.py.
Values/masks are kept separate; no interpolation or regular resampling is used.
"""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


DIST_LOOPS = ('FIC05', 'FIC09', 'FIC11', 'LIC02', 'LIC03', 'LIC04', 'LIC05', 'LIC06', 'LIC12')
DIST_TARGETS = tuple(n + '.PV' for n in DIST_LOOPS)
DIST_EXOG = tuple(n + '.OP' for n in DIST_LOOPS)


@dataclass
class ObservationTable:
    time: np.ndarray
    values: np.ndarray
    mask: np.ndarray
    names: tuple
    duplicate_rows: int


def load_long_table(path, targets=DIST_TARGETS, exog=DIST_EXOG):
    """Read either date/data/cols or time/variable/value, numeric or datetime.

    Return the exact timestamp union ordered internally as [Y, X]. Duplicate
    cells use the official PhysioNet mean-reduction convention (reported).
    Original numeric time origins are preserved. Datetimes become elapsed seconds.
    """
    names = tuple(targets) + tuple(exog)
    if not targets or not exog or len(set(names)) != len(names):
        raise ValueError('Targets and exogenous names must be nonempty and disjoint')
    if Path(path).suffix == '.npz':
        with np.load(path, allow_pickle=False) as cached:
            source_names = cached['names'].tolist()
            indices = [source_names.index(n) for n in names]
            return ObservationTable(cached['time'], cached['values'][:, indices],
                cached['mask'][:, indices], names, int(cached['duplicate_rows']))
    header = list(pd.read_csv(path, nrows=0).columns)
    if set(('date', 'data', 'cols')).issubset(header):
        tc, vc, nc = 'date', 'data', 'cols'
    elif set(('time', 'value', 'variable')).issubset(header):
        tc, vc, nc = 'time', 'value', 'variable'
    else:
        raise ValueError('Expected date,data,cols or time,variable,value')
    pieces = {n: [] for n in names}
    datetime_mode = None
    with pd.read_csv(path, usecols=[tc, vc, nc], chunksize=250_000,
                     dtype={nc: 'category', vc: 'float64'}, float_precision='round_trip') as reader:
        for frame in reader:
            frame = frame.loc[frame[nc].isin(names)].copy()
            if frame.empty:
                continue
            numeric = pd.to_numeric(frame[tc], errors='coerce')
            is_datetime = bool(numeric.isna().any())
            if datetime_mode is not None and datetime_mode != is_datetime:
                raise ValueError('Mixed numeric and datetime time representations')
            datetime_mode = is_datetime
            if is_datetime:
                stamps = pd.to_datetime(frame[tc], utc=True, errors='raise', format='mixed')
                if stamps.isna().any():
                    raise ValueError('Missing timestamps')
                # Keep integer nanoseconds until the common origin is subtracted.
                frame['_time'] = stamps.astype('datetime64[ns, UTC]').astype('int64')
            else:
                frame['_time'] = numeric.astype('float64')
            if not np.isfinite(frame[['_time', vc]].to_numpy()).all():
                raise ValueError('Observed timestamps and values must be finite')
            for name, group in frame.groupby(nc, observed=True, sort=False):
                pieces[str(name)].append((group['_time'].to_numpy(), group[vc].to_numpy()))
    for name in names:
        if not pieces[name]:
            raise ValueError('No observations for variable: ' + name)
    return _assemble_long_table(pieces, names, datetime_mode)


def _assemble_long_table(pieces, names, datetime_mode):
    arrays = {}
    duplicates = 0
    origin = min(min(p[0].min() for p in pieces[n]) for n in names) if datetime_mode else 0
    for name in names:
        parts = pieces.pop(name)
        t = np.concatenate([p[0] for p in parts])
        v = np.concatenate([p[1] for p in parts])
        order = np.argsort(t, kind='stable')
        t, v = t[order], v[order]
        starts = np.r_[0, np.flatnonzero(t[1:] != t[:-1])+1]
        counts = np.diff(np.r_[starts, len(t)])
        duplicates += int(len(t)-len(starts))
        t, v = t[starts], np.add.reduceat(v, starts)/counts
        if datetime_mode:
            t = (t - origin).astype(np.float64) / 1e9
        arrays[name] = (t, v)
    union = np.unique(np.concatenate([arrays[n][0] for n in names]))
    values = np.zeros((len(union), len(names)), dtype=np.float32)
    mask = np.zeros(values.shape, dtype=bool)
    for j, n in enumerate(names):
        t, v = arrays.pop(n)
        indices = np.searchsorted(union, t)
        values[indices, j] = v
        mask[indices, j] = True
    if not np.isfinite(values).all():
        raise ValueError('Values overflow float32')
    return ObservationTable(union, values, mask, names, duplicates)


class TimeRecords(Dataset):
    """Adapt one continuous series to disjoint t-PatchGNN-style records.

    Full records span [start, start+history+forecast). The final incomplete
    record and records without history/target observations are reported/skipped.
    No interpolation, gap compression, or overlapping train/test windows.
    """
    def __init__(self, table, history, forecast, target_dim):
        if history <= 0 or forecast <= 0:
            raise ValueError('history and forecast must be positive seconds')
        self.table, self.history, self.forecast = table, history, forecast
        self.target_dim = target_dim
        self.duration = history + forecast
        if table.time[0] < 0:
            raise ValueError('Relative seconds must be nonnegative')
        self.records = []
        self.skipped = 0
        for rid in range(int(table.time[-1] // self.duration)):
            start = rid * self.duration
            a, h, b = np.searchsorted(table.time, [start, start+history, start+self.duration])
            if not table.mask[a:h].any() or not table.mask[h:b, :target_dim].any():
                self.skipped += 1
                continue
            self.records.append((rid, int(a), int(h), int(b)))
        if len(self.records) < 5:
            raise ValueError('Need at least five full records with history and target observations')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rid, a, h, b = self.records[index]
        return (rid, self.table.time[a:b] - rid*self.duration,
                self.table.values[a:b], self.table.mask[a:b])


def split_records(records, protocol='official-random'):
    """Official parse_datasets: 80% seen, then 75% of seen for training."""
    if protocol == 'chronological':
        n = len(records)
        a, b = int(n*.6), int(n*.8)
        return {'train': list(range(a)), 'val': list(range(a, b)), 'test': list(range(b, n))}
    if protocol != 'official-random':
        raise ValueError('Unknown split protocol')
    seen, test = train_test_split(np.arange(len(records)), train_size=.8,
                                  random_state=42, shuffle=True)
    train, val = train_test_split(seen, train_size=.75, random_state=42, shuffle=False)
    return {'train': train.tolist(), 'val': val.tolist(), 'test': test.tolist()}


def normalization_from_seen(records, indices):
    """Match official get_data_min_max(seen_data): train+validation, not test."""
    dim = records.table.values.shape[1]
    low, high, time_max = np.full(dim, np.inf), np.full(dim, -np.inf), 0.
    for i in indices:
        _, t, v, m = records[i]
        time_max = max(time_max, float(t.max()))
        for j in range(dim):
            observed = v[m[:, j], j]
            if len(observed):
                low[j] = min(low[j], float(observed.min()))
                high[j] = max(high[j], float(observed.max()))
    if not np.isfinite(low).all() or not np.isfinite(high).all() or time_max <= 0:
        raise ValueError('Seen records must contain every selected variable and positive time')
    return {'data_min': low.tolist(), 'data_max': high.tolist(), 'time_max': time_max}


def pack_patches(records, start, duration, width, channels, stride=None):
    """Pack actual observations per variable per fixed-duration patch.

    Output is [B,K,L,C], identical to split_and_patch_batch's layout. L is the
    maximum observed count, not the union length; empty patches retain a slot.
    Membership depends on physical time, never on retained observation count.
    """
    stride = width if stride is None else stride
    if width <= 0 or not 0 < stride <= width:
        raise ValueError('Require 0 < patch stride <= patch width')
    k = math.ceil(max(duration-width, 0)/stride)+1
    slots, length = {}, 1
    for b, (_, t, v, mask) in enumerate(records):
        for p in range(k):
            lower, upper = start+p*stride, min(start+p*stride+width, start+duration)
            interval = (t >= lower) & (t < upper)
            for c, j in enumerate(channels):
                indices = np.flatnonzero(interval & mask[:, j])
                slots[b, p, c] = indices
                length = max(length, len(indices))
    shape = (len(records), k, length, len(channels))
    data, time, mask = np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, bool)
    for (b, p, c), ids in slots.items():
        _, tt, vv, _ = records[b]
        data[b, p, :len(ids), c] = vv[ids, channels[c]]
        time[b, p, :len(ids), c] = tt[ids]
        mask[b, p, :len(ids), c] = True
    return data, time, mask


class IrregularCollator:
    def __init__(self, history, forecast, patch_seconds, target_dim, normalization,
                 keep_rate=1., seed=1, patch_stride=None):
        if not 0 < patch_seconds <= min(history, forecast) or not 0 < keep_rate <= 1:
            raise ValueError('Invalid patch duration or keep rate')
        self.history, self.forecast, self.width = history, forecast, patch_seconds
        self.stride = patch_seconds if patch_stride is None else patch_stride
        if not 0 < self.stride <= self.width:
            raise ValueError('Require 0 < patch stride <= width')
        self.target_dim, self.keep_rate = target_dim, keep_rate
        self.low = np.asarray(normalization['data_min'])
        self.scale = np.asarray(normalization['data_max']) - self.low
        self.scale = self.scale + (self.scale == 0)*1e-8
        self.time_max = normalization['time_max']
        self.rng = np.random.RandomState(seed)

    def __call__(self, batch):
        normalized = []
        queries, labels, label_masks = [], [], []
        for rid, t, v, m in batch:
            v = np.where(m, (v-self.low)/self.scale, 0).astype(np.float32)
            future = t >= self.history
            queries.append(t[future]);labels.append(v[future, :self.target_dim])
            label_masks.append(m[future, :self.target_dim])
            m = m.copy()
            # Optional official-style historical TIMEPOINT subsampling. Future
            # X and target labels remain available under the chosen DAG protocol.
            if self.keep_rate < 1:
                ids = np.flatnonzero((t < self.history) & m.any(axis=1))
                retained = self.rng.choice(ids, int(len(ids)*self.keep_rate), replace=False)
                m[np.setdiff1d(ids, retained)] = False
            normalized.append((rid, t, v, m))
        dim = len(self.low)
        hist = pack_patches(normalized, 0, self.history, self.width, range(dim), self.stride)
        future_x = pack_patches(normalized, self.history, self.forecast, self.width,
                                range(self.target_dim, dim), self.stride)
        q = max(map(len, queries))
        qt = np.zeros((len(batch), q), np.float32)
        raw_query_time = np.zeros((len(batch), q), np.float64)
        y = np.zeros((len(batch), q, self.target_dim), np.float32)
        ym = np.zeros(y.shape, bool)
        qi = np.zeros(qt.shape, np.int64)
        for b, tt in enumerate(queries):
            n = len(tt)
            qt[b, :n] = tt/self.time_max
            raw_query_time[b, :n] = tt
            y[b, :n], ym[b, :n] = labels[b], label_masks[b]
            qi[b, :n] = np.minimum(np.floor((tt-self.history)/self.stride).astype(np.int64), future_x[0].shape[1]-1)
        result = {'observed_data': hist[0], 'observed_tp': hist[1]/self.time_max, 'observed_mask': hist[2],
                  'future_exog': future_x[0], 'future_exog_tp': future_x[1]/self.time_max,
                  'future_exog_mask': future_x[2], 'tp_to_predict': qt,
                  'data_to_predict': y, 'mask_predicted_data': ym, 'query_patch_indices': qi,
                  'record_ids': np.asarray([r[0] for r in batch], dtype=np.int64),
                  'query_time_seconds': raw_query_time,
                  'query_lengths': np.asarray([len(t) for t in queries], dtype=np.int64)}
        return {k: torch.from_numpy(v) for k, v in result.items()}
