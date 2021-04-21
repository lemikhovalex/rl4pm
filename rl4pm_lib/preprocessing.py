import pandas as pd
import torch
from sklearn.preprocessing import LabelBinarizer


def get_t_w(df):
    _df = df.copy()
    _dt_s_mn = _df['timestamp'].apply(lambda x: (x - x.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())
    _dt_s_mn += _df['timestamp'].apply(lambda x: x.weekday() * 24 * 60 * 60)
    return _dt_s_mn.values


def get_t_e(df):
    te = df['timestamp'].copy().diff()
    tr_diff = df['trace_id'].diff().fillna(1)
    te[tr_diff != 0] = 0.
    return te.values * 1e-9


def get_t_t(df):
    traces = list(set(df['trace_id']))
    out = df.copy()[['timestamp', 'trace_id']]
    t_ts = {}
    for t in traces:
        t_ts[t] = df['timestamp'][df['trace_id'] == t].min()
    out['tt'] = out.apply(lambda x: (x['timestamp'] - t_ts[x['trace_id']]).total_seconds(), axis=1)
    return out['tt'].values


def scale_tw(df):
    # df['tw'] = df['tw'] / (24 * 60 * 60)
    return 24 * 60 * 60.


def scale_tt(df):
    traces = list(set(df['trace_id'].values))
    max_tt = 0
    for t_id in traces:
        loc_df = df[df['trace_id'] == t_id]
        max_time = loc_df['timestamp'].max()
        min_time = loc_df['timestamp'].min()
        _max_tt = (max_time - min_time).total_seconds()
        if _max_tt > max_tt:
            max_tt = _max_tt
    # df['tt'] = df['tt'] / max_tt
    return max_tt


def scale_te(df):
    traces = list(set(df['trace_id'].values))
    max_te = 0
    for t_id in traces:
        loc_df = df[df['trace_id'] == t_id]['te']
        _max_te = loc_df.diff().dropna().max()
        if _max_te > max_te:
            max_te = _max_te
    # df['te'] = df['te'] / max_te
    return max_te


class DfPreprocesser:
    def __init__(self):
        self.core_oh = LabelBinarizer()
        self.tw = None
        self.te = None
        self.tt = None
        self.n_classes = 0

    def fit(self, df: pd.DataFrame):
        assert 'activity' in df.columns.values
        self.core_oh.fit(df['activity'])

        self.tw = get_t_w(df)
        self.te = get_t_e(df)
        self.tt = get_t_t(df)

    def transform(self, df: pd.DataFrame, inplace=False):
        if not inplace:
            out = df.copy()
        else:
            out = df
        oh_np = self.core_oh.transform(out['activity'])
        oh = pd.DataFrame(oh_np, columns=self.core_oh.classes_)
        out = pd.concat([df, oh], axis=1)
        out.drop(columns=['activity', 'timestamp'], inplace=True)

        self.n_classes = len(set(df['activity']))
        time_related_df = pd.DataFrame({'tt': self.tt,
                                        'te': self.te,
                                        'tw': self.tw
                                        })
        out = pd.concat([time_related_df, out], axis=1)
        return out


class PaperScaler:
    def __init__(self, column_features=None):
        if column_features is None:
            column_features = {'te': 0, 'tt': 1, 'tw': 2}
        self.column_features = column_features
        self.scales = {'te': 1., 'tt': 1., 'tw': 2.}

    def fit(self, df: pd.DataFrame):
        cols = df.columns.values
        assert 'timestamp' in cols
        assert 'trace_id' in cols
        assert 'te' in cols

        self.scales['tw'] = scale_tw(df)
        self.scales['te'] = scale_te(df)
        self.scales['tt'] = scale_tt(df)

    def transform(self, x: torch.tensor, inplace=True):
        if not inplace:
            out = x.clone()
        else:
            out = x
        for _cf in self.column_features:
            out[:, :, self.column_features[_cf]] = out[:, :, self.column_features[_cf]] / self.scales[_cf]

