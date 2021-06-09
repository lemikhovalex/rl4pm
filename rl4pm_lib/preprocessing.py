import pandas as pd
import torch
from sklearn.preprocessing import LabelBinarizer


def get_t_w(df: pd.DataFrame):
    _df = df.copy()
    _dt_s_mn = _df['timestamp'].apply(lambda x: (x - x.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())
    _dt_s_mn += _df['timestamp'].apply(lambda x: x.weekday() * 24 * 60 * 60)
    return _dt_s_mn.values


def get_t_e(df: pd.DataFrame):
    cp = df[['timestamp', 'trace_id']].copy()
    cp['te'] = cp['timestamp'].diff()
    start_of_trace = (cp['trace_id'].diff() != 0.)
    cp.loc[start_of_trace, 'te'] = 0.
    cp['te'] = cp['te'].apply(lambda x: x.total_seconds() if type(x) != float else x)
    return cp['te'].values


def get_t_t(df: pd.DataFrame):
    traces = list(set(df['trace_id']))
    out = df.copy()[['timestamp', 'trace_id']]
    t_ts = {}
    for t in traces:
        t_ts[t] = out['timestamp'][out['trace_id'] == t].min()
    out['tt'] = out.apply(lambda x: (x['timestamp'] - t_ts[x['trace_id']]).total_seconds(), axis=1)
    return out['tt'].values


def scale_tw(df: pd.DataFrame):
    return 24 * 60 * 60.


def scale_tt(df):
    traces = list(set(df['trace_id'].values))
    max_tt = 0
    for t_id in traces:
        loc_df = df[df['trace_id'] == t_id]
        _max_tt = loc_df['tt'].max()
        if _max_tt > max_tt:
            max_tt = _max_tt
    # df['tt'] = df['tt'] / max_tt
    return max_tt


def scale_te(df: pd.DataFrame):
    traces = list(set(df['trace_id'].values))
    max_te = 0
    for t_id in traces:
        loc_df = df[df['trace_id'] == t_id]['te']
        _max_te = loc_df.diff().dropna().max()
        if _max_te > max_te:
            max_te = _max_te
    # df['te'] = df['te'] / max_te
    return max_te


class AllInOnePrepro:
    def __init__(self):
        self.df_prepro = DfPreprocesser()
        self.paper_prepro = PaperScaler()


class DfPreprocesser:
    def __init__(self):
        self.core_oh = LabelBinarizer()
        self.n_classes = 0

    def fit(self, df: pd.DataFrame):
        assert 'activity' in df.columns.values
        self.core_oh.fit(df['activity'])
        self.n_classes = len(set(df['activity']))

    def transform(self, df: pd.DataFrame, inplace=False):
        if not inplace:
            out = df.copy()
        else:
            out = df

        tw = get_t_w(out)
        te = get_t_e(out)
        tt = get_t_t(out)

        oh_np = self.core_oh.transform(out['activity'])

        oh = pd.DataFrame(oh_np, columns=self.core_oh.classes_)
        out = pd.concat([out, oh], axis=1)

        out.drop(columns=['activity'], inplace=True)

        time_related_df = pd.DataFrame({'tt': tt,
                                        'te': te,
                                        'tw': tw
                                        })
        out = pd.concat([time_related_df, out], axis=1)
        return out


class PaperScalerPd:
    def __init__(self, column_features=None, drop_useless=True):
        if column_features is None:
            column_features = {'te': 0, 'tt': 1, 'tw': 2}
        self.column_features = column_features
        self.scales = {'te': 1., 'tt': 1., 'tw': 2.}
        self.cols_to_scale = {}
        self.drop_useless = drop_useless

    def fit(self, df: pd.DataFrame, y=None):
        cols = df.columns.values
        assert 'trace_id' in cols

        self.scales['tw'] = scale_tw(df)
        self.scales['te'] = scale_te(df)
        self.scales['tt'] = scale_tt(df)

        for _cf in self.column_features:
            self.cols_to_scale[_cf] = []

        for _cf in self.column_features:
            for _col in df.columns.values:
                if _col[:len(_cf)] == _cf:
                    self.cols_to_scale[_cf].append(_col)

        return self

    def transform(self, x: pd.DataFrame, inplace=False):
        if not inplace:
            out = x.copy()
        else:
            out = x
        if self.drop_useless:
            if 'trace_id' in out.columns.values:
                out.drop(columns=['trace_id'], inplace=True)
            if 'timestamp' in out.columns.values:
                out.drop(columns=['timestamp'], inplace=True)

        for _cf in self.column_features:
            for _col in self.cols_to_scale[_cf]:
                out[_col] = out[_col] / self.scales[_cf]
        return out


class PaperScaler:
    def __init__(self, column_features=None):
        if column_features is None:
            column_features = {'te': 0, 'tt': 1, 'tw': 2}
        self.column_features = column_features
        self.scales = {'te': 1., 'tt': 1., 'tw': 1.}

    def fit(self, df: pd.DataFrame, y=None):
        cols = df.columns.values
        assert 'trace_id' in cols
        assert 'te' in cols

        self.scales['tw'] = scale_tw(df)
        self.scales['te'] = scale_te(df)
        self.scales['tt'] = scale_tt(df)

        return self

    def transform(self, x: torch.tensor, inplace=True):
        if not inplace:
            out = x.clone()
        else:
            out = x

        if len(x.shape) == 3:
            for _cf in self.column_features:
                out[:, :, self.column_features[_cf]] = out[:, :, self.column_features[_cf]] / self.scales[_cf]
        if type(out) == pd.DataFrame:
            for _cf in self.column_features:
                out[_cf] = out[_cf] / self.scales[_cf]
        return out
    
def is_there_cycle(x) -> bool:
    out = None
    if len(x) < 3:
        out = False
    else:
        out = False
        _i = 0
        l = len(x)
        while (not out) and (_i + 2 <= l - 1):
            if (x[_i] == x[_i+2]) or (x[_i] == x[_i+1]):
                out = True
            _i += 1
    return out


def number_of_cycles(x) -> int:
    out = 0
    if len(x) < 3:
        out = 0
    else:
        out = 0
        _i = 0
        l = len(x)
        while (_i + 2 <= l - 1):
            if (x[_i] == x[_i+2]) or (x[_i] == x[_i+1]):
                out += 1
            _i += 1
    return out