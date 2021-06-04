import pandas as pd
import numpy as np


def make_window_features_for_trace(df, win_len):
    _win_len = win_len - 1
    out = df[_win_len:].copy()
    out.reset_index(drop=True, inplace=True)
    sh = df.shape[0]
    for _i in range(win_len - 1):
        df_to_app = df[_i:sh - _win_len + _i].copy()

        rename_dict = {col: col + f'__{_i + 1}' for col in df_to_app.columns}
        df_to_app.rename(columns=rename_dict, inplace=True)
        df_to_app.reset_index(drop=True, inplace=True)

        out = pd.concat([out, df_to_app], axis=1)
    out.dropna(inplace=True)
    return out


def make_window_features(df, win_len):
    traces = list(set(df['trace_id'].values))
    outs = []
    labels = []
    tes = []
    for _i, trace in enumerate(traces):
        _df = df[df['trace_id'] == trace]
        outs.append(make_window_features_for_trace(_df, win_len)[:-1])  # one must left 4 prediction
        labels.append(_df.drop(columns=['te', 'tt', 'tw', 'trace_id']).values.argmax(axis=1)[win_len:])
        tes.append(_df['te'].values[win_len:])
    return pd.concat(outs, axis=0), np.concatenate(labels), np.concatenate(tes)
