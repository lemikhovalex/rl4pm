from torch.utils.data import Dataset
from rl4pm_lib.utils_supervised import make_window_features
from rl4pm_lib.utils import fill_trace

import pandas as pd
import torch


class ProcessesDataset(Dataset):
    def __init__(self, df, win_len):
        self.win_len = win_len
        self.df_win, self.labels, self.tes = make_window_features(df, win_len)
        self.df_win.drop(columns=[f'trace_id__{_w+1}' for _w in range(win_len - 1)], inplace=True)
        self.labels = pd.DataFrame({'trace_id': self.df_win['trace_id'],
                                    'label': self.labels
                                    }
                                   )
        self.tes = pd.DataFrame({'trace_id': self.df_win['trace_id'],
                                 'te': self.tes
                                 }
                                )
        # figure out what is max leen for this dataset
        self.max_len = 0
        self.ids_set = set(df['trace_id'].values)
        for _id in self.ids_set:
            _len = df[df['trace_id'] == _id].shape[0]
            if _len > self.max_len:
                self.max_len = _len

        # create init tensors as zeros
        # -1 for columns with trace_id
        self.tensor_data = torch.zeros(len(self.ids_set), self.max_len, self.df_win.shape[1] - 1)
        self.tensor_labels = torch.zeros(len(self.ids_set), self.max_len, 1)
        self.tensor_tes = torch.zeros(len(self.ids_set), self.max_len, 1)
        self.is_done = torch.ones(len(self.ids_set), self.max_len, 1)

        for _i, i in enumerate(self.ids_set):
            # select data for the trace id
            _np_d = self.df_win[self.df_win['trace_id'] == i].drop(columns=['trace_id']).values
            _np_l = self.labels[self.labels['trace_id'] == i].drop(columns=['trace_id']).values
            _np_tes = self.tes[self.tes['trace_id'] == i].drop(columns=['trace_id']).values

            self.tensor_data[_i] = torch.as_tensor(fill_trace(_np_d, self.max_len))

            self.tensor_labels[_i] = torch.as_tensor(fill_trace(_np_l, self.max_len, fill_value=-1)).long()

            self.tensor_tes[_i] = torch.as_tensor(fill_trace(_np_tes, self.max_len, fill_value=-1))
            # if "answers" are equal to -1 then
            self.is_done[_i] = (self.tensor_tes[_i] != -1).bool()

        self.tensor_data = self.tensor_data.transpose(0, 1).view(self.max_len, len(self.ids_set), -1)
        self.is_done = self.is_done.transpose(0, 1).view(self.max_len, len(self.ids_set), -1)
        self.tensor_tes = self.tensor_tes.transpose(0, 1).view(self.max_len, len(self.ids_set), -1)
        self.tensor_labels = self.tensor_labels.transpose(0, 1).view(self.max_len, len(self.ids_set), -1)

    def __len__(self):
        return self.tensor_data.shape[1]

    def __getitem__(self, idx):

        data = self.tensor_data[:, idx, :]
        label = self.tensor_labels[:, idx, :]
        tes = self.tensor_tes[:, idx, :]
        is_done = self.is_done[:, idx, :]

        out = {"data": data, "label": label, 'tes': tes, 'is_done': is_done}
        return out
