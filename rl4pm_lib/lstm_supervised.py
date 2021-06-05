from torch.utils.data import Dataset
from rl4pm_lib.utils_supervised import make_window_features
from rl4pm_lib.utils import fill_trace

import pandas as pd
import torch


class ProcessesDataset(Dataset):
    def __init__(self, df, win_len):
        self.win_len = win_len
        self.df_win, self.labels, self.tes = make_window_features(df, win_len)
        self.df_win.drop(columns=[f'trace_id__{_w + 1}' for _w in range(win_len - 1)], inplace=True)
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


def prepro_batch_from_loader(data_dict: dict, device):
    x = data_dict['data'].to(device)
    input_size = x.shape[-1]
    _bs = x.shape[0]  # define btch size (it can be not as defined for residula piece of data)
    x = x.transpose(0, 1).view(-1, _bs, input_size)

    is_done = data_dict['is_done']
    is_done = is_done.transpose(0, 1).view(-1, _bs, 1)

    true_label = data_dict['label'].to(device)
    true_label = true_label.transpose(0, 1).view(-1, _bs, 1)

    true_tes = data_dict['tes'].to(device)
    true_tes = true_tes.transpose(0, 1).view(-1, _bs, 1)

    return {'data': x, 'is_done': is_done, 'label': true_label, 'tes': true_tes}


def train_one_epoch(dataloader, device,
                    loss_ac, loss_te,
                    optimizer,
                    model: torch.nn.Module, n_classes):
    total_true_labels = None
    total_pred_labels = None
    total_pred_tes = None
    total_true_tes = None
    model.to(device)
    for batch, data_dict in enumerate(dataloader):
        data_dict = prepro_batch_from_loader(data_dict, device)

        x = data_dict['data']
        is_done = data_dict['is_done']
        true_labels = data_dict['label']
        true_tes = data_dict['tes']

        pred_tes, pred_labels = model(x)

        # whanna drop useless padded -1, which are stored in is_done
        is_done = is_done.reshape(-1).bool()
        true_labels = true_labels.reshape(-1)[is_done]
        pred_labels = pred_labels.reshape((-1, n_classes))[is_done]

        pred_tes = pred_tes.reshape(-1)[is_done]
        true_tes = true_tes.reshape(-1)[is_done]

        # ok let's calc losses

        # loss_ac_ = loss_ac(pred_label, true_label.long())
        optimizer.zero_grad()

        loss_te_ = loss_te(pred_tes, true_tes)
        loss_te_.backward(retain_graph=True)

        loss_ac_ = loss_ac(pred_labels, true_labels.long())
        loss_ac_.backward()

        optimizer.step()

        if total_true_labels is None:
            total_true_labels = true_labels.data.cpu()
            total_pred_labels = pred_labels.data.cpu()
            total_pred_tes = pred_tes.data.cpu()
            total_true_tes = true_tes.data.cpu()
        else:
            total_true_labels = torch.cat([total_true_labels, true_labels.data.cpu()], dim=0)
            total_pred_labels = torch.cat([total_pred_labels, pred_labels.data.cpu()], dim=0)
            total_pred_tes = torch.cat([total_pred_tes, pred_tes.data.cpu()], dim=0)
            total_true_tes = torch.cat([total_true_tes, true_tes.data.cpu()], dim=0)
    return {'true_label': total_true_labels.long(), 'pred_label': total_pred_labels,
            'true_tes': total_true_tes, 'pred_tes': total_pred_tes
            }


def for_evaluate(dataloader, model, device, n_classes):
    total_true_labels = None
    total_pred_labels = None
    total_pred_tes = None
    total_true_tes = None

    model.to(device)
    for batch, data_dict in enumerate(dataloader):
        data_dict = prepro_batch_from_loader(data_dict, device)
        x = data_dict['data']
        is_done = data_dict['is_done']
        true_labels = data_dict['label']
        true_tes = data_dict['tes']

        with torch.no_grad():
            pred_tes, pred_labels = model(x)

            # whanna drop useless padded -1, which are stored in is_done
            is_done = is_done.reshape(-1).bool()
            true_labels = true_labels.reshape(-1)[is_done]
            pred_labels = pred_labels.reshape((-1, n_classes))[is_done]

            pred_tes = pred_tes.reshape(-1)[is_done]
            true_tes = true_tes.reshape(-1)[is_done]
        if total_true_labels is None:
            total_true_labels = true_labels.cpu().data
            total_pred_labels = pred_labels.cpu().data
            total_pred_tes = pred_tes.cpu().data
            total_true_tes = true_tes.cpu().data
        else:
            total_true_labels = torch.cat([total_true_labels, true_labels.data], dim=0)
            total_pred_labels = torch.cat([total_pred_labels, pred_labels.data], dim=0)
            total_pred_tes = torch.cat([total_pred_tes, pred_tes.data], dim=0)
            total_true_tes = torch.cat([total_true_tes, true_tes.data], dim=0)
        return {'true_label': total_true_labels.long(), 'pred_label': total_pred_labels,
                'true_tes': total_true_tes, 'pred_tes': total_pred_tes
                }
