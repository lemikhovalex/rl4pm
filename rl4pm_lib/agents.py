import torch
from torch.nn import functional as t_functional


class NetAgent(torch.nn.Module):
    def __init__(self, input_size, hidden_layer, n_lstm, out_shape):
        super(NetAgent, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_layer, batch_first=True, num_layers=n_lstm)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(hidden_layer, out_shape)

    def forward(self, x, h):
        x, h = self.lstm(x, h)
        x = self.relu(x)
        x = self.fc(x)
        return x, h


class AgentAct(torch.nn.Module):
    def __init__(self, input_size, hidden_layer, n_lstm, out_shape):
        super(AgentAct, self).__init__()
        self.n_lstm = n_lstm
        self.net = NetAgent(input_size, hidden_layer, n_lstm, out_shape)
        self.target_net = NetAgent(input_size, hidden_layer, n_lstm, out_shape)
        self.hidden = hidden_layer

    def forward(self, x, hidden=None):
        if hidden is None:
            h_te = torch.zeros((self.n_lstm, x.shape[0], self.hidden), requires_grad=True)
            c_te = torch.zeros((self.n_lstm, x.shape[0], self.hidden), requires_grad=True)
            hidden = (h_te, c_te)
        return self.net(x, hidden)

    def sample_action(self, x, hidden, stoch=False):
        q_values, hidden = self.net(x, hidden)

        act_idx = self.sample_action_from_q(q_values, stoch=stoch)
        act_idx = act_idx.view(act_idx.shape[0])
        return act_idx, hidden

    def sample_action_from_q(self, q_values, stoch=False):

        if stoch:
            _max = torch.max(q_values, dim=2, keepdim=True).values
            _min = torch.min(q_values, dim=2, keepdim=True).values
            _sc = _max - _min
            q_values = q_values / _sc
            dist = torch.distributions.Categorical(t_functional.softmax(q_values, dim=1))
            act_idx = dist.sample()

        else:
            act_idx = q_values.argmax(dim=2)
            # act_idx = q_values.argmax(dim=2)
        return act_idx


class AgentTeDiscrete(torch.nn.Module):
    def __init__(self, input_size, hidden_layer, n_lstm, te_intervals):
        super(AgentTeDiscrete, self).__init__()
        self.n_lstm = n_lstm
        self.net = NetAgent(input_size, hidden_layer, n_lstm, len(te_intervals))
        self.target_net = NetAgent(input_size, hidden_layer, n_lstm, len(te_intervals))
        self.te_intervals = te_intervals
        self.hidden = hidden_layer

    def forward(self, x, hidden=None):
        if hidden is None:
            h_te = torch.zeros((self.n_lstm, x.shape[0], self.hidden), requires_grad=True)
            c_te = torch.zeros((self.n_lstm, x.shape[0], self.hidden), requires_grad=True)
            hidden = (h_te, c_te)
        return self.net(x, hidden)

    def sample_action(self, x, hidden, stoch=False):
        q_values, hidden = self.net(x, hidden)

        t_idx = self.sample_action_from_q(q_values, stoch)
        t_idx = t_idx.view(t_idx.shape[0])
        return t_idx, hidden

    def sample_action_from_q(self, q_values, stoch=False):
        # print(f'AgentTeDiscrete.sample_action_from_q:: q_values.shape={q_values.shape}')
        if stoch:
            _max = torch.max(q_values, dim=2, keepdim=True).values
            _min = torch.min(q_values, dim=2, keepdim=True).values
            _sc = _max - _min
            q_values = q_values / _sc
            dist = torch.distributions.Categorical(t_functional.softmax(q_values, dim=1))
            t_idx = dist.sample()
        else:
            t_idx = q_values.argmax(dim=2)
        return t_idx

    def act_to_te(self, t_idx):
        device = next(self.parameters()).device
        out = torch.zeros(t_idx.shape, device=device)
        for i in range(out.shape[0]):
            out[i] = (self.te_intervals[t_idx[i]][0] + self.te_intervals[t_idx[i]][1]) / 2.
        return out
