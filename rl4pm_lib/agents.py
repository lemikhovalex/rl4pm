import torch
from torch.nn import functional as t_functional


def sample_action_from_q(q_values, stoch=False):
    """
    sampling, based on q-values
    Args:
        q_values: q-values
        stoch: greedy, or with stoch. basicly function is all about this storhc logic

    Returns:
        indexes of q-values, which must be chosen for Bellman equation
    """
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


class NetAgent(torch.nn.Module):
    """
    Net used for agents, lstm-based
    Args:
        input_size(int): nuber of features for input
        hidden_layer(int): size of hidden layer
        n_lstm(int): number of lstm stacked for NN
        out_shape(int): out dimension

    Attributes:
        lstm(torch.nn.LSTM): lstm-based piece of nn
        relu(torch.nn.ReLU): activation function
        fc(torch.nn.Linear): fully connected layer
    """
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


class BaseAgent(torch.nn.Module):
    """
    Base agent, suitble for discrete actions
    Args:
        input_size(int): nuber of features for input. for NN creation
        hidden_layer(int): size of hidden layer. for NN creation
        n_lstm(int): number of lstm stacked for NN. for NN creation
        out_shape(int): out dimension. for NN creation
    Attributes
        n_lstm(int): number of lstm stacked for NN. for NN creation
        net(torhc.nn.Module): basicly net, which returns q-values
        hidden_layer(int): size of hidden layer. for NN creation

    """
    def __init__(self, input_size, hidden_layer, n_lstm, out_shape):
        super(BaseAgent, self).__init__()
        self.n_lstm = n_lstm
        self.net = NetAgent(input_size, hidden_layer, n_lstm, out_shape)
        self.hidden = hidden_layer

    def forward(self, x, hidden=None):
        if hidden is None:
            h_te = torch.zeros((self.n_lstm, x.shape[0], self.hidden), requires_grad=True)
            c_te = torch.zeros((self.n_lstm, x.shape[0], self.hidden), requires_grad=True)
            hidden = (h_te, c_te)
        return self.net(x, hidden)

    def sample_action(self, x, hidden=None, stoch=False):
        """
        is smaples actions with given state and hidden output, in RNN
        Args:
            x: state, for RNN
            hidden: as for RNN
            stoch: if action will be sampled by randomly, or not

        Returns: indexes of q -values(keys for discrete actions) and hidden state, maybe need to save it

        """
        q_values, hidden = self.net(x, hidden)

        act_idx = self.sample_action_from_q(q_values, stoch=stoch)
        act_idx = act_idx.view(act_idx.shape[0])
        return act_idx, hidden

    def sample_action_from_q(self, q_values: torch.tensor, stoch=False):
        """
        method which samples action directly from q-value. it is usefull while computin Bellman error
        Args:
            q_values(torhc.tensor): qvalues, based on which the actions must be sampled
            stoch(bool): flag, if action must be produced randomly, or greedy

        Returns:

        """
        return sample_action_from_q(q_values=q_values, stoch=stoch)


class AgentAct(BaseAgent):
    def __init__(self, input_size, hidden_layer, n_lstm, out_shape):
        super(AgentAct, self).__init__(input_size=input_size, hidden_layer=hidden_layer, n_lstm=n_lstm,
                                       out_shape=out_shape)
        self.n_lstm = n_lstm
        self.net = NetAgent(input_size, hidden_layer, n_lstm, out_shape)
        self.hidden = hidden_layer


class AgentTeDiscrete(BaseAgent):
    def __init__(self, input_size, hidden_layer, n_lstm, te_intervals):
        super(AgentTeDiscrete, self).__init__(input_size=input_size, hidden_layer=hidden_layer, n_lstm=n_lstm,
                                              out_shape=len(te_intervals))
        self.n_lstm = n_lstm
        self.net = NetAgent(input_size, hidden_layer, n_lstm, len(te_intervals))
        self.te_intervals = te_intervals
        self.hidden = hidden_layer

    def act_to_te(self, t_idx):
        """
        additional method to convert key(index) to real value
        Args:
            t_idx: indexes of time intervals

        Returns: real values of next te predictionsm from indexes

        """
        device = next(self.parameters()).device
        out = torch.zeros(t_idx.shape, device=device)
        for i in range(out.shape[0]):
            out[i] = (self.te_intervals[t_idx[i]][0] + self.te_intervals[t_idx[i]][1]) / 2.
        return out
