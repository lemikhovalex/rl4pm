import random
import numpy as np
import torch


def transform_hidden(x: np.ndarray, not_done):
    num_layers, batch, hidden_size = x.shape
    x = x.reshape((batch, num_layers, hidden_size))
    x = x[not_done]
    return x


class State:
    def __init__(self, state: np.ndarray,
                 h_ac: np.ndarray,
                 c_ac: np.ndarray,
                 h_te: np.ndarray,
                 c_te: np.ndarray):
        self.state = state
        self.h_ac = h_ac
        self.c_ac = c_ac
        self.h_te = h_te
        self.c_te = c_te

    def filter(self, is_done: np.ndarray):
        not_done = np.logical_not(is_done)
        self.state = self.state[not_done]
        self.h_ac = transform_hidden(self.h_ac, not_done)
        self.h_te = transform_hidden(self.h_te, not_done)
        self.c_ac = transform_hidden(self.c_ac, not_done)
        self.c_te = transform_hidden(self.c_te, not_done)

    def input_for_nn(self, is_dones: np.ndarray, device):
        not_done = np.logical_not(is_dones).reshape(-1)
        n_features = self.state.shape[-1]
        hidden_te_shape = self.h_te.shape[-1]
        hidden_ac_shape = self.h_ac.shape[-1]
        obs_t = self.state.reshape((-1, 1, n_features))[not_done]
        obs_h_te = self.h_te.reshape((-1, 1, hidden_te_shape))[not_done].reshape((1, -1, hidden_te_shape))
        obs_c_te = self.c_te.reshape((-1, 1, hidden_te_shape))[not_done].reshape((1, -1, hidden_te_shape))
        obs_h_ac = self.h_ac.reshape((-1, 1, hidden_ac_shape))[not_done].reshape((1, -1, hidden_te_shape))
        obs_c_ac = self.c_ac.reshape((-1, 1, hidden_ac_shape))[not_done].reshape((1, -1, hidden_te_shape))
        return torch.as_tensor(obs_t, device=device, dtype=torch.float), \
               torch.as_tensor(obs_h_te, device=device, dtype=torch.float), \
               torch.as_tensor(obs_c_te, device=device, dtype=torch.float), \
               torch.as_tensor(obs_h_ac, device=device, dtype=torch.float), \
               torch.as_tensor(obs_c_ac, device=device, dtype=torch.float)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, item):
        raise NotImplemented
        # return State(state=self.state[item].unsqueeze(0),
        #              h_ac=self.h_te[item].unsqueeze(0), h_te=self.h_te[item].unsqueeze(0),
        #              c_te=self.c_te[item].unsqueeze(0), c_ac=self.c_te[item].unsqueeze(0))

    def __iter__(self):
        return iter([self.state,
                     self.h_te, self.c_te,
                     self.h_ac, self.c_te])

    def to(self, device):
        raise NotImplemented
        # self.state = self.state.to(device=device)
        # self.h_ac = self.h_ac.to(device=device)
        # self.c_ac = self.c_ac.to(device=device)
        # self.h_te = self.h_te.to(device=device)
        # self.c_te = self.c_te.to(device=device)


class Datum(object):
    def __init__(self, obs_t: State,
                 action_te: np.ndarray,
                 action_ac: np.ndarray,
                 reward_te: np.ndarray,
                 reward_ac: np.ndarray,
                 obs_tp1: State,
                 dones: np.ndarray):
        self.obs_t = obs_t
        self.action_te = action_te
        self.action_ac = action_ac
        self.reward_te = reward_te
        self.reward_ac = reward_ac
        self.obs_tp1 = obs_tp1
        self.dones = dones
        self.shape = obs_t.state.shape

    def __iter__(self):
        return iter([self.obs_t,
                     self.action_te, self.action_ac,
                     self.reward_te, self.reward_ac,
                     self.obs_tp1,
                     self.dones])

    def filter(self):
        self.action_te = self.action_te[np.logical_not(self.dones)]
        self.action_ac = self.action_ac[np.logical_not(self.dones)]
        self.reward_te = self.reward_te[np.logical_not(self.dones)]
        self.reward_ac = self.reward_ac[np.logical_not(self.dones)]
        self.obs_t.filter(self.dones)
        self.obs_tp1.filter(self.dones)
        self.dones = self.dones[np.logical_not(self.dones)]

    def to(self, device):
        raise NotImplemented
        # self.obs_t.to(device=device)
        # self.action_te.to(device=device)
        # self.action_ac.to(device=device)
        # self.reward_te.to(device=device)
        # self.reward_ac.to(device=device)
        # self.obs_tp1.to(device=device)
        # self.dones.to(device=device)


def _replay_buff_view_state(x, le, n_trails):
    return np.concatenate(x).reshape((le, n_trails, -1))


def _replay_buff_view(x, le, n_trails):
    return np.concatenate(x).reshape((le, n_trails, -1))


class ReplayMemory(object):
    def __init__(self, size, traces=64):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.traces = traces
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def _push(self, datum: Datum):
        if self._next_idx >= len(self._storage):
            self._storage.append(datum)
        else:
            self._storage[self._next_idx] = datum
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def push(self, datum: Datum):
        self._push(datum)

    def _encode_sample(self, idxes):
        actions_te, actions_ac, rewards_te, rewards_ac, dones = [], [], [], [], []
        obses_tp1_s, obses_t_s = [], []
        obses_tp1_h_ac, obses_tp1_h_te, obses_tp1_c_ac, obses_tp1_c_te = [], [], [], []
        obses_t_h_ac, obses_t_h_te, obses_t_c_ac, obses_t_c_te = [], [], [], []
        le = len(idxes)

        for i in idxes:
            datum = self._storage[i]
            # inp, next_te, next_ac, reward_te, reward_ac, n_inp, is_done
            obs_t, act_te, act_ac, rew_te, rew_ac, obs_tp1, done = datum
            # print(f'\tact_te.shape={act_te.shape}')
            actions_te.append(act_te)
            actions_ac.append(act_ac)
            rewards_te.append(rew_te)
            rewards_ac.append(rew_ac)
            dones.append(done)

            obses_tp1_s.append(obs_tp1.state)
            obses_t_s.append(obs_t.state)

            obses_t_h_ac.append(obs_t.h_ac)
            obses_t_h_te.append(obs_t.h_te)
            obses_t_c_ac.append(obs_t.c_ac)
            obses_t_c_te.append(obs_t.c_te)

            obses_tp1_h_ac.append(obs_tp1.h_ac)
            obses_tp1_h_te.append(obs_tp1.h_te)
            obses_tp1_c_ac.append(obs_tp1.c_ac)
            obses_tp1_c_te.append(obs_tp1.c_te)

        obses_tp1 = State(state=_replay_buff_view_state(obses_tp1_s, le, self.traces),
                          h_te=_replay_buff_view_state(obses_tp1_h_te, le, self.traces),
                          h_ac=_replay_buff_view_state(obses_tp1_h_ac, le, self.traces),
                          c_te=_replay_buff_view_state(obses_tp1_c_te, le, self.traces),
                          c_ac=_replay_buff_view_state(obses_tp1_c_ac, le, self.traces)
                          )
        obses_t = State(state=_replay_buff_view_state(obses_t_s, le, self.traces),
                        h_te=_replay_buff_view_state(obses_t_h_te, le, self.traces),
                        h_ac=_replay_buff_view_state(obses_t_h_ac, le, self.traces),
                        c_te=_replay_buff_view_state(obses_t_c_te, le, self.traces),
                        c_ac=_replay_buff_view_state(obses_t_c_ac, le, self.traces)
                        )

        actions_te = _replay_buff_view(actions_te, le, self.traces)
        actions_ac = _replay_buff_view(actions_ac, le, self.traces)
        rewards_te = _replay_buff_view(rewards_te, le, self.traces)
        rewards_ac = _replay_buff_view(rewards_ac, le, self.traces)
        dones = _replay_buff_view(dones, le, self.traces)

        #     inp,     next_te,    next_ac,    reward_te,  reward_ac,  n_inp,     is_done
        out = obses_t, actions_te, actions_ac, rewards_te, rewards_ac, obses_tp1, dones
        return out

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def __len__(self):
        return len(self._storage)

    def is_full(self):
        return len(self._storage) == self._maxsize
