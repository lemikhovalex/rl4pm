import random
import torch


class State:
    def __init__(self, state: torch.tensor,
                 h_ac: torch.tensor,
                 c_ac: torch.tensor,
                 h_te: torch.tensor,
                 c_te: torch.tensor):
        self.state = state
        self.h_ac = h_ac
        self.c_ac = c_ac
        self.h_te = h_te
        self.c_te = c_te


class Datum(object):
    def __init__(self, obs_t: State,
                 action_te: torch.tensor,
                 action_ac: torch.tensor,
                 reward_te: torch.tensor,
                 reward_ac: torch.tensor,
                 obs_t1: State,
                 dones: torch.tensor):
        self.obs_t = obs_t
        self.action_te = action_te
        self.action_ac = action_ac
        self.reward_te = reward_te
        self.reward_ac = reward_ac
        self.obs_t1 = obs_t1
        self.dones = dones

    def __iter__(self):
        return iter([self.obs_t,
                     self.action_te, self.action_ac,
                     self.reward_te, self.reward_ac,
                     self.obs_t1,
                     self.dones])


def _replay_buff_view_state(x, le, n_trails):
    return torch.cat(x).view((le, n_trails, -1))


def _replay_buff_view(x, le, n_trails):
    return torch.cat(x).view((le, n_trails, -1))


class ReplayMemory(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, datum: Datum):

        if self._next_idx >= len(self._storage):
            self._storage.append(datum)
        else:
            self._storage[self._next_idx] = datum
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        actions_te, actions_ac, rewards_te, rewards_ac, dones = [], [], [], [], []
        obses_tp1_s, obses_t_s = [], []
        obses_tp1_h_ac, obses_tp1_h_te, obses_tp1_c_ac, obses_tp1_c_te = [], [], [], []
        obses_t_h_ac, obses_t_h_te, obses_t_c_ac, obses_t_c_te = [], [], [], []

        le = len(idxes)
        assert le != 0

        n_trails = -1
        for i in idxes:
            data = self._storage[i]
            # inp, next_te, next_ac, reward_te, reward_ac, n_inp, is_done
            obs_t, act_te, act_ac, rew_te, rew_ac, obs_tp1, done = data
            n_trails = act_te.shape[0]
            actions_te.append(torch.as_tensor(act_te))
            actions_ac.append(torch.as_tensor(act_ac))
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
        assert n_trails >= 0
        obses_tp1 = {'s': _replay_buff_view_state(obses_tp1_s, le, n_trails),
                     'h_te': _replay_buff_view(obses_tp1_h_te, le, n_trails),
                     'c_te': _replay_buff_view(obses_tp1_h_te, le, n_trails),
                     'h_ac': _replay_buff_view(obses_tp1_h_ac, le, n_trails),
                     'c_ac': _replay_buff_view(obses_tp1_h_ac, le, n_trails),
                     }
        obses_t = {'s': _replay_buff_view_state(obses_t_s, le, n_trails),
                   'h_te': _replay_buff_view(obses_t_h_te, le, n_trails),
                   'c_te': _replay_buff_view(obses_t_h_te, le, n_trails),
                   'h_ac': _replay_buff_view(obses_t_h_ac, le, n_trails),
                   'c_ac': _replay_buff_view(obses_t_h_ac, le, n_trails),
                   }

        actions_te = _replay_buff_view_state(actions_te, le, n_trails)
        actions_ac = _replay_buff_view_state(actions_ac, le, n_trails)
        rewards_te = _replay_buff_view_state(rewards_te, le, n_trails)
        rewards_ac = _replay_buff_view_state(rewards_ac, le, n_trails)
        dones = _replay_buff_view_state(dones, le, n_trails)

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


