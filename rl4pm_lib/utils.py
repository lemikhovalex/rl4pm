import random
import torch
import numpy as np


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

    def push(self, datum):

        if self._next_idx >= len(self._storage):
            self._storage.append(datum)
        else:
            self._storage[self._next_idx] = datum
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions_te, actions_ac, rewards_te, rewards_ac, obses_tp1, dones = [], [], [], [], [], [], []
        assert len(idxes) != 0
        for i in idxes:
            data = self._storage[i]
            # inp, next_te, next_ac, reward_te, reward_ac, n_inp, is_done
            obs_t, act_te, act_ac, rew_te, rew_ac, obs_tp1, done = data
            obses_t.append(torch.as_tensor(obs_t))
            actions_te.append(torch.as_tensor(act_te))
            actions_ac.append(torch.as_tensor(act_ac))
            rewards_te.append(rew_te)
            rewards_ac.append(rew_ac)
            obses_tp1.append(torch.as_tensor(obs_tp1))
            dones.append(done)
        n_trails = obs_t.shape[0]
        le = len(idxes)

        obses_t = torch.cat(obses_t).view((le, n_trails, -1))
        actions_te = torch.cat(actions_te).view((le, n_trails, -1))
        actions_ac = torch.cat(actions_ac).view((le, n_trails, -1))
        rewards_te = torch.cat(rewards_te).view((le, n_trails, -1))
        rewards_ac = torch.cat(rewards_ac).view((le, n_trails, -1))
        obses_tp1 = torch.cat(obses_tp1).view((le, n_trails, -1))
        dones = torch.cat(dones).view((le, n_trails, -1))

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


def play_and_record(agent_te, agent_ac, env, exp_replay):
    agent_te.eval()
    agent_te.eval()
    inp = env.reset()
    n_traces = inp.shape[0]
    inp = inp.view(n_traces, 1, -1).float()
    is_done = torch.zeros(env.data.shape[0]).bool()
    h_a = torch.zeros(1, n_traces, agent_ac.hidden)
    c_a = torch.zeros(1, n_traces, agent_ac.hidden)
    h_t = torch.zeros(1, n_traces, agent_te.hidden)
    c_t = torch.zeros(1, n_traces, agent_te.hidden)

    episode_te_rew = None
    episode_ac_rew = None
    n = 0

    while not is_done.all():
        next_ac, (h_a, c_a) = agent_ac.sample_action(x=inp, hidden=(h_a, c_a), stoch=True)
        next_te, (h_t, c_t) = agent_te.sample_action(x=inp, hidden=(h_t, c_t), stoch=True)

        n_inp, (reward_te, reward_ac), is_done, add_inf = env.step(agent_te.act_to_te(next_te), next_ac)
        n_inp = n_inp.view(n_traces, 1, -1).float()
        datum = (inp, next_te, next_ac, reward_te, reward_ac, n_inp, is_done)

        if episode_te_rew is None:
            episode_te_rew = reward_te
        else:
            episode_te_rew += reward_te
        if episode_ac_rew is None:
            episode_ac_rew = reward_ac
        else:
            episode_ac_rew += reward_ac
        try:
            n += is_done.sum().item()
        except AttributeError:
            n += is_done

        exp_replay.push(datum)
        inp = n_inp

    try:
        episode_te_rew = episode_te_rew.sum().item()
    except AttributeError:
        pass
    try:
        episode_ac_rew = episode_ac_rew.sum().item()
    except AttributeError:
        pass

    return episode_te_rew, episode_ac_rew, n


def fill_trace(trace_np_matrix, max_len):
    need_pad = max_len - trace_np_matrix.shape[0]
    pad = np.zeros((need_pad, trace_np_matrix.shape[1]))
    return np.concatenate((trace_np_matrix, pad))


def extract_trace_features(df, trace_id, max_len):
    df_id = df[df['trace_id'] == trace_id].drop(columns=['timestamp', 'trace_id', 'activity'])
    trace_vals = df_id.values
    trace_vals = fill_trace(trace_vals, max_len)
    trace_vals = torch.as_tensor(trace_vals).unsqueeze(0)
    return trace_vals


def get_traces_matrix(df, env_trace_ids):
    env_matrix = None
    max_len = 0
    for t_id in env_trace_ids:
        trace_len = df[df['trace_id'] == t_id].shape[0]
        if max_len < trace_len:
            max_len = trace_len

    for _i, t_id in enumerate(env_trace_ids):
        if env_matrix is not None:

            trace_vals = extract_trace_features(df, t_id, max_len)
            env_matrix = torch.cat([env_matrix, trace_vals])
        else:
            env_matrix = extract_trace_features(df, t_id, max_len)

    return env_matrix
