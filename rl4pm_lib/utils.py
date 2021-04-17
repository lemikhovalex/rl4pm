import torch
import numpy as np
from .replay_buffer import State, Datum


def play_and_record(agent_te, agent_ac, env, exp_replay):
    agent_te.eval()
    agent_te.eval()
    inp = env.reset()
    n_traces = inp.shape[0]
    inp = inp.view(n_traces, 1, -1).float()
    is_done = torch.zeros(env.data.shape[0]).bool()
    h_a = torch.zeros(agent_ac.n_lstm, n_traces, agent_ac.hidden)
    c_a = torch.zeros(agent_ac.n_lstm, n_traces, agent_ac.hidden)
    h_t = torch.zeros(agent_te.n_lstm, n_traces, agent_te.hidden)
    c_t = torch.zeros(agent_te.n_lstm, n_traces, agent_te.hidden)

    episode_te_rew = None
    episode_ac_rew = None
    n = 0

    while not is_done.all():

        state_t = State(state=inp, h_ac=h_a, c_ac=c_a,
                        h_te=h_t, c_te=c_t)
        next_ac, (h_a, c_a) = agent_ac.sample_action(x=inp, hidden=(h_a, c_a), stoch=True)

        next_te, (h_t, c_t) = agent_te.sample_action(x=inp, hidden=(h_t, c_t), stoch=True)

        n_inp, (reward_te, reward_ac), is_done, add_inf = env.step(agent_te.act_to_te(next_te), next_ac)
        n_inp = n_inp.view(n_traces, 1, -1).float()

        state_t_next = State(state=n_inp, h_ac=h_a, c_ac=c_a,
                             h_te=h_t, c_te=c_t)

        datum = Datum(obs_t=state_t, action_te=next_te, action_ac=next_ac, reward_ac=reward_ac, reward_te=reward_te,
                      obs_t1=state_t_next, dones=is_done)
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
