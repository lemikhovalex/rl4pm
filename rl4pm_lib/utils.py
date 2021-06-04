import torch
import numpy as np
import pandas as pd
from .replay_buffer import State, Datum
import matplotlib.pyplot as plt
from math import floor, ceil


def play_and_record(agent_te, agent_ac, env, exp_replay,
                    process_dvice=None,
                    dest_device=torch.device('cpu'), stoch=True):
    if process_dvice is None:
        process_dvice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_te.eval()
    agent_ac.eval()
    inp = env.reset()
    n_traces = inp.shape[0]
    inp = inp.view(n_traces, 1, -1).float().to(device=process_dvice)
    is_done = torch.zeros(env.data.shape[0], device=process_dvice).bool()
    h_a = torch.zeros(agent_ac.n_lstm, n_traces, agent_ac.hidden, device=process_dvice)
    c_a = torch.zeros(agent_ac.n_lstm, n_traces, agent_ac.hidden, device=process_dvice)
    h_t = torch.zeros(agent_te.n_lstm, n_traces, agent_te.hidden, device=process_dvice)
    c_t = torch.zeros(agent_te.n_lstm, n_traces, agent_te.hidden, device=process_dvice)

    agent_te.to(device=process_dvice)
    agent_ac.to(device=process_dvice)
    if env.device != process_dvice:
        env.to(device=process_dvice)

    episode_te_rew = None
    episode_ac_rew = None
    n = 0

    while not is_done.all():

        state_t = State(state=inp,
                        h_ac=h_a, c_ac=c_a,
                        h_te=h_t, c_te=c_t)
        next_ac, (h_a, c_a) = agent_ac.sample_action(x=inp, hidden=(h_a, c_a), stoch=stoch)

        next_te, (h_t, c_t) = agent_te.sample_action(x=inp, hidden=(h_t, c_t), stoch=stoch)
        n_inp, (reward_te, reward_ac), is_done, add_inf = env.step((agent_te.act_to_te(next_te), next_ac))
        n_inp = n_inp.view(n_traces, 1, -1).float()

        state_t_next = State(state=n_inp,
                             h_ac=h_a, c_ac=c_a,
                             h_te=h_t, c_te=c_t)
        datum = Datum(obs_t=state_t, action_te=next_te, action_ac=next_ac, reward_ac=reward_ac, reward_te=reward_te,
                      obs_tp1=state_t_next, dones=is_done)
        if episode_te_rew is None:
            episode_te_rew = reward_te
        else:
            episode_te_rew += reward_te
        if episode_ac_rew is None:
            episode_ac_rew = reward_ac
        else:
            episode_ac_rew += reward_ac
        try:
            n += is_done.logical_not().sum().item()
        except AttributeError:
            n += 1 - is_done

        if process_dvice != dest_device:
            datum.to(device=dest_device)

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


def fill_trace(trace_np_matrix, max_len, fill_value=0):
    need_pad = max_len - trace_np_matrix.shape[0]
    pad = np.ones((need_pad, trace_np_matrix.shape[1])) * fill_value
    return np.concatenate((trace_np_matrix, pad))


def extract_trace_features(df: pd.DataFrame, trace_id, max_len):
    to_dr = []
    for col in ['timestamp', 'trace_id', 'activity']:
        if col in df.columns:
            to_dr.append(col)
    if trace_id is not None:
        trace_vals = df[df['trace_id'] == trace_id].drop(columns=to_dr).values
        trace_vals = fill_trace(trace_vals, max_len)
        trace_vals = torch.as_tensor(trace_vals).unsqueeze(0)
    else:
        n_features = df.drop(columns=to_dr).shape[1]
        trace_vals = torch.zeros(1, max_len, n_features)
    return trace_vals


def extend_env_matrix(env_matrix, df: pd.DataFrame, t_id, max_len):
    if env_matrix is not None:
        trace_vals = extract_trace_features(df, t_id, max_len)
        assert trace_vals.shape[2] == 27
        env_matrix = torch.cat([env_matrix, trace_vals])
    else:
        env_matrix = extract_trace_features(df, t_id, max_len)
        assert env_matrix.shape[2] == 27
    return env_matrix


def get_traces_matrix(df: pd.DataFrame, env_trace_ids):
    env_matrix = None
    max_len = 0
    for t_id in env_trace_ids:
        if t_id is not None:
            trace_len = df[df['trace_id'] == t_id].shape[0]
            if max_len < trace_len:
                max_len = trace_len

    for t_id in env_trace_ids:
        env_matrix = extend_env_matrix(env_matrix, df, t_id, max_len)

    return env_matrix


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def plot_laerning_process(te_rewards, ac_rewards, losses_te, losses_ac):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    fig.tight_layout(pad=5)
    axs[0, 0].plot(te_rewards, label='$t_e$')
    axs[0, 0].set(xlabel='epoch', ylabel='accuracy', title='Accuracy score for \n$t_e$ prediction')
    axs[1, 0].plot(ac_rewards, label='next activity')
    axs[1, 0].set(xlabel='epoch', ylabel='accuracy', title='Accuracy score for \nnext activity prediction')

    axs[0, 1].plot(losses_ac, label='$t_e$')
    axs[0, 1].set(xlabel='batch', ylabel='reward', title='Loss \n$t_e$ prediction')
    axs[1, 1].plot(losses_te, label='next activity')
    axs[1, 1].set(xlabel='batch', ylabel='reward', title='Loss \nnext activity prediction')
    plt.legend()
    plt.show()


def split_to_fixed_bucket(array, bucket_size, fill_none=True):
    index = len(array)
    out = []
    while index > 0:
        beg = max(0, index - bucket_size)
        end = index
        out.append(array[beg: end])
        index -= bucket_size
    out[-1].extend([None] * (-1 * index))
    return out


def split_list_to_buckets(array, n):
    n_buckets = ceil(len(array) / n)
    extra_size = n_buckets * n - len(array)
    out = []
    fp = 0
    lp = n
    for i in range(n_buckets):
        lp = lp - int(extra_size > 0)
        out.append(array[fp: lp])
        extra_size -= 1
        fp = lp
        lp += n
    return out
