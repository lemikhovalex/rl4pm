import torch
import numpy as np
from .replay_buffer import State, Datum


def play_and_record(agent_te, agent_ac, env, exp_replay,
                    process_dvice=None,
                    dest_device=torch.device('cpu')):
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
        next_ac, (h_a, c_a) = agent_ac.sample_action(x=inp, hidden=(h_a, c_a), stoch=True)

        next_te, (h_t, c_t) = agent_te.sample_action(x=inp, hidden=(h_t, c_t), stoch=True)

        # print('utils.play_and_record::')
        # print(f'\tprocess_device={process_dvice}')
        # print(f'\tagent_te.device={next(agent_te.parameters()).device}')
        # print(f'\tagent_ac.device={next(agent_ac.parameters()).device}')
        # print(f'\tnext_te.device={next_te.device}')
        # print(f'\tagent_te.act_to_te(next_te).device={agent_te.act_to_te(next_te).device}')
        # print(f'\tnext_ac.device={next_ac.device}')
        n_inp, (reward_te, reward_ac), is_done, add_inf = env.step(agent_te.act_to_te(next_te), next_ac)
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


def fill_trace(trace_np_matrix, max_len):
    need_pad = max_len - trace_np_matrix.shape[0]
    pad = np.zeros((need_pad, trace_np_matrix.shape[1]))
    return np.concatenate((trace_np_matrix, pad))


def extract_trace_features(df, trace_id, max_len):
    if trace_id is not None:
        df_id = df[df['trace_id'] == trace_id].drop(columns=['timestamp', 'trace_id', 'activity'])
        trace_vals = df_id.values
        trace_vals = fill_trace(trace_vals, max_len)
        trace_vals = torch.as_tensor(trace_vals).unsqueeze(0)
    else:
        trace_vals = torch.zeros(1, max_len, df.shape[1] - 3)
    return trace_vals


def extend_env_matrix(env_matrix, df, t_id, max_len):
    if env_matrix is not None:
        trace_vals = extract_trace_features(df, t_id, max_len)
        env_matrix = torch.cat([env_matrix, trace_vals])
    else:
        env_matrix = extract_trace_features(df, t_id, max_len)
    return env_matrix


def get_traces_matrix(df, env_trace_ids):
    env_matrix = None
    max_len = 0
    for t_id in env_trace_ids:
        if t_id is not None:
            trace_len = df[df['trace_id'] == t_id].shape[0]
            if max_len < trace_len:
                max_len = trace_len

    for _i, t_id in enumerate(env_trace_ids):
        env_matrix = extend_env_matrix(env_matrix, df, t_id, max_len)

    return env_matrix


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
