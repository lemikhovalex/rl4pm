import torch
from .agents import AgentAct, AgentTeDiscrete
from .utils import play_and_record
from torch.nn import functional as t_functional


class Agency:
    def __init__(self, input_size, hidden, n_lstm, te_intervals, ac_learning_rate,
                 te_learning_rate, n_classes, discount_factor):
        self.hidden_size = hidden
        self.input_size = input_size
        self.n_lstm = n_lstm
        self.te_agent = AgentTeDiscrete(input_size=input_size, hidden_layer=hidden, n_lstm=n_lstm,
                                        te_intervals=te_intervals).float()
        self.ac_agent = AgentAct(input_size=input_size, hidden_layer=hidden, n_lstm=n_lstm,
                                 out_shape=n_classes).float()
        self.discount_factor = discount_factor
        self.te_agent_targ = AgentTeDiscrete(input_size=input_size, hidden_layer=hidden, n_lstm=n_lstm,
                                             te_intervals=te_intervals).float()
        self.ac_agent_targ = AgentAct(input_size=input_size, hidden_layer=hidden, n_lstm=n_lstm,
                                      out_shape=n_classes).float()
        self.refresh_target()

        self.te_opt = torch.optim.Adam(self.te_agent.parameters(), lr=ac_learning_rate)
        self.ac_opt = torch.optim.Adam(self.ac_agent.parameters(), lr=te_learning_rate)

    def refresh_target(self, polyak_avg=1):
        for target_param, param in zip(self.te_agent_targ.parameters(), self.te_agent.parameters()):
            target_param.data.copy_(polyak_avg * param + (1 - polyak_avg) * target_param)

        for target_param, param in zip(self.ac_agent_targ.parameters(), self.ac_agent.parameters()):
            target_param.data.copy_(polyak_avg * param + (1 - polyak_avg) * target_param)

    def train(self, env, exp_replay, batch_size):
        self.te_agent.train()
        self.ac_agent.train()
        self.ac_agent_targ.train()
        self.te_agent_targ.train()
        # play and record
        _ = env.reset()
        episode_te_rew, episode_ac_rew = 0, 0
        with torch.no_grad():
            while len(exp_replay) < batch_size // 2:
                _episode_te_rew, _episode_ac_rew, n = play_and_record(self.te_agent, self.ac_agent, env, exp_replay)
                episode_te_rew += _episode_te_rew
                episode_ac_rew += _episode_ac_rew

        # get data from memory

        # train
        upd_size = batch_size if len(exp_replay) > batch_size else len(exp_replay) // 2
        states, actions_te, actions_ac, rewards_te, rewards_ac, next_states, is_dones = exp_replay.sample(upd_size)

        te_agent_loss, ac_agent_loss = self.get_losses(states, actions_te, actions_ac, rewards_te, rewards_ac,
                                                       next_states, is_dones)

        # update networks
        self.te_opt.zero_grad()
        te_agent_loss.backward()
        self.te_opt.step()

        self.ac_opt.zero_grad()
        ac_agent_loss.backward()
        self.ac_opt.step()

        return episode_te_rew, episode_ac_rew, is_dones.logical_not().long().sum().item()

    def get_loss_discrete_agent(self, obs_t, h_t, c_t, actions, rewards,
                                obs_tp1, h_tp1, c_tp1, is_dones, agent):

        n_trails = obs_t.shape[1]
        max_len = obs_t.shape[0]
        obs_t = obs_t.view(-1, 1, self.input_size)
        h_t = h_t.view(agent.n_lstm, -1, self.hidden_size)
        c_t = c_t.view(agent.n_lstm, -1, self.hidden_size)

        qs, _ = agent(x=obs_t, hidden=(h_t, c_t))
        qs = qs.view(max_len, n_trails, -1)
        predicted_q_values = torch.gather(input=qs, index=actions, dim=2)

        with torch.no_grad():
            obs_tp1 = obs_tp1.view(-1, 1, self.input_size)
            h_tp1 = h_tp1.view(agent.n_lstm, -1, self.hidden_size)
            c_tp1 = c_tp1.view(agent.n_lstm, -1, self.hidden_size)
            qs_next, _ = agent(x=obs_tp1, hidden=(h_tp1, c_tp1))
            qs_next = qs_next.view(max_len, n_trails, -1)
            actions_idx_next = agent.sample_action_from_q(qs_next, stoch=True)
        predicted_next_q_values = torch.gather(input=qs_next, index=actions_idx_next.unsqueeze(2), dim=2)

        q_reference = rewards + self.discount_factor * predicted_next_q_values
        q_reference = q_reference.view(-1, 1)[is_dones.logical_not()]
        predicted_q_values = predicted_q_values.view(-1, 1)[is_dones.logical_not()]

        # # mean squared error loss to minimize
        loss = t_functional.smooth_l1_loss(predicted_q_values, q_reference).mean()

        return loss

    def get_losses(self, states, actions_te, actions_ac, rewards_te, rewards_ac,
                   next_states, is_dones):
        is_dones = is_dones.view(-1)

        obs_t = states['s'].float()
        obs_t_h_te = states['h_te']
        obs_t_h_ac = states['h_ac']
        obs_t_c_te = states['c_te']
        obs_t_c_ac = states['c_ac']
        actions_te = actions_te.long()
        actions_ac = actions_ac.long()
        rewards_te = rewards_te.float()
        rewards_ac = rewards_ac.float()

        obs_tp1 = next_states['s'].float()
        obs_tp1_h_te = next_states['h_te']
        obs_tp1_h_ac = next_states['h_ac']
        obs_tp1_c_te = next_states['c_te']
        obs_tp1_c_ac = next_states['c_ac']
        te_agent_loss = self.get_loss_discrete_agent(obs_t=obs_t, h_t=obs_t_h_te, c_t=obs_t_c_te,
                                                     actions=actions_te, rewards=rewards_te, is_dones=is_dones,
                                                     obs_tp1=obs_tp1, h_tp1=obs_tp1_h_te, c_tp1=obs_tp1_c_te,
                                                     agent=self.te_agent)

        ac_agent_loss = self.get_loss_discrete_agent(obs_t=obs_t, h_t=obs_t_h_ac, c_t=obs_t_c_ac,
                                                     actions=actions_ac, rewards=rewards_ac, is_dones=is_dones,
                                                     obs_tp1=obs_tp1, h_tp1=obs_tp1_h_ac, c_tp1=obs_tp1_c_ac,
                                                     agent=self.ac_agent)
        return te_agent_loss, ac_agent_loss
