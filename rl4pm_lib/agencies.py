import torch
from .agents import AgentAct, AgentTeDiscrete
from .utils import play_and_record
from torch.nn import functional as t_functional


class Agency:
    def __init__(self, input_size, hidden, n_lstm, te_intervals, ac_learning_rate,
                 te_learning_rate, n_classes, discount_factor):
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

        # play and record
        _ = env.reset()
        with torch.no_grad():
            episode_te_rew, episode_ac_rew = play_and_record(self.te_agent, self.ac_agent, env, exp_replay)

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

        return episode_te_rew, episode_ac_rew

    def get_loss_discrete_agent(self, states, actions, rewards,
                                next_states, is_dones, agent):
        is_dones = is_dones.view(-1)

        states = states.float()
        actions = actions.long()
        rewards = rewards.float()
        next_states = next_states.float()

        qs, _ = agent(states)
        predicted_q_values = torch.gather(input=qs, index=actions, dim=2)

        with torch.no_grad():
            qs_next, _ = agent(next_states)
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
        te_agent_loss = self.get_loss_discrete_agent(states, actions_te, rewards_te,
                                                     next_states, is_dones, agent=self.te_agent)

        ac_agent_loss = self.get_loss_discrete_agent(states, actions_ac, rewards_ac,
                                                     next_states, is_dones, agent=self.ac_agent)
        return te_agent_loss, ac_agent_loss
