import torch
from .agents import AgentAct, AgentTeDiscrete, BaseAgent
from .utils import init_weights
from torch.nn import functional as t_functional
from .replay_buffer import State


class Agency:
    """
    class that includes agents(target net, and net for predicting) and does train
    Args:
        input_size(int): nuber of features for input. for agent creation
        hidden(int): size of hidden layer. for agent creation
        n_lstm(int): number of lstm stacked. for agent creation
        te_intervals(list): list of tuples (float, float), which are beginnings and ends of tr intervals for reward
                            constructions. if predicted te, and true in the same interval, then +1, else +0
        n_classes(int): n_classes for action predictions
        discount_factor(float): Q(s) = \sum_{t=0}^{\inf} r_{t} \cdot discount_factor^{t}
    Attributes:
        input_size(int): nuber of features for input. for agent creation
        hidden_size(int): size of hidden layer. for agent creation
        n_lstm(int): number of lstm stacked. for agent creation
        te_intervals(list): list of tuples (float, float), which are beginnings and ends of tr intervals for reward
                            constructions. if predicted te, and true in the same interval, then +1, else +0
        discount_factor(float): Q(s) = \sum_{t=0}^{\inf} r_{t} \cdot discount_factor^{t}
        ac_agent(AgentAct): agent, which predicts actions and have nn inside it
        te_agent(AgentTeDiscrete): agent, which predicts actions and have nn inside it
        ac_agent_targ(AgentAct): target version of ac_agent, not ot change nn weights so quiqly
        te_agent_targ(AgentTeDiscrete): target version of te_agent, not ot change nn weights so quiqly
        te_opt(torch.optim): optimizer for te_agent
        ac_opt(torch.optim): optimizer ac_agent
    """

    def __init__(self, input_size, hidden, n_lstm, te_intervals, ac_learning_rate,
                 te_learning_rate, n_classes, discount_factor):
        self.hidden_size = hidden
        self.input_size = input_size
        self.n_lstm = n_lstm
        self.ac_agent = AgentAct(input_size=input_size, hidden_layer=hidden, n_lstm=n_lstm,
                                 out_shape=n_classes).float()
        self.te_agent = AgentTeDiscrete(input_size=input_size, hidden_layer=hidden, n_lstm=n_lstm,
                                        te_intervals=te_intervals).float()
        self.discount_factor = discount_factor

        self.te_agent.apply(init_weights)
        self.ac_agent.apply(init_weights)

        self.te_agent_targ = AgentTeDiscrete(input_size=input_size, hidden_layer=hidden, n_lstm=n_lstm,
                                             te_intervals=te_intervals).float()
        self.ac_agent_targ = AgentAct(input_size=input_size, hidden_layer=hidden, n_lstm=n_lstm,
                                      out_shape=n_classes).float()
        self.refresh_target()

        self.te_opt = torch.optim.Adam(self.te_agent.parameters(), lr=ac_learning_rate)
        self.ac_opt = torch.optim.Adam(self.ac_agent.parameters(), lr=te_learning_rate)

    def refresh_target(self, polyak_avg=1):
        """
        this method refresh terget agents smoothly (polyak average)
        Args:
            polyak_avg: paarm_targ_new = paarm_new * polyak_avg + paarm_targ_old * (1 - polyak_avg)

        Returns: nothing

        """
        for target_param, param in zip(self.te_agent_targ.parameters(), self.te_agent.parameters()):
            target_param.data.copy_(polyak_avg * param + (1 - polyak_avg) * target_param)

        for target_param, param in zip(self.ac_agent_targ.parameters(), self.ac_agent.parameters()):
            target_param.data.copy_(polyak_avg * param + (1 - polyak_avg) * target_param)

    def train(self, exp_replay, batch_size):
        """
        not a .train() as for nn.Module))) it trains agents on batch from exp replay
        Args:
            exp_replay(it can sample, push and len()): replay buffer, from which returns
                                                        (batch_size, n_traces, n_features_nn).
                                                        n_traces is an exp_replay attribute
            batch_size(int): size of batch, sampled from exp_replay

        Returns: te_loss(float), ac_loss(float): Bellman losses

        """
        self.te_agent.train()
        self.ac_agent.train()
        self.ac_agent_targ.train()
        self.te_agent_targ.train()
        # train
        upd_size = batch_size if len(exp_replay) > batch_size else int(len(exp_replay) * 0.8)
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

        return te_agent_loss.item(), ac_agent_loss.item()

    def get_loss_discrete_agent(self, obs_t: torch.tensor, h_t: torch.tensor, c_t: torch.tensor, actions: torch.tensor,
                                rewards: torch.tensor, obs_tp1: torch.tensor, h_tp1: torch.tensor, c_tp1: torch.tensor,
                                agent: BaseAgent):
        """
        method which calculates loss for descrete predictions. based on (state, reward, next_state)
        Args:
            obs_t(torch.tensor): observation
            h_t(torch.tensor): accumulated hidden, flrom lstm, for current obs
            c_t(torch.tensor): accumulated cell, flrom lstm, for current obs
            actions(torch.tensor): next action, whether action or te
            rewards(torch.tensor): rewards acton
            obs_tp1(torch.tensor): next observation, provided from obs, after action
            h_tp1(torch.tensor): accumulated hidden, flrom lstm, for next_obs
            c_tp1(torch.tensor): accumulated cell, flrom lstm, for next_obs
            agent(torch.nn.Module): agent, which samples actions and have nn inside

        Returns: Bellman loss

        """
        n_trails = obs_t.shape[1]
        max_len = obs_t.shape[0]
        qs, _ = agent(x=obs_t, hidden=(h_t, c_t))
        qs = qs.view(max_len, n_trails, -1)
        predicted_q_values = torch.gather(input=qs, index=actions.unsqueeze(2), dim=2).view(-1)

        with torch.no_grad():

            qs_next, _ = agent(x=obs_tp1, hidden=(h_tp1, c_tp1))
            qs_next = qs_next.view(max_len, n_trails, -1)
            actions_idx_next = agent.sample_action_from_q(qs_next, stoch=True)
        predicted_next_q_values = torch.gather(input=qs_next, index=actions_idx_next.unsqueeze(2), dim=2).view(-1)
        q_reference = rewards + self.discount_factor * predicted_next_q_values

        # mean squared error loss to minimize
        loss = t_functional.smooth_l1_loss(predicted_q_values, q_reference).mean()

        return loss

    def get_losses(self, states: State, actions_te: torch.tensor, actions_ac: torch.tensor, rewards_te: torch.tensor,
                   rewards_ac: torch.tensor, next_states: State, is_dones: torch.tensor):
        """
        this function returns all needed losses for train
        Args:
            states: dict('s': torch.tensor, - observation
                   'h_te': torch.tensor, - accumulated hidden, flrom lstm, for current obs, te agent
                   'c_te': torch.tensor, - accumulated cell, flrom lstm, for current obs, te agent
                   'h_ac': torch.tensor, - accumulated hidden, flrom lstm, for current obs, action agent
                   'c_ac': torch.tensor - accumulated cell, flrom lstm, for current obs, action agent
                   )
            actions_te(torch.tensor): prediction for te, indexes
            actions_ac(torch.tensor): prediction for te, indexes
            rewards_te(torch.tensor): rewards te
            rewards_ac(torch.tensor): rewards acton
            next_states(torch.tensor): : dict('s': torch.tensor, - next observation
                   'h_te': torch.tensor, - accumulated hidden, flrom lstm, for next obs, te agent
                   'c_te': torch.tensor, - accumulated cell, flrom lstm, for next obs, te agent
                   'h_ac': torch.tensor, - accumulated hidden, flrom lstm, for next obs, action agent
                   'c_ac': torch.tensor - accumulated cell, flrom lstm, for next obs, action agent
                   )
            is_dones(torch.tensor): tensors with true values if the next_states are final

        Returns: te_agent_loss, ac_agent_loss, provided by self.get_loss_discrete_agent

        """

        loc_device = next(self.te_agent.parameters()).device
        states.to(loc_device)
        next_states.to(loc_device)
        obs_tp0, obs_tp0_h_te, obs_tp0_c_te, obs_tp0_h_ac, obs_tp0_c_ac = states.input_for_nn(is_dones=is_dones)
        obs_tp1, obs_tp1_h_te, obs_tp1_c_te, obs_tp1_h_ac, obs_tp1_c_ac = next_states.input_for_nn(is_dones=is_dones)

        is_dones = is_dones.view(-1)
        actions_te = actions_te.long().view(-1)[is_dones.logical_not()].unsqueeze(1).to(loc_device)
        actions_ac = actions_ac.long().view(-1)[is_dones.logical_not()].unsqueeze(1).to(loc_device)
        rewards_te = rewards_te.float().view(-1)[is_dones.logical_not()].to(loc_device)
        rewards_ac = rewards_ac.float().view(-1)[is_dones.logical_not()].to(loc_device)

        te_agent_loss = self.get_loss_discrete_agent(obs_t=obs_tp0, h_t=obs_tp0_h_te, c_t=obs_tp0_c_te,
                                                     actions=actions_te, rewards=rewards_te,
                                                     obs_tp1=obs_tp1, h_tp1=obs_tp1_h_te, c_tp1=obs_tp1_c_te,
                                                     agent=self.te_agent)

        ac_agent_loss = self.get_loss_discrete_agent(obs_t=obs_tp0, h_t=obs_tp0_h_ac, c_t=obs_tp0_c_ac,
                                                     actions=actions_ac, rewards=rewards_ac,
                                                     obs_tp1=obs_tp1, h_tp1=obs_tp1_h_ac, c_tp1=obs_tp1_c_ac,
                                                     agent=self.ac_agent)
        return te_agent_loss, ac_agent_loss

    def to(self, device: torch.device):
        """
        moves agents to device
        Args:
            device: gpu or cpu

        Returns:

        """
        self.ac_agent.to(device)
        self.te_agent.to(device)
        self.ac_agent_targ.to(device)
        self.te_agent_targ.to(device)
