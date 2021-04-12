import torch
import gym


def get_next_input(prev_inp, next_act, next_te, column_feature):
    out = prev_inp[:, 1:]
    next_event = torch.zeros(prev_inp.shape[0], prev_inp.shape[2])
    next_event[:, column_feature['te']] = next_te
    last_event = prev_inp[:, -1].squeeze(1)

    next_event[:, column_feature['tt']] = last_event[:, column_feature['tt']] + next_te

    next_event[:, column_feature['tw']] = (last_event[:, column_feature['tw']] + next_te) % (7 * 24 * 60 * 60)
    # one hot transformation from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/5
    act_onehot = torch.FloatTensor(out.shape[0], out.shape[2] - len(column_feature))
    act_onehot.zero_()
    act_onehot.scatter_(1, next_act.long().view(-1, 1), 1)
    next_event[:, len(column_feature):] = act_onehot

    out = torch.cat([out, next_event.unsqueeze(1)], dim=1)
    return out


def get_te_reward(true: torch.tensor, pred: torch.tensor, intervals):
    masks = []
    for inter in intervals:
        true_here = (true >= inter[0]) * (true < inter[1])
        pred_here = (pred >= inter[0]) * (pred < inter[1])
        masks.append(true_here * pred_here)
    out = torch.stack(masks).T.sum(dim=1)
    return out


def get_act_reward(true_act_oh, pred_act_oh):
    mult = (true_act_oh * pred_act_oh)
    return mult.sum(dim=1)


class PMEnv(gym.Env):
    def __init__(self, data: torch.tensor, intervals_te_rew, column_to_time_features, window_size):
        self.data = data
        self.pred_counter = window_size
        self.trace_index = None
        self.intervals = intervals_te_rew
        self.column_feature = column_to_time_features
        self.win = window_size
        self.given_state = None

    def reset(self, trace_n=None):
        self.pred_counter = self.win
        out = self.data[:, :self.win]
        self.given_state = out
        self.trace_index = trace_n
        return out

    def step(self, next_te: torch.tensor, next_act: torch.tensor):
        '''
        returns: next_s, (reward_te, reward_act), is_done, add_inf
        '''
        true_te = self.data[:, self.pred_counter, self.column_feature['te']]
        te_rew = get_te_reward(true=true_te, pred=next_te, intervals=self.intervals)
        # (f'te_reward = \n{te_rew}')
        true_act_oh = self.data[:, self.pred_counter, len(self.column_feature):]

        # TODO here rises an error
        pred_act_oh = torch.zeros(self.data.shape[0], self.data.shape[-1] - len(self.column_feature), dtype=int)
        pred_act_oh[range(pred_act_oh.shape[0]), next_act.long()] = 1

        act_rew = get_act_reward(true_act_oh=true_act_oh, pred_act_oh=pred_act_oh)

        next_s = get_next_input(prev_inp=self.given_state,
                                next_act=next_act,
                                next_te=next_te,
                                column_feature=self.column_feature)
        self.given_state = next_s

        is_done = next_s[:, self.win - 1, len(self.column_feature):].sum(axis=1)
        is_done *= -1
        is_done += 1
        is_done += (self.pred_counter == (self.data.shape[1] - 1))
        is_done = is_done.bool()

        self.pred_counter += 1
        return next_s, (te_rew, act_rew), is_done, {}
