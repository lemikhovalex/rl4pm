import numpy as np
import gym
from .preprocessing import PaperScaler


def get_next_input_replaced_by_pred(prev_inp: np.ndarray, next_act: np.ndarray, next_te: np.ndarray,
                                    column_feature: dict, window_size: int):
    """
    construct a new input, based on previous one. s_t = [(a_{t-1}, a_{t}), (te_{t-1}, te_{t}), ...]
    got predicted a'_{t+1}.
    so s_{t+1} = [(a_{t}, a`_{t+1}), (te`_{t}, te_{t+1}), ...]
    and s_{t+2} = [(a`_{t+1}, a`_{t+2}), (te`_{t+1}, te_{t+2}), ...]
    Args:
        prev_inp: previous inp for nn, shape=[n_traces, window_size, n_features_for_event]
        next_act: predicted next activity, shape=[n_traces]
        next_te: predicted next te, shape=[n_traces]
        column_feature: map from not-one-hot features to idex
        window_size: size of window, sent to NN
    Returns: torch tensor, whith state, extended by next predictions, and shorted,

    """
    out = prev_inp[:, window_size - 1:]
    next_event = np.zeros((prev_inp.shape[0], prev_inp.shape[2]))
    next_event[:, column_feature['te']] = next_te
    last_event = np.squeeze(prev_inp[:, -1], axis=1)

    next_event[:, column_feature['tt']] = last_event[:, column_feature['tt']] + next_te

    next_event[:, column_feature['tw']] = (last_event[:, column_feature['tw']] + next_te) % (7 * 24 * 60 * 60)
    # one hot transformation from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/5
    act_onehot = (np.arange(out.shape[2] - len(column_feature)) == next_act[:, None]).astype(np.float32)
    next_event[:, len(column_feature):] = act_onehot

    out = np.concatenate([out, np.expand_dims(next_event, axis=1)], axis=1)
    return out


def get_te_reward_categorized(true: np.ndarray, pred: np.ndarray, intervals):
    """
    This function recieves tensors with predicted next-time delta intervals. It naturally recieves float tensors
    the reward depends on time intervals. If predicted and truth are in the same interal, then +1, else +0
    Args:
        true: torch tensor of float, predicted te
        pred: torch tensor of float, ground-truth te
        intervals: list of tuples of intervals - intervals for categorized prediction

    Returns:
        torch tensor with rewards for predictions

    """
    masks = []
    for inter in intervals:
        true_here = (true > inter[0]) * (true <= inter[1])
        pred_here = (pred > inter[0]) * (pred <= inter[1])
        masks.append(true_here * pred_here)
    out = np.stack(masks).sum(axis=0)
    return out


def get_ctegorial_reward(true_act_oh: np.ndarray, pred_act_oh: np.ndarray):
    """
    function to recieve reward for categorized features as  +1 if guessed, 0 if not
    args can be swaped
    Args:
        true_act_oh: tensor with true one-hot predictions
        pred_act_oh: tensor with predicted one-hot predictions

    Returns: tensor, reduced dim with one-hots

    """
    assert true_act_oh.shape == pred_act_oh.shape
    mult = (true_act_oh * pred_act_oh)
    return mult.sum(axis=1)


class PMEnv(gym.Env):
    """
    a class, inherited from gym.Env. It is combined env which recieves step - delta time for next step and next action
    basicly it is a cover for data, which just manipulate it
    Args:
        scaler(.preprocessing.PaperScaler): it is needed to process data before it comes to Model
        data(np.ndarray): the data tensor which is parsed trace.
                            data.shape = (n_traces, max_len, n_features_for_event). Last features are one-hot encoded
                            activity. First ones - time related or custom. All of them must present in column_feature
        intervals_te_rew(list[(float, float)]): this env reacts on next event time prediction as for classification
                                                problem - if prediction is in same interval as true value.
                                                So this is a list of turple of two - begin and end of interval
        column_to_time_features(dict): mapping from not-one-hot-feature column to it possition in tensor
        window_size(int): next state is a window of size window_size.
    Attributes:
        scaler(.preprocessing.PaperScaler): it is needed to process data before it comes to Model
        data(np.ndarray): the data tensor which is parsed trace.
                            data.shape = (n_traces, max_len, n_features_for_event). Last features are one-hot encoded
                            activity. First ones - time related or custom. All of them must present in column_feature
        intervals_te_rew(list[(float, float)]): this env reacts on next event time prediction as for classification
                                                problem - if prediction is in same interval as true value.
                                                So this is a list of turple of two - begin and end of interval
        column_to_time_features(dict): mapping from not-one-hot-feature column to it possition in tensor
        window_size(int): next state is a window of size window_size.
        trace_index(int): index which will be must be predicted
        win(int): window size (from arg)
        given_state(np.ndarray): state, which was return from .step(), or .reset() at previous moment.
    """

    def render(self, mode='human'):
        pass

    def __init__(self, data: np.ndarray, intervals_te_rew, column_to_time_features, window_size, scaler=PaperScaler()):
        self.scaler = scaler
        self.data = np.copy(data)
        self.pred_counter = window_size
        self.trace_index = None
        self.intervals = intervals_te_rew
        self.column_feature = column_to_time_features
        self.win = window_size
        self.given_state = None

    def reset(self):
        """
        Basic method to prepare env for processing. Reset env to initial condition
        Note:
            must run it after env creation
        Returns:
            state(torch.tensor): initial state of env

        """
        self.pred_counter = self.win
        out = self.data[:, :self.win]
        self.given_state = out

        out = self.scaler.transform(out, inplace=True)
        return out

    def get_next_input(self, prev_inp: np.ndarray, next_act: np.ndarray, next_te: np.ndarray):
        return get_next_input_replaced_by_pred(prev_inp=prev_inp, next_act=next_act, next_te=next_te,
                                               column_feature=self.column_feature, window_size=self.win)

    def step(self, action: (np.ndarray, np.ndarray)):
        """
        Basic method for interracting with env.
        Returns:
            next_s(torch.tensor), (te_rew(torch.tensor), act_rew(torch.tensor)), is_done(torhc.tensor), add_inf(dict)):
                next_s: next state. shape=(n_traces, window_size, n_features_of_event)
                te_rew, act_rew: rewards for next time and activity prediction, tensors.
                                    shape=(n_traces, 1, 1)
                is_done: bool tensor, indicates, if there is a final event in trace. it can be indicated by one-hot
                            features - all zeros
                add_inf: some additional information

        """
        assert self.pred_counter is not None, 'env was not reset'
        next_te, next_act = action
        true_te = self.data[:, self.pred_counter, self.column_feature['te']]
        te_rew = get_te_reward_categorized(true=true_te, pred=next_te, intervals=self.intervals)
        # (f'te_reward = \n{te_rew}')
        true_act_oh = self.data[:, self.pred_counter, len(self.column_feature):]

        pred_act_oh = np.zeros((self.data.shape[0], self.data.shape[-1] - len(self.column_feature)),
                               dtype=int)
        pred_act_oh[range(pred_act_oh.shape[0]), next_act.astype(int)] = 1

        act_rew = get_ctegorial_reward(true_act_oh=true_act_oh, pred_act_oh=pred_act_oh)
        next_s = self.get_next_input(prev_inp=self.given_state, next_act=next_act, next_te=next_te)
        self.given_state = np.copy(next_s)

        is_done = next_s[:, self.win - 1, len(self.column_feature):].sum(axis=1)
        is_done *= -1
        is_done += 1
        is_done += (self.pred_counter == (self.data.shape[1] - 1))
        is_done = np.array(is_done, dtype=bool)

        self.pred_counter += 1
        # do scaling
        next_s = self.scaler.transform(next_s, inplace=True)

        return next_s, (te_rew, act_rew), is_done, {}


class PMEnvOneStepCons(PMEnv):
    def __init__(self, data: np.ndarray, intervals_te_rew: list, column_to_time_features: dict, window_size: int,
                 scaler=PaperScaler()):
        super(PMEnvOneStepCons, self).__init__(data, intervals_te_rew, column_to_time_features, window_size,
                                               scaler=scaler)

    def get_next_input(self, prev_inp: np.ndarray, next_act: np.ndarray, next_te: np.ndarray):
        return self.data[:, self.pred_counter - self.win:self.pred_counter, :]
