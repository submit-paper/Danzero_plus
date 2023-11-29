import pickle

import numpy as np
#import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def mlp(sizes, activation, output_activation=nn.Identity,use_init=False):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        if use_init:
            net = nn.Linear(sizes[j], sizes[j+1])
            orthogonal_init(net)
            layers += [net, act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def shared_mlp(obs_dim, sizes, activation,use_init=False):  # 分两个叉，一个是过softmax的logits，另一个不过，就是单纯的q(s,a)，这里是前面的共享层
    layers = []
    shapes = [obs_dim] + list(sizes)
    for j in range(len(shapes) - 1):
        act = activation
        if use_init:
            net = nn.Linear(shapes[j], shapes[j+1])
            orthogonal_init(net)
            layers += [net, act()]
        else:
            layers += [nn.Linear(shapes[j], shapes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, legalaction=torch.tensor(list(range(10))).to(torch.float32)):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, legalaction)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, legal_action):
        logits = torch.squeeze(self.logits_net(obs)) - (1 - legal_action) * 1e6
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPQ(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.q_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.q_net(obs), -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(512, 512, 512, 512, 256), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space
        self.shared = shared_mlp(obs_dim[1], hidden_sizes, activation, use_init=True)
        self.pi = mlp([hidden_sizes[-1], 128, action_space], activation, use_init=True)  # 输出logits
        self.v = mlp([hidden_sizes[-1], 128, 1], activation, use_init=True)  # 输出q(s,a)


    def step(self, obs, legal_action):
        with torch.no_grad():
            shared_feature = self.shared(obs)
            # print(shared_feature.shape, legal_action.shape)
            logits = torch.squeeze(self.pi(shared_feature)) - (1 - legal_action) * 1e8
            pi = Categorical(logits=logits)
            a = pi.sample()
            logp_a = pi.log_prob(a)  # 该动作的log(pi)

            value = torch.squeeze(self.v(shared_feature), -1)
            #print('value', value.shape)
            v = torch.max(value)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def forward(self, obs, act, legal_action):
        shared_feature = self.shared(obs)
        value = torch.squeeze(self.v(shared_feature), -1)
        logits = torch.squeeze(self.pi(shared_feature)) - (1 - legal_action) * 1e8
        pi = Categorical(logits=logits)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a, value
    
    def act(self, obs):
        return self.step(obs)[0]

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return self.state_dict()
    

class MLPQNetwork(nn.Module):
    def __init__(self, observation_space,
                 hidden_sizes=(512, 512, 512, 512, 512), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space

        # build Q function
        self.q = MLPQ(obs_dim, hidden_sizes, activation)

    def load_tf_weights(self, weights):
        name = ['q_net.0.weight', 'q_net.0.bias', 'q_net.2.weight', 'q_net.2.bias', 'q_net.4.weight', 'q_net.4.bias', 'q_net.6.weight', 'q_net.6.bias', 'q_net.8.weight', 'q_net.8.bias', 'q_net.10.weight', 'q_net.10.bias']
        tensor_weights = []
        for weight in weights:
            temp = torch.tensor(weight).T
            tensor_weights.append(temp)
        new_weights = dict(zip(name, tensor_weights))
        self.q.load_state_dict(new_weights)
        print('load tf weights success')

    def get_max_10index(self, data):
        q_list = self.q(data)
        q_list = q_list.cpu().detach().numpy()
        return q_list.argsort()[-10:][::-1]


if __name__ == '__main__':
    model  = MLPActorCritic((10, 568), 1)
    model_q = MLPQNetwork(567)
    b = np.load("/home/zhaoyp/guandan_tog/actor_ppo/debug128.npy", allow_pickle=True).item()
    print(b.keys())
    state = b['x_batch'][0]
    print('allaction', b['actions'][0], type(b['actions'][0]))
    add = 10 * np.ones(shape=(state.shape[0],1))
    state2 = np.append(state, add, axis=1)
    print(state2.shape)
    # print(model.step(torch.tensor(state).to(torch.float32)))
    index2action = model_q.get_max_10index(torch.tensor(state).to(torch.float32))
    print(index2action)
    print(state2[index2action].shape)
    action = model.step(torch.tensor(state2[index2action]).to(torch.float32), torch.tensor([1,1,0,0,0,0,0,0,0,0]).to(torch.float32))
    print('action', action)
    index = index2action[action[0]]
    print(index)
    # s = torch.tensor(state).to(torch.float32)
    # with open('/home/zhaoyp/guandan_tog/actor_torch/q_network.ckpt', 'rb') as f:
    #     new_weights = pickle.load(f)
    # name = ['q_net.0.weight', 'q_net.0.bias', 'q_net.2.weight', 'q_net.2.bias', 'q_net.4.weight', 'q_net.4.bias', 'q_net.6.weight', 'q_net.6.bias', 'q_net.8.weight', 'q_net.8.bias', 'q_net.10.weight', 'q_net.10.bias']
    # tensor_weights = []
    # for weight in new_weights:
    #     temp = torch.tensor(weight).T
    #     tensor_weights.append(temp)
    # weights = dict(zip(name, tensor_weights))
    # print(weights['q_net.0.weight'].shape)
    # print(weights['q_net.10.weight'].shape)
    # model.q.load_state_dict(weights)
    # info = model.q(s)
    # print(info)
    # print('Model state_dict', model.q.state_dict().keys())
    # print(model.q.state_dict()['q_net.0.weight'].shape)
    # print(model.q.state_dict()['q_net.10.weight'].shape)
