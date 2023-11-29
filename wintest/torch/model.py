import numpy as np
#import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def shared_mlp(obs_dim, sizes, activation):  # 分两个叉，一个是过softmax的logits，另一个不过，就是单纯的q(s,a)，这里是前面的共享层
    layers = []
    shapes = [obs_dim] + list(sizes)
    for j in range(len(shapes) - 1):
        act = activation
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
        self.shared = shared_mlp(obs_dim[1], hidden_sizes, activation)
        self.pi = mlp([hidden_sizes[-1], 128, action_space], activation)  # 输出logits
        self.v = mlp([hidden_sizes[-1], 128, 1], activation)    # 输出q(s,a)


    def step(self, obs, legal_action):
        obs = torch.tensor(obs).to(torch.float32)
        legal_action = torch.tensor(legal_action).to(torch.float32)
        with torch.no_grad():
            shared_feature = self.shared(obs)
            # print(shared_feature.shape, legal_action.shape)
            logits = torch.squeeze(self.pi(shared_feature)) - (1 - legal_action) * 1e8
            a = torch.argmax(logits)
        # del obs, legal_action
        # return a.numpy().item(), v.numpy().item(), logp_a.numpy().item()
        return a.numpy()

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

    def get_max_n_index(self, data, n):
        #data = data[:,:-3]
        q_list = self.q(torch.tensor(data).to(torch.float32))
        q_list = q_list.detach().numpy()
        return q_list.argsort()[-n:][::-1].tolist()

if __name__ == '__main__':
    model  = MLPActorCritic((10, 567), 1)
    model_q = MLPQNetwork(567)
    b = np.load("/home/zhaoyp/guandan_tog/actor_ppo/debug128.npy", allow_pickle=True).item()
    print(b.keys())
    state = b['x_batch'][0]
    n = 3
    index2action = model_q.get_max_n_index(torch.tensor(state).to(torch.float32),n)
    
    # state = np.random.random((513, ))
    # action1 = np.random.random((54, ))
    # action2 = np.random.random((54, ))
    # action3 = np.random.random((54, ))
    # b = np.load("/home/zhaoyp/guandan_tog_tog/actor_torch/debug145.npy", allow_pickle=True).item()
    # print(b.keys())
    # print(b['obs_cut'].shape)
    # print(b['obs'].shape)
    
    # print('time1')
    # objgraph.show_most_common_types(limit=30)
    # objgraph.show_growth()


    # print('time2')
    # objgraph.show_most_common_types(limit=30)
    # objgraph.show_growth()

    # a, v, p = model.step(state, legal_index)

    # print('time3')
    # objgraph.show_most_common_types(limit=30)
    # objgraph.show_growth()

    # print(a,v,p)
    # print(type(a),type(v),type(p))
