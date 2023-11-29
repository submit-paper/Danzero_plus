import time
from argparse import ArgumentParser
from multiprocessing import Process
from random import randint

import numpy as np
import zmq
import pickle
import torch
import io
from model import MLPActorCritic, MLPQNetwork
from pyarrow import deserialize, serialize

ActionNumber = 2

parser = ArgumentParser()
parser.add_argument('--ip', type=str, default='172.15.15.2',
                    help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000,
                    help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001,
                    help='Learner server port to subscribe model parameters')
parser.add_argument('--exp_path', type=str, default='/home/root/log',
                    help='Directory to save logging data, model parameters and config file')
parser.add_argument('--num_saved_ckpt', type=int, default=4,
                    help='Number of recent checkpoint files to be saved')
parser.add_argument('--observation_space', type=int, default=(567,),
                    help='The YAML configuration file')
parser.add_argument('--action_space', type=int, default=(5, 216),
                    help='The YAML configuration file')
parser.add_argument('--epsilon', type=float, default=0.01,
                    help='Epsilon')
parser.add_argument('--iter', type=int, default=0,
                    help='update steps for the tested model')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class Player():
    def __init__(self, args) -> None:
        # 模型初始化
        self.model_id = args.iter * 2000 + 500
        self.model = MLPActorCritic((ActionNumber, 516+ActionNumber * 54), ActionNumber)
        with open('./models/ppo{}.pth'.format(self.model_id), 'rb') as f:
            new_weights = CPU_Unpickler(f).load()
        print('load model:', self.model_id)
        self.model.set_weights(new_weights)
        self.model_q = MLPQNetwork(567)
        with open('./q_network.ckpt', 'rb') as f:
            tf_weights = pickle.load(f)
        self.model_q.load_tf_weights(tf_weights)

    def sample(self, state) -> int:
        states = state['x_batch']
        legal_action = ActionNumber
        legal_index = np.ones(ActionNumber)
        state_no_action = state['x_no_action']
        if len(states) >= ActionNumber:
            indexs = self.model_q.get_max_n_index(states, ActionNumber)
            dqn_states = np.asarray(states[indexs])
            top_actions = dqn_states[:, -54:].flatten()
            states = np.concatenate((state_no_action, top_actions))

        elif len(states) < ActionNumber:
            legal_action = len(states)
            legal_index[legal_action:] = np.zeros(ActionNumber-legal_action)
            top_indexs = self.model_q.get_max_n_index(states, ActionNumber)
            dqn_states = np.asarray(states[top_indexs])
            top_actions = dqn_states[:,-54:].flatten()
            states = np.concatenate((state_no_action, top_actions)) # 把动作先添加进来
            supple = np.zeros(54 * (ActionNumber - legal_action))
            states = np.concatenate((states,supple))
            indexs = list(range(ActionNumber))

        action = self.model.step(states, legal_index)
        return indexs[action]


def run_one_player(index, args):
    player = Player(args)

    # 初始化zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://*:{6000+index}')

    action_index = 0
    while True:
        state = deserialize(socket.recv())
        action_index = player.sample(state)
        # print(f'actor{index} do action number {action_index}')
        socket.send(serialize(action_index).to_buffer())


def main():
    # 参数传递
    args, _ = parser.parse_known_args()

    def exit_wrapper(index, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_player(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(players):
                    if _i != index:
                        _p.terminate()

    players = []
    for i in [1, 3]:
        # print(f'start{i}')
        p = Process(target=exit_wrapper, args=(i, args))
        p.start()
        time.sleep(0.5)
        players.append(p)

    for player in players:
        player.join()


if __name__ == '__main__':
    main()
