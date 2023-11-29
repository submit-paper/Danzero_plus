import numpy as np
import time
import torch
from utils.mpi_pytorch import (mpi_avg_grads, setup_pytorch_for_mpi,
                                      sync_params)
from utils.mpi_tools import (mpi_avg, mpi_fork, num_procs, proc_id)
from common import get_config_params
from torch.optim import Adam, RMSprop


class PPOAgent:
    def __init__(self, model, clip_ratio=0.2, lr=1e-4, train_iters=20, target_kl=0.01) -> None:
        self.ac = model
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)
        setup_pytorch_for_mpi()
        sync_params(self.ac)
    
    def update(self, data):
        # Set up optimizers for policy and value function

        # pi_l_old, pi_info_old = self.compute_loss_pi(data)
        # pi_l_old = pi_l_old.item()
        # v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for _ in range(self.train_iters):
            self.optimizer.zero_grad()
            loss_pi, loss_v, loss_ent, pi_info = self.compute_loss(data)
            loss = loss_pi + 0.5 * loss_v + 0.05 * loss_ent
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                break
            loss.backward()
            mpi_avg_grads(self.ac)    # average grads across MPI processes
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 10)
            self.optimizer.step()
            time.sleep(0.1)
        #time.sleep(1)

        return {
            'pg_loss': loss_pi.cpu().detach().numpy(),
            'vf_loss': loss_v.cpu().detach().numpy(),
            'entropy': pi_info['ent'],
            'clip_rate': pi_info['cf'],
            'kl': pi_info['kl'],
        }

    # Set up function for computing PPO policy loss
    def compute_loss(self, data):
        obs, act, adv, logp_old, legalaction = torch.tensor(data['obs']).to(torch.float32).to(self.device), torch.tensor(data['act']).to(torch.float32).to(self.device), torch.tensor(data['adv']).to(torch.float32).to(self.device), torch.tensor(data['logp']).to(torch.float32).to(self.device), torch.tensor(data['legal']).to(torch.float32).to(self.device)
        

        # Policy loss
        pi, logp, value = self.ac.forward(obs, act, legalaction)
        ratio = torch.exp(logp - logp_old)
        clipped_ratio = torch.clamp(ratio, 0.0, 3.0)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(clipped_ratio * adv, clip_adv)).mean()

        # value loss
        ret = torch.tensor(data['ret']).to(torch.float32).to(self.device)
        #print('reward shape', ret.shape)
        #print('value', value, 'ret', ret.shape, ret)
        loss_v = ((value - ret) ** 2).mean() * 0.5
        
        # entropy loss
        loss_ent = -1 * pi.entropy().mean()
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, loss_v, loss_ent, pi_info

    def get_weights(self):
        return self.ac.get_weights()

    def export_config(self) -> dict:
        """Export dictionary as configurations"""
        param_dict = {p: getattr(self, p) for p in get_config_params(self)}
        return param_dict


if __name__ == '__main__':
    from model import MLPActorCritic, MLPQNetwork
    device = 'cuda'
    model = MLPActorCritic((10, 567), 1).to(device)
    model_q = MLPQNetwork(567).to(device)
    #ppoagent = PPOAgent(model)
    b = np.load("/home/zhaoyp/guandan_tog/actor_ppo/debug128.npy", allow_pickle=True).item()
    state = b['x_batch'][0]
    print(b['actions'])
    index2action = model_q.get_max_10index(torch.tensor(state).to(torch.float32).to(device))
    state = state[index2action]
    obs = {'obs': state, 'act': b['actions'][0], 'logp': b['neglogps'][0], 
        'adv': b['returns'][0]-b['values'][0], 'ret': b['returns'][0]}
    print(obs, obs['obs'].shape)
    #info = ppoagent.update(obs)
    #print(info)
    # print(torch.cuda.is_available())
