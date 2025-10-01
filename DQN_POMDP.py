import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union
from gym import spaces
from torch.nn import functional as F
import torch.nn as nn
import gym
import gym_dssat_pdi
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.init as init
import random
import os


part = True

BATCH_SIZE = 640
LR = 0.00003
epsilon_start = 1.0
epsilon_end = 0.003
epsilon_decay = 0.9991
GAMMA = 0.994
N_ACTIONS = 21
if part:
    N_STATES = 10
else:
    N_STATES = 28
observation_date = 5
TARGET_REPLACE_ITER = 2500
MEMORY_CAPACITY = 100000
ferrate = 10
irgrate = 0
year = 1965

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def Nnormal(s):
    a = np.asarray(s, dtype=np.float32)
    if a.size == 0:
        return [0.0]
    m = float(a.mean())
    r = float(a.max() - a.min())
    if r < 1e-8:
        r = 1.0
    return ((a - m) / r).tolist()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hz = 64
        self.gru = nn.GRU(input_size=N_STATES, hidden_size=hz, batch_first=True)
        self.fc = nn.Linear(hz, N_ACTIONS)
        self.relu = nn.ReLU()
        init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        if x.dim() == 2:  # (T, N_STATES) -> (1, T, N_STATES)
            x = x.unsqueeze(0)
        out, _ = self.gru(x)           # (B, T, hz)
        last = out[:, -1, :]           # (B, hz)
        last = self.relu(last)
        return self.fc(last)           # (B, N_ACTIONS)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=MEMORY_CAPACITY, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def push5(self, pppp_s, ppp_s, pp_s, p_s, s, a, r, s_):
        max_prio = self.priorities.max() if self.buffer else 1.0
        transition = (pppp_s, ppp_s, pp_s, p_s, s, a, r, s_)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta: float = 0.4):
        size = len(self.buffer)
        if size == 0:
            raise ValueError("Replay buffer is empty, cannot sample.")
        prios = self.priorities[: size] if size < self.capacity else self.priorities
        prios = np.maximum(prios, 1e-8)
        probs = prios ** self.prob_alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum
        indices = np.random.choice(size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = float(prio)


class DQN(object):
    def __init__(self):
        self.eval_net = Net().to(device)
        self.target_net = Net().to(device)
        self.learn_step_counter = 0
        self.memory = PrioritizedReplayBuffer()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()
        self.epsilon = epsilon_start

    @torch.no_grad()
    def choose_action(self, seq_states):

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, N_ACTIONS)
        x = torch.as_tensor(np.array(seq_states, dtype=np.float32), device=device)
        q = self.eval_net(x)  # (1, N_ACTIONS)
        return int(q.argmax(dim=1).item())

    def store_transition5(self, pppp_s, ppp_s, pp_s, p_s, s, a, r, s_):
        self.memory.push5(pppp_s, ppp_s, pp_s, p_s, s, a, r, s_)

    def soft_update_target(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def learnPOMDP5(self):
        if len(self.memory) < BATCH_SIZE:
            return

        self.soft_update_target()
        self.learn_step_counter += 1

        b_memory, indices = self.memory.sample(BATCH_SIZE)
        pppps = [x[0] for x in b_memory]
        ppps  = [x[1] for x in b_memory]
        pps   = [x[2] for x in b_memory]
        ps    = [x[3] for x in b_memory]
        ss    = [x[4] for x in b_memory]
        acs   = [x[5] for x in b_memory]
        rs    = [x[6] for x in b_memory]
        ns    = [x[7] for x in b_memory]

        # 构造批量 (B, 5, N_STATES)
        states_batch = torch.as_tensor(
            [[pppps[i], ppps[i], pps[i], ps[i], ss[i]] for i in range(BATCH_SIZE)],
            dtype=torch.float32, device=device
        )
        next_states_batch = torch.as_tensor(
            [[ppps[i], pps[i], ps[i], ss[i], ns[i]] for i in range(BATCH_SIZE)],
            dtype=torch.float32, device=device
        )
        actions_batch = torch.as_tensor(acs, dtype=torch.long, device=device)      # (B,)
        rewards_batch = torch.as_tensor(rs,  dtype=torch.float32, device=device)   # (B,)

        # 前向（批量）
        q_eval = self.eval_net(states_batch)                                       # (B, N_ACTIONS)
        q_taken = q_eval.gather(1, actions_batch.unsqueeze(1)).squeeze(1)         # (B,)

        with torch.no_grad():
            q_next = self.target_net(next_states_batch).max(dim=1).values          # (B,)
            q_target = rewards_batch + GAMMA * q_next

        loss = self.loss_func(q_taken, q_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        td_error = (q_target - q_taken).detach().abs()
        self.memory.update_priorities(indices, td_error.flatten().cpu().numpy())

        # 训练过程偶尔打印显存（确认 GPU 使用）
        if self.learn_step_counter % 200 == 0:
            alloc = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
            print(f"[learn {self.learn_step_counter}] loss={loss.item():.4f}, mem={alloc/(1024**2):.1f}MB")


env_args = {
    'mode': 'all',
    # 'mode': 'fertilization',
    'seed': 123,
    'random_weather': False,
    'fileX_template_path': './IUAF9901.MZX',
    'experiment_number': 1,
    'auxiliary_file_paths': ['./IUAF.CLI']
}
env = gym.make('GymDssatPdi-v0', **env_args)



dqn = DQN()
rwdlist = []

print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)
print("Model device:", next(dqn.eval_net.parameters()).device)
if torch.cuda.is_available():
    print("Allocated before training (MB):", torch.cuda.memory_allocated()/(1024**2))

for i in range(6000):
    state_list = []
    print('<<<<<<<<<Episode:', i)
    state = env.reset()

    if part:
        pstate = [state['cumsumfert'], state['dap'],
                  state['istage'], state['pltpop'], state['rain'], state['sw'][1], state['tmax'],
                  state['tmin'], state['vstage'], state['xlai']]
    else:
        pstate = [state['cleach'], state['cnox'], state['cumsumfert'], state['dap'], state['dtt'], state['es'],
                  state['grnwt'],
                  state['istage'], state['nstres'], state['pcngrn'], state['pltpop'], state['rain'], state['rtdep'],
                  state['runoff'], state['srad'], state['sw'][1], state['swfac'], state['tleachd'], state['tmax'],
                  state['tmin'],
                  state['tnoxd'], state['topwt'], state['totir'], state['trnu'], state['vstage'], state['wtdep'],
                  state['wtnup'], state['xlai']]
    pstate = Nnormal(pstate)
    episode_reward_sum = 0.0
    n = 0
    icheck = 0


    if len(state_list) < 1:
        for _ in range(observation_date):
            state_list.append(pstate)

    while True:
        states = state_list[-observation_date:]  # 长度 5
        a = dqn.choose_action(states)

        atr = {"anfer": a * ferrate, 'amir': a * irgrate}
        state, r, done, info = env.step(atr)
        n += 1
        if done:
            break

        istage = state['istage']
        if icheck == 2:
            break
        else:
            if istage == 6:
                icheck += 1

            if part:
                next_state = [state['cumsumfert'], state['dap'],
                              state['istage'], state['pltpop'], state['rain'], state['sw'][1], state['tmax'],
                              state['tmin'], state['vstage'], state['xlai']]
            else:
                next_state = [state['cleach'], state['cnox'], state['cumsumfert'], state['dap'], state['dtt'], state['es'],
                              state['grnwt'],
                              state['istage'], state['nstres'], state['pcngrn'], state['pltpop'], state['rain'], state['rtdep'],
                              state['runoff'], state['srad'], state['sw'][1], state['swfac'], state['tleachd'], state['tmax'],
                              state['tmin'],
                              state['tnoxd'], state['topwt'], state['totir'], state['trnu'], state['vstage'], state['wtdep'],
                              state['wtnup'], state['xlai']]
            next_state = Nnormal(next_state)
            state_list.append(next_state)

            if observation_date == 5:
                pppps = state_list[-observation_date - 1]
                ppps  = state_list[-observation_date]
                pps   = state_list[-observation_date + 1]
                ps    = state_list[-observation_date + 2]

                dqn.store_transition5(pppps, ppps, pps, ps, pstate, a, float(r), next_state)
                episode_reward_sum += float(r)
                pstate = next_state

                dqn.learnPOMDP5()


    dqn.epsilon = max(epsilon_end, dqn.epsilon * epsilon_decay)
    rwdlist.append(episode_reward_sum)
    print(f'Episode: {i} rewards={episode_reward_sum:.4f}, epsilon={dqn.epsilon:.4f}')


model = dqn.eval_net
save_name = f"{year}R_dqn_{'part' if part else 'all'}_pomdp.pth"
torch.save(model.state_dict(), save_name)
print("Saved:", save_name)


fertuse = []
iruse = []
rewards = []

for ii in range(1):
    state_list = []
    print('Testing <<<<<<<<<Episode:', ii)
    state = env.reset()
    icheck = 0
    if part:
        pstate = [state['cumsumfert'], state['dap'],
                  state['istage'], state['pltpop'], state['rain'], state['sw'][1], state['tmax'],
                  state['tmin'], state['vstage'], state['xlai']]
    else:
        pstate = [state['cleach'], state['cnox'], state['cumsumfert'], state['dap'], state['dtt'], state['es'],
                  state['grnwt'],
                  state['istage'], state['nstres'], state['pcngrn'], state['pltpop'], state['rain'], state['rtdep'],
                  state['runoff'], state['srad'], state['sw'][1], state['swfac'], state['tleachd'], state['tmax'],
                  state['tmin'],
                  state['tnoxd'], state['topwt'], state['totir'], state['trnu'], state['vstage'], state['wtdep'],
                  state['wtnup'], state['xlai']]
    pstate = Nnormal(pstate)
    episode_reward_sum = 0.0
    n = 0

    if len(state_list) < 1:
        for _ in range(observation_date):
            state_list.append(pstate)

    test_eps = 0.0

    while True:
        states = state_list[-observation_date:]
        ori_eps = dqn.epsilon
        dqn.epsilon = test_eps
        a = dqn.choose_action(states)
        dqn.epsilon = ori_eps

        atr = {"anfer": a * ferrate, 'amir': a * irgrate}
        state, r, done, info = env.step(atr)
        n += 1
        if done:
            break
        istage = state['istage']
        if icheck == 2:
            break
        else:
            if istage == 6:
                icheck += 1
            if part:
                next_state = [state['cumsumfert'], state['dap'],
                              state['istage'], state['pltpop'], state['rain'], state['sw'][1], state['tmax'],
                              state['tmin'], state['vstage'], state['xlai']]
            else:
                next_state = [state['cleach'], state['cnox'], state['cumsumfert'], state['dap'], state['dtt'], state['es'],
                              state['grnwt'],
                              state['istage'], state['nstres'], state['pcngrn'], state['pltpop'], state['rain'], state['rtdep'],
                              state['runoff'], state['srad'], state['sw'][1], state['swfac'], state['tleachd'], state['tmax'],
                              state['tmin'],
                              state['tnoxd'], state['topwt'], state['totir'], state['trnu'], state['vstage'], state['wtdep'],
                              state['wtnup'], state['xlai']]
            at = atr['anfer']
            ar = atr['amir']
            next_state = Nnormal(next_state)
            state_list.append(next_state)
            fertuse.append(at)
            iruse.append(ar)
            episode_reward_sum += float(r)
            pstate = next_state
    rewards.append(episode_reward_sum)

print('DQN totalfertuse=', sum(fertuse))
print('DQN totaliruse=', sum(iruse))
print('DQN Reward=', rewards[-1] if len(rewards) > 0 else 0.0)


plt.figure(figsize=(20, 10))
plt.plot(rwdlist, label="reward")
plt.legend()
plt.savefig(f"{year} DQN POMDP Rewards.jpg")

plt.figure(figsize=(20, 10))
plt.plot(fertuse)
plt.savefig(f"{year} DQN POMDP Fertuse.jpg")

plt.figure(figsize=(20, 10))
plt.plot(iruse)
plt.savefig(f"{year} DQN POMDP iruse.jpg")

plt.figure(figsize=(20, 10))
plt.plot(rewards)
plt.savefig(f"{year} DQN POMDP test Rewards.jpg")

# Cleanup
env.close()
