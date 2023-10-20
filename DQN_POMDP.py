import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F
import torch.nn as nn
import gym
import gym_dssat_pdi
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.init as init

part = True

BATCH_SIZE = 64
LR = 0.00003
epsilon_start=1
epsilon_end=0.003
epsilon_decay=0.9991
GAMMA = 0.994
N_ACTIONS = 21
if part :
    N_STATES = 10
else:
    N_STATES = 28
observation_date=5
TARGET_REPLACE_ITER=2500
MEMORY_CAPACITY=100000
ferrate=10
irgrate=0
year = 1965

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Nnormal(s):
    smean = sum(s) / len(s)
    maxs = max(s)-min(s)
    result = [(x - smean) / maxs for x in s]
    return result



class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()

        hz=64
        self.gru = nn.GRU(input_size=N_STATES, hidden_size=hz, batch_first=True)
        self.fc = nn.Linear(hz, N_ACTIONS)
        self.relu = nn.ReLU()
        init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x, h = self.gru(x)
        x = self.relu(x)
        actions_value = self.fc(x)

        return actions_value


class PrioritizedReplayBuffer:
    def __init__(self, capacity=MEMORY_CAPACITY, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)


    def push3(self, pp_s,p_s,s, a, r, s_):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((pp_s,p_s,s, a, r, s_))
        else:
            self.buffer[self.pos] = (pp_s,p_s,s, a, r, s_)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def push5(self, pppp_s,ppp_s,pp_s, p_s, s, a, r, s_):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((pppp_s,ppp_s,pp_s, p_s, s, a, r, s_))
        else:
            self.buffer[self.pos] = (pppp_s,ppp_s,pp_s, p_s, s, a, r, s_)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def push7(self, p7_s,p6_s,pppp_s,ppp_s,pp_s, p_s, s, a, r, s_):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((p7_s,p6_s,pppp_s,ppp_s,pp_s, p_s, s, a, r, s_))
        else:
            self.buffer[self.pos] = (p7_s,p6_s,pppp_s,ppp_s,pp_s, p_s, s, a, r, s_)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        return samples, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = PrioritizedReplayBuffer()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()
        self.eval_net.to(device)
        self.target_net.to(device)

    def choose_action(self, result):
        x = torch.Tensor(result).to(device)
        if np.random.uniform() > EPSILON:
            with torch.no_grad():  
                actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

        
    def store_transition5(self, pppp_s,ppp_s,pp_s, p_s, s, a, r, s_):
        self.memory.push5(pppp_s,ppp_s,pp_s, p_s, s, a, r, s_)
        

    def learnPOMDP5(self):

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        b_memory, indices = self.memory.sample(BATCH_SIZE)
        pppps = [x[0] for x in b_memory]
        ppps = [x[1] for x in b_memory]
        pps = [x[2] for x in b_memory]
        ps = [x[3] for x in b_memory]
        ss = [x[4] for x in b_memory]
        acs = [x[5] for x in b_memory]
        rs = [x[6] for x in b_memory]
        ns = [x[7] for x in b_memory]
        Q_values = []
        Q_Targets = []
        for x in range(len(pps)):
            state = [pppps[x],ppps[x],pps[x], ps[x], ss[x]]
            reward = [rs[x]]
            action = [acs[x]]
            next_state = [ppps[x],pps[x],ps[x], ss[x], ns[x]]
            tss = torch.Tensor(state).to(device)
            ta = torch.LongTensor(np.array(action)).to(device)
            Q_value = self.eval_net(tss).gather(1, ta.unsqueeze(1)).squeeze(1)

            Q_values.append(Q_value)
            next_state_np = np.array(next_state)
            next_state_tensor = torch.Tensor(next_state_np).to(device)
            Q_next = self.target_net(next_state_tensor).detach()
            #Q_next = self.target_net(torch.Tensor(next_state)).detach()
            Q_Next = torch.max(Q_next, 1)[1].cpu().data.numpy()[0]
            Q_Target = reward + GAMMA * Q_Next
            Q_Targets.append(Q_Target.item())

        Q_values_tensor = torch.cat(Q_values).view(BATCH_SIZE, 1).to(device)
        Q_Targets_tensor = torch.Tensor(Q_Targets).view(BATCH_SIZE, 1).to(device)
        loss = self.loss_func(Q_values_tensor, Q_Targets_tensor)

        td_error = Q_Targets_tensor - Q_values_tensor

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, np.abs(td_error.detach().cpu().numpy()))
 
# Create environment
env_args = {
    'mode': 'all',
    #'mode': 'fertilization',
    'seed': 123,
    'random_weather': True,
    'fileX_template_path': './IUAF9901.MZX',
    'experiment_number': 1,
    'auxiliary_file_paths':['./IUAF.CLI']
    }
env=gym.make('GymDssatPdi-v0', **env_args)

dqnnet=Net()

#Train DQN
dqn = DQN()
rwdlist=[]
EPSILON=epsilon_start



for i in range(6000):
    state_list = []
    print('<<<<<<<<<Episode: %s' % i)
    state = env.reset()
    #print('state%s'%i,state)
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
    pstate=Nnormal(pstate)
    episode_reward_sum = 0
    n=0
    icheck=0
    if len(state_list)<1:

        if observation_date==5:
            state_list.append(pstate)
            state_list.append(pstate)
            state_list.append(pstate)
            state_list.append(pstate)
            state_list.append(pstate)


    while True:
        states = state_list[-observation_date:]
        a = dqn.choose_action(states)
       
        atr = {"anfer": a * ferrate, 'amir': a * irgrate}
        state, r, done, info = env.step(atr)
        n+=1
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
            next_state=Nnormal(next_state)
            state_list.append(next_state)

            if observation_date == 5:
                pppps = state_list[-observation_date - 1]
                ppps=state_list[-observation_date]
                pps = state_list[-observation_date+1]
                ps = state_list[-observation_date+2]
                new_r=r
                dqn.store_transition5(pppps,ppps,pps,ps,pstate, a, new_r, next_state)
                episode_reward_sum += new_r
                pstate=next_state
                dqn.learnPOMDP5()


    EPSILON = max(epsilon_end, EPSILON * epsilon_decay)
    rwdlist.append(episode_reward_sum)
    print('Episode: %s' % i, 'rewards=', episode_reward_sum)
    icheck=0





model=dqn.eval_net
if part:
    torch.save(model.state_dict(), str(year)+'R_dqn_part_pomdp.pth')
else:
    torch.save(model.state_dict(), str(year)+'R_dqn_all_pomdp.pth')







fertuse=[]
iruse=[]
rewards=[]
for ii in range(1):
    state_list = []
    print('Testing <<<<<<<<<Episode: %s' % ii)
    state = env.reset()
    icheck=0
    #print('state%s'%i,state)
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
    pstate=Nnormal(pstate)
    episode_reward_sum = 0
    n=0
    if len(state_list)<1:
       
        elif observation_date==5:
            state_list.append(pstate)
            state_list.append(pstate)
            state_list.append(pstate)
            state_list.append(pstate)
            state_list.append(pstate)
    while True:
        states = state_list[-observation_date:]
        a = dqn.choose_action(states)
        atr = {"anfer": a * ferrate,'amir': a * irgrate}
        state, r, done, info = env.step(atr)
        n+=1
        if done:
            break
        istage = state['istage']
        if icheck == 2:
            #episode_reward_sum += r
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
            next_state=Nnormal(next_state)
            state_list.append(next_state)
            fertuse.append(at)
            iruse.append(ar)
            if observation_date == 5:
                pppps = state_list[-observation_date - 1]
                ppps=state_list[-observation_date]
                pps = state_list[-observation_date+1]
                ps = state_list[-observation_date+2]
                new_r=r
                episode_reward_sum += new_r
                pstate=next_state               
    rewards.append(episode_reward_sum)








print('DQN totalfertuse=', sum(fertuse))
print('DQN totaliruse=', sum(iruse))
print('DQN Reward=',episode_reward_sum)
plt.figure(figsize=(20,10))
plt.plot(rwdlist,color="blue",label="reward")
if part:
    plt.savefig(str(year)+'DQN Part POMDP Rewards.jpg')
else:
    plt.savefig(str(year)+'DQN all POMDP Rewards.jpg')

plt.figure(figsize=(20,10))
plt.plot(fertuse,color="blue")
if part:
    plt.savefig(str(year)+'DQN Part POMDP Fertuse.jpg')
else:
    plt.savefig(str(year)+'DQN all POMDP Fertuse.jpg')

plt.figure(figsize=(20,10))
plt.plot(iruse,color="blue")
if part:
    plt.savefig(str(year)+'DQN Part POMDP iruse.jpg')
else:
    plt.savefig(str(year)+'DQN all POMDP iruse.jpg')

plt.figure(figsize=(20, 10))
#plt.plot(Expertfertuse, color="green")
plt.plot(rewards, color="blue")
if part:
    plt.savefig(str(year)+'DQN POMDP part test Rewards.jpg')
else:
    plt.savefig(str(year)+'DQN POMDP all test Rewards.jpg')

# Cleanup
env.close()
