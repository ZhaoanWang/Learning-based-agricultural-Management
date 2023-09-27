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
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


part=True



# DQN

BATCH_SIZE = 640
LR = 0.00003
epsilon_start=1
epsilon_end=0
epsilon_decay=0.9991
GAMMA = 0.993
TARGET_REPLACE_ITER = 2500
MEMORY_CAPACITY = 150000
if part :
    N_STATES = 10
else:
    N_STATES = 28
N_ACTIONS=21
ferrate=10
irgrate=0
#beta = 0.01
q_table_actions = 0
random_actions = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

year=1987

class Net(nn.Module):
    def __init__(self):
        hl=256
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, hl)
        self.bn1 = nn.LayerNorm(hl)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hl, hl)
        self.bn2 = nn.LayerNorm(hl)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hl, hl)
        self.bn3 = nn.LayerNorm(hl)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(hl, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        actions_value = self.out(x)
        return actions_value

class PrioritizedReplayBuffer:
    def __init__(self, capacity=MEMORY_CAPACITY, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state)

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
        self.eval_net= Net()
        self.target_net = Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = PrioritizedReplayBuffer()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()
        self.eval_net.to(device)
        self.target_net.to(device)

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        if np.random.uniform() > EPSILON:
            with torch.no_grad():
                actions_value1 = self.eval_net.forward(x)
            action1 = torch.max(actions_value1, 1)[1].data.cpu().numpy()
            action1 = action1[0]
        else:
            action1 = np.random.randint(0, N_ACTIONS)
        return action1


    def store_transition(self, s, a, r, s_):
        self.memory.push(s, a, r, s_)


    def learn(self):
       
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        b_memory, indices = self.memory.sample(BATCH_SIZE)
        b_s = torch.FloatTensor([x[0] for x in b_memory]).to(device)
        b_a = torch.LongTensor([[x[1]] for x in b_memory]).to(device)
        b_r = torch.FloatTensor([[x[2]] for x in b_memory]).to(device)
        b_s_ = torch.FloatTensor([x[3] for x in b_memory]).to(device)


        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        td_error = q_target - q_eval

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, np.abs(td_error.detach().cpu().numpy()))

def Nnormal(s):
    smean = sum(s) / len(s)
    maxs = max(s)-min(s)
    result = [(x - smean) / maxs for x in s]
    return result


# Create environment
env_args = {
    'mode': 'all',
    #'mode': 'fertilization',
    'seed': 123,
    'random_weather': False,
    'fileX_template_path': './IUAF9901.MZX',
    'experiment_number': 1,
    #'auxiliary_file_paths':['./IUAF.CLI']
    }
env_dqn= gym.make('GymDssatPdi-v0', **env_args)

#Train DQN
dqn = DQN()
rwdlist=[]
EPSILON=epsilon_start
moving_average_rewards = []

for i in range(6000):
    print('<<<<<<<<<Episode: %s' % i)
    state = env_dqn.reset()


    if part:

        s = [state['cumsumfert'], state['dap'],
            state['istage'], state['pltpop'], state['rain'], state['sw'][1], state['tmax'],
            state['tmin'], state['vstage'], state['xlai']]
    else:
        s = [state['cleach'], state['cnox'], state['cumsumfert'], state['dap'], state['dtt'], state['es'],
             state['grnwt'],
             state['istage'], state['nstres'], state['pcngrn'], state['pltpop'], state['rain'], state['rtdep'],
             state['runoff'], state['srad'], state['sw'][1], state['swfac'], state['tleachd'], state['tmax'],
             state['tmin'],
             state['tnoxd'], state['topwt'], state['totir'], state['trnu'], state['vstage'], state['wtdep'],
             state['wtnup'], state['xlai']]
    s = Nnormal(s)
    episode_reward_sum = 0
    n=0
    done = False
    icheck=0

    while not done:
        a = dqn.choose_action(s)
   
        atr = {"anfer": a * ferrate,'amir': a * irgrate}
        nstate, r, done, info = env_dqn.step(atr)

        if done:
            break
        istage = nstate['istage']
        if istage == 6:
            icheck += 1
            print(r)
        if icheck == 2:
            episode_reward_sum += r

            break
        else:
            weight=nstate['grnwt']
            if part:
                s_ = [nstate['cumsumfert'], nstate['dap'],
                      nstate['istage'], nstate['pltpop'], nstate['rain'], nstate['sw'][1], nstate['tmax'],
                      nstate['tmin'], nstate['vstage'], nstate['xlai']]
            else:
                s_ = [nstate['cleach'], nstate['cnox'], nstate['cumsumfert'], nstate['dap'], nstate['dtt'],
                      nstate['es'],
                      nstate['grnwt'],
                      nstate['istage'], nstate['nstres'], nstate['pcngrn'], nstate['pltpop'], nstate['rain'],
                      nstate['rtdep'],
                      nstate['runoff'], nstate['srad'], nstate['sw'][1], nstate['swfac'], nstate['tleachd'],
                      nstate['tmax'],
                      nstate['tmin'],
                      nstate['tnoxd'], nstate['topwt'], nstate['totir'], nstate['trnu'], nstate['vstage'],
                      nstate['wtdep'],
                      nstate['wtnup'], nstate['xlai']]

            s_ = Nnormal(s_)
            dqn.store_transition(s, a, r, s_)
            episode_reward_sum += r
            s = s_
            dqn.learn()

    print('Episode: %s' % i,'rewards=',episode_reward_sum)
    q_table_actions = 0
    random_actions = 0
    EPSILON = max(epsilon_end, EPSILON * epsilon_decay)
    print('EPSILON=',EPSILON)
    rwdlist.append(episode_reward_sum)
    icheck=0

model=dqn.eval_net
if part:
    torch.save(model.state_dict(), str(year)+'Rdqn_part_mdp.pth')
else:
    torch.save(model.state_dict(), str(year)+'Rdqn_full_mdp.pth')


DQN_reward = []
Expert_reward = []
DQN_grnwt = []
Expert_grnwt = []
DQN_topwt = []
Expert_topwt = []
DQN_cumsumfert = []
Expert_cumsumfert = []
DQNiruse=[]
print('Start Testing!')
for tt in range(1):
    # DQN Test
    state = env_dqn.reset()

    if part:
        s = [state['cumsumfert'], state['dap'],
             state['istage'], state['pltpop'], state['rain'], state['sw'][1], state['tmax'],
             state['tmin'], state['vstage'], state['xlai']]
    else:
        s = [state['cleach'], state['cnox'], state['cumsumfert'], state['dap'], state['dtt'], state['es'],
             state['grnwt'],
             state['istage'], state['nstres'], state['pcngrn'], state['pltpop'], state['rain'], state['rtdep'],
             state['runoff'], state['srad'], state['sw'][1], state['swfac'], state['tleachd'], state['tmax'],
             state['tmin'],
             state['tnoxd'], state['topwt'], state['totir'], state['trnu'], state['vstage'], state['wtdep'],
             state['wtnup'], state['xlai']]
    s = Nnormal(s)
    episode_reward_sum = 0
    EPSILON = 0
    DQNfertuse = []
    DQNiruse = []
    done = False
    icheck=0
    while not done:
        a = dqn.choose_action(s)
        atr = {"anfer": a * ferrate,'amir': a * irgrate}
        state, r, done, info = env_dqn.step(atr)
        if done:
            break
        istage=state['istage']
        if istage == 6:
            icheck +=1
        if icheck == 2:
            break
        else:

            at = atr['anfer']
            DQNfertuse.append(at)
            ferttotle = state['cumsumfert']
            new_r = r
            episode_reward_sum += new_r
            weight=state['grnwt']
            if part:
                state = [state['cumsumfert'], state['dap'],
                         state['istage'], state['pltpop'], state['rain'], state['sw'][1], state['tmax'],
                         state['tmin'], state['vstage'], state['xlai']]
            else:
                state = [state['cleach'], state['cnox'], state['cumsumfert'], state['dap'], state['dtt'], state['es'],
                         state['grnwt'],
                         state['istage'], state['nstres'], state['pcngrn'], state['pltpop'], state['rain'],
                         state['rtdep'],
                         state['runoff'], state['srad'], state['sw'][1], state['swfac'], state['tleachd'],
                         state['tmax'],
                         state['tmin'],
                         state['tnoxd'], state['topwt'], state['totir'], state['trnu'], state['vstage'], state['wtdep'],
                         state['wtnup'], state['xlai']]
            state = Nnormal(state)
            s = state
    DQN_reward.append(episode_reward_sum)
    DQN_cumsumfert.append(ferttotle)
   
for pp in (
DQN_reward,
):
    print(' mean=', mean(pp))

plt.figure(figsize=(20, 10))
plt.plot(DQNfertuse, color="blue")
if part:
    plt.savefig(str(year)+'MDP DQN part Fert.jpg')
else:
    plt.savefig(str(year)+'MDP DQN  all Fert.jpg')


plt.figure(figsize=(20, 10))
plt.plot(DQN_reward, color="blue")
if part:
    plt.savefig(str(year)+'MDP DQN part reward.jpg')
else:
    plt.savefig(str(year)+'MDP DQN all reward.jpg')


plt.figure(figsize=(20,10))
plt.plot(rwdlist,color="blue")
if part:
    plt.savefig(str(year)+'MDP DQN part Training Rewards.jpg')
else:
    plt.savefig(str(year)+'MDP DQN all Training Rewards.jpg')
# Cleanup
env.close()
