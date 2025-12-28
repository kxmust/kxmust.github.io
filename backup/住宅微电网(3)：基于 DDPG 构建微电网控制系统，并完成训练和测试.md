**说明**：本期构建的 DDPG 算法是最基础的版本，难以实现优异的调度结果，并且对超参数非常敏感，难以调优。本文旨在介绍深度强化学习算法在微电网能源管理中的应用，如果需要实现更加优异的性能，可以从以下内容进行优化：
1. 设计更加优异的奖励函数。
2. 引入更好的探索噪声。
3. 引入优先经验回放。
4. 优化网络架构，比如加入循环神经网络来提高抽取特征的能力。
5. 优化超参数。

[完整代码](https://github.com/kxmust/residential-microgrid-ddpg.git)


## 1. 构建DDPG算法
DDPG算法是一个常用的 Actor-Critic 算法，具体过程不过多介绍，直接给出代码：
```python
import torch
import torch.nn.functional as F
import collections
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.fc1.bias, 0.1)

        self.fc_out = torch.nn.Linear(hidden_dim, action_dim)
        torch.nn.init.kaiming_uniform_(self.fc_out.weight, nonlinearity='tanh')
        torch.nn.init.constant_(self.fc_out.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = F.tanh(self.fc_out(x))
        return action

class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.fc1.bias, 0.1)

        self.fc_out = torch.nn.Linear(hidden_dim, 1)
        torch.nn.init.uniform_(self.fc_out.weight, -0.003, 0.003)
        torch.nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        q = self.fc_out(x)
        return q

class DDPG:
    """ DDPG算法 """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma,
                 actor_lr, critic_lr, tau, gamma, device, is_train):
        # 构建 DDPG 网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 构建优化器
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=actor_lr, momentum=0.9)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=critic_lr, momentum=0.9)

        # 其他参数
        self.action_bound = action_bound
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.is_train = is_train
        self.noise_decay = 0.99
        self.std = 0.1
        self.train_num = 0
        self.min_sigma = 0.01
        self.initial_sigma = sigma

        # 经验池
        self.replay_buffer = ReplayBuffer(capacity=50000)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().squeeze(0).numpy()

        # 如果是训练，则给动作添加噪声，增加探索
        if self.is_train:
            noise = np.random.randn() * self.sigma
            action += noise
            action = np.clip(action, -1, 1)

        return action * self.action_bound

    def decay_noise(self):
        # 噪声衰减
        self.sigma = max(self.min_sigma, self.sigma * self.noise_decay)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if (self.train_num + 1) % 10 == 0:
            self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
            self.soft_update(self.critic, self.target_critic)  # 软更新价值网络
```

## 2. 构建训练函数和测试函数
```python
from DDPG import DDPG
from residential_env import MicrogridEnv
from get_data import rm_data
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(0)
torch.manual_seed(0)

# 构建微电网环境
env = MicrogridEnv(rm_data, episode_length=48)

# 基于 DDPG 构建控制系统
control_agent = DDPG(
    state_dim = 4,    # 输入四个状态，[负载, 光伏, 电价, SOC]
    action_dim = 1,   # 输出电池充放电功率
    hidden_dim = 64,
    action_bound = env.action_bound,
    sigma = 0.3,
    actor_lr = 3e-4,
    critic_lr = 4e-4,
    gamma = 0.99,
    tau = 0.8,
    is_train = 1,
    device = device,
)

def train_agent(env, agent, num_episodes, minimal_size, batch_size, is_plt = False):
    return_list = []
    env.is_train = True
    agent.is_train = True

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(env._normalize(state)).item()
                    next_state, reward, done, _, _ = env.step(action)
                    agent.replay_buffer.add(env._normalize(state), action/env.action_bound,
                                                    reward, env._normalize(next_state), done)
                    state = next_state
                    episode_return += reward
                    if agent.replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = agent.replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)

                agent.train_num += 1
                if (agent.train_num + 1) % 25 == 0:
                    agent.decay_noise()

                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),
                                      'sigma': '%.3f' % agent.sigma})


                pbar.update(1)
    torch.save(agent.actor.state_dict(), 'Model_Save/actor_weights.pth')
    if is_plt:
        # 平滑训练曲线
        def moving_average(a, window_size):
            cumulative_sum = np.cumsum(np.insert(a, 0, 0))
            middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
            r = np.arange(1, window_size - 1, 2)
            begin = np.cumsum(a[:window_size - 1])[::2] / r
            end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
            return np.concatenate((begin, middle, end))

        episodes_list = list(range(len(return_list)))
        mv_return = moving_average(return_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DDPG')
        plt.savefig('plots/DDPG_train.png')
        plt.show()

    return return_list

def test_microgrid(env, agent, test_time, plot=True):
    state, _ = env.reset()
    done = False
    total_cost = 0.0

    # 记录各变量
    records = {
        't': [],
        'load': [],
        'pv': [],
        'ug_price': [],
        'soc': [],
        'battery_power': [],
        'grid_power': [],
        'reward': [],
        'cost': [],
    }

    while not done:
        # 策略输出动作
        action = agent.take_action(env._normalize(state)).item()
        next_state, reward, done, _, _ = env.step(action)

        # 解析状态
        load, pv, ug_price, soc = state
        power_battery = env._clip_battery_action(action, soc)

        Pe = load - pv - power_battery  # 电网交互功率

        if Pe >= 0:
            cost = Pe * ug_price
        else:
            cost = Pe * ug_price * 0.7

        total_cost += cost

        # 保存记录
        records['t'].append(env.len_t)
        records['load'].append(load)
        records['pv'].append(pv)
        records['ug_price'].append(ug_price)
        records['soc'].append(soc)
        records['battery_power'].append(power_battery)
        records['grid_power'].append(Pe)
        records['reward'].append(reward)
        records['cost'].append(cost)

        state = next_state

        if env.len_t >= test_time-1:
            done = True

    # 转成 numpy 数组
    for k in records:
        records[k] = np.array(records[k])

    print(f"\n✅ 测试完成，总电费成本为：{total_cost:.2f} 元, SOC均值为:{np.mean(records['soc']):.2f}")

    # 绘图部分
    if plot:
        t = records['t']
        plt.figure(figsize=(12, 8))

        # (1) 功率流：负载、光伏、电池功率、电网功率
        plt.subplot(3, 1, 1)
        plt.plot(t, records['load'], label='Load(kW)', c='c', linewidth=2)
        plt.plot(t, records['pv'], label='PV(kW)', c='g', linewidth=2)
        plt.plot(t, records['ug_price'], label='UG_Price(Cents/kW)', color='tab:orange', linewidth=2)
        plt.bar(t, records['battery_power'], label='Battery_Power(kW)', color='tab:blue', alpha=0.5)
        plt.ylabel("kW")
        plt.title("RM Scheduling Process")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)

        # (2) SOC变化
        plt.subplot(3, 1, 2)
        plt.plot(t, records['soc'], color='tab:green', linewidth=2, label='SOC')
        plt.ylabel("SOC")
        plt.title("The change of SOC")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)

        # (3) 电价与成本
        plt.subplot(3, 1, 3)
        plt.plot(t, records['cost'], label='Cost(Cents)', color='tab:red', linewidth=1.5, linestyle='--')
        plt.ylabel("Price")
        plt.title("The change of COST")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return total_cost, records

if __name__ == "__main__":
    is_train = 0     # 是否训练
    is_test = 1      # 是否测试
    if is_train:
        train_agent(env, control_agent, num_episodes=10000, minimal_size=2000, batch_size=64, is_plt = True)
    if is_test:
        env.is_train = False
        control_agent.is_train = False
        control_agent.actor.load_state_dict(torch.load('Model_Save/actor_weights.pth'))
        test_microgrid(env, control_agent, test_time=48, plot=True)
```
## 3. 展示训练结果

**训练过程**：
<div style="text-align: center;">
  <img 
    src="https://github.com/user-attachments/assets/02c86102-05cf-46cb-b78d-7fad55df1598" 
    alt="描述文本" 
    width="640" 
    height="480" 
    style="display: block; margin: 0 auto;"
  >
  <p style="text-align: center; font-style: italic;">图1：训练过程中的奖励变化</p>
</div>


**测试结果**：
<div style="text-align: center;">
  <img 
    src="https://github.com/user-attachments/assets/7f7afeb2-a665-4ce2-98cc-8c2da1379f0b" 
    alt="描述文本" 
    width="960" 
    height="640" 
    style="display: block; margin: 0 auto;"
  >
  <p style="text-align: center; font-style: italic;">图2：测试结果</p>
</div>
