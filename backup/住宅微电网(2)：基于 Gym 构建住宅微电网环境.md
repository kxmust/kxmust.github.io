上一篇内容详细介绍了住宅微电网能源管理的优化目标，所有组件和相关约束。这期内容将基于 Gym 环境来构建一个住宅微电网模型用来模拟住宅微电网的运行。后续会基于深度强化学习算法构建微电网能源管理系统来实现住宅微电网的智能化调度。
## 1. 数据处理
首先从"home_1_training_30days.csv "文件中读取负载数据和光伏数据，从"electricity_price_training.csv"文件中读取电价数据。
```python
# 导入数据
load_data = pd.read_csv(r'data/home_1_training_30days.csv',
                        header=0).iloc[:, 1]
pv_data = pd.read_csv(r'data/home_1_training_30days.csv',
                      header=0).iloc[:, 2]
price_data = pd.read_csv(r'data/electricity_price_training.csv',
                         header=0).iloc[:, 1]
```

构建一个数据类方法，可以通过pv(t)，来读取t时刻的光伏数据；通过load(t)，来读取t时刻的负载数据；通过ug_price(t)，来读取t时刻的电价数据；plot_figure(data_type, hours)来显示数据，比如输入（'pv', 24）表示画出24小时的PV数据。
```python
class RM_Data:
    def __init__(self, load_data, pv_data, price_data):
        self.pv_data = pv_data
        self.load_data = load_data
        self.price_data = price_data
        self.len_data = len(load_data)

    def pv(self, t):
        return self.pv_data[t]

    def ug_price(self, t):
        return self.price_data[t]

    def load(self, t):
        return self.load_data[t]

    def plot_figure(self, data_type, hours, save_dir='plots'):
        """
        绘制时间序列数据图表

        参数:
        data_type: 数据类型，可选 'pv'、'load'、'price'
        hours: 时间长度（小时）
        save_dir: 图片保存目录
        """

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 生成时间序列（假设从0点开始）
        time_points = list(range(hours))

        # 根据数据类型读取数据
        data = []
        for t in time_points:
            if data_type.lower() == 'pv':
                data.append(self.pv(t))  # 调用pv函数
            elif data_type.lower() == 'load':
                data.append(self.load(t))  # 调用load函数
            elif data_type.lower() == 'price':
                data.append(self.ug_price(t))  # 调用price函数
            else:
                raise ValueError(f"不支持的数据类型: {data_type}，请使用 'pv', 'load' 或 'price'")

        data = np.array(data)

        # 创建时间标签（从当前时间开始）
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        time_stamps = [base_time + timedelta(hours=t) for t in time_points]

        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))

        # 根据数据类型设置图表属性
        if data_type.lower() == 'pv':
            title = f'PV Power (0-{hours}Hours)'
            ylabel = 'Power (kW)'
            color = '#FFA500'  # 橙色
            line_style = '-o'
            alpha = 0.7
        elif data_type.lower() == 'load':
            title = f'Load Power (0-{hours}Hours)'
            ylabel = 'Power (kW)'
            color = '#1E90FF'  # 道奇蓝
            line_style = '-s'
            alpha = 0.7
        elif data_type.lower() == 'price':
            title = f'Electricity Price (0-{hours}Hours)'
            ylabel = 'Price (yuan/kWh)'
            color = '#32CD32'  # 石灰绿
            line_style = '-^'
            alpha = 0.7

        # 绘制线图
        ax.plot(time_stamps, data, line_style, color=color,
                linewidth=2, markersize=6, alpha=alpha,
                label=data_type.upper())

        # 填充区域（针对pv和load）
        if data_type.lower() in ['pv', 'load']:
            ax.fill_between(time_stamps, 0, data, alpha=0.2, color=color)

        # 设置图表属性
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # 设置x轴时间格式
        if hours <= 24:
            # 每小时显示一个标签
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, hours // 12)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            # 超过24小时，按天显示
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

        # 旋转x轴标签
        plt.xticks(rotation=45)

        # 添加网格
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.5)

        # 添加图例
        ax.legend(loc='best', fontsize=10)

        # 添加统计信息文本框
        stats_text = f"""
        Max Value: {data.max():.2f}
        Min Value: {data.min():.2f}
        Avg Value: {data.mean():.2f}
        Total Times: {hours}Hours
        """

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 自动调整布局
        plt.tight_layout()

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_{hours}h_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)

        # 保存图片
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"The figure is saved: {filepath}")

        # 显示图表
        plt.show()

        return filepath
```

打印出负载、光伏和电价数据。
```python
rm_data = RM_Data(load_data, pv_data, price_data)
rm_data.plot_figure('price', hours=24)
rm_data.plot_figure('load', hours=24)
rm_data.plot_figure('pv', hours=24)
```

<div style="text-align: center;">
  <img 
    src="https://github.com/user-attachments/assets/102ea669-3de2-4723-849d-63e36599af29" 
    alt="描述文本" 
    width="700" 
    height="500" 
    style="display: block; margin: 0 auto;"
  >
  <p style="text-align: center; font-style: italic;">图 1：电费数据</p>
</div>

<div style="text-align: center;">
  <img 
    src="https://github.com/user-attachments/assets/e4e22d9c-097d-4e7f-b20e-02d135aedbf0" 
    alt="描述文本" 
    width="700" 
    height="500" 
    style="display: block; margin: 0 auto;"
  >
  <p style="text-align: center; font-style: italic;">图 2：负载数据</p>
</div>

<div style="text-align: center;">
  <img 
    src="https://github.com/user-attachments/assets/365ea8fd-2e0d-40cf-ba17-7b13d6d12872" 
    alt="描述文本" 
    width="700" 
    height="500" 
    style="display: block; margin: 0 auto;"
  >
  <p style="text-align: center; font-style: italic;">图 3：光伏数据</p>
</div>

## 2. 基于Gym构建住宅微电网模型
基于上一篇描述的组件和相关约束，以及优化目标，基于Gym构建微电网模型。
```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MicrogridEnv(gym.Env):

    def __init__(self, rm_data, episode_length=24, is_train=False):
        super(MicrogridEnv, self).__init__()

        # 环境参数
        self.episode_length = episode_length
        self.data_length = rm_data.len_data

        # 微电网参数
        self.max_pv = 4.0
        self.max_load = 5.0
        self.max_ug_price = 5.2

        # 将电池的充放电功率设置为相同
        self.battery_charge_power = - 5.0
        self.battery_discharge_power = 5.0
        self.battery_capacity = 50.0    #(kWh)
        self.battery_eff = 1.0    # 电池转换效率
        self.initial_soc = 0.5

        # 奖励系数(优化目标)
        self.m1 = 0.60
        self.m2 = 0.40

        # 数据
        self.rm_data = rm_data

        # 定义状态空间: [负载, 光伏, 电价, SOC]
        # 状态空间取值的上限
        self.obs_high = np.array([self.max_load, self.max_pv,
                             self.max_ug_price, 1.0], dtype=np.float32)
        # 状态空间取值下限
        self.obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 定义强化学习的观察空间，系统会自动进行归一化处理
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        # 定义动作空间: 电池充放电功率
        self.action_space = spaces.Box(low=np.array([self.battery_charge_power], dtype=np.float32),
                                       high=np.array([self.battery_discharge_power], dtype=np.float32),
                                       dtype=np.float32)

        # 初始化状态
        self.len_t = 0      # 已经模拟的时长
        self.state = None
        self.done = False
        self.episode_start_idx = 0   # 开始训练时，随机抽取一个时刻作为初始时刻

        # 动作
        self.action_bound = self.action_space.high.item()
        self.is_train = is_train

        # 共有 720个数据，后两天用于测试
        self.test_start = 672

    # 初始化环境
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 随机选择起始点（增强数据多样性）
        if self.is_train:
            max_start = self.test_start - self.episode_length -1
            self.episode_start_idx = self.np_random.integers(0, max_start) \
                if max_start > 0 else 0
        else:
            self.episode_start_idx = self.test_start    # 测试时使用后两天的数据

        self.len_t = 0
        self.done = False

        load = self.rm_data.load(self.episode_start_idx)
        pv = self.rm_data.pv(self.episode_start_idx)
        ug_price = self.rm_data.ug_price(self.episode_start_idx)
        if self.is_train:
            soc = np.random.uniform(0.3, 0.7)
        else:
            soc = self.initial_soc

        self.state = np.array([load, pv, ug_price, soc], dtype=np.float32)
        return self.state, {}

    # 对输入数据归一化
    def _normalize(self, state):
        return (state - self.obs_low) / (self.obs_high - self.obs_low + 1e-8)

    def _clip_battery_action(self, battery_power, current_soc):
        """根据当前 SOC 裁剪电池动作，确保物理可行性"""
        # 计算 SOC 约束下的功率限制
        max_charge = (current_soc - 1.0) * self.battery_capacity / self.battery_eff
        max_discharge = current_soc * self.battery_capacity / self.battery_eff

        clipped_power = np.clip(
            battery_power,
            max(self.battery_charge_power, max_charge),  # 充电（负功率）
            min(self.battery_discharge_power, max_discharge)  # 放电（正功率）
        )
        return clipped_power

    # 环境步进，基于动作，给出下一时刻的状态
    def step(self, action):
        # 当前状态
        load, pv, ug_price, soc = self.state

        # 当前动作
        battery_power = self._clip_battery_action(action, soc)

        # 奖励
        reward = self.reward_function(self.state, action)

        # 计算下一时刻状态
        self.len_t += 1
        self.episode_start_idx += 1

        if self.is_train:
            self.done = self.len_t >= self.episode_length

        if not self.done:
            next_load = self.rm_data.load(self.episode_start_idx)
            next_pv = self.rm_data.pv(self.episode_start_idx)
            next_ug_price = self.rm_data.ug_price(self.episode_start_idx)
            next_soc = soc - battery_power * self.battery_eff / self.battery_capacity

            # SOC 安全检查（理论上不应该发生）
            if next_soc < 0. - 1e-7 or next_soc > 1. + 1e-7:
                reward -= 5.0  # 大惩罚
                next_soc = np.clip(next_soc, 0., 1.)
                self.done = True

            next_state = np.array([next_load, next_pv, next_ug_price, next_soc], dtype=np.float32)

            self.state = next_state

        return self.state, reward, self.done, False, {}

    # 奖励函数
    def reward_function(self, state, action):
        load, pv, ug_price, soc = state
        battery_power = self._clip_battery_action(action, soc)


        # 电网交互功率（正：买电；负：卖电）
        Pe = load - pv - battery_power * self.battery_eff
        if Pe >= 0:
            cost = Pe * ug_price
        else:
            cost = Pe * ug_price * 0.7  # 卖电收益较低

        profit_norm =  -cost/ (self.max_load * self.max_ug_price)

        # SOC平衡惩罚
        next_soc = soc - battery_power * self.battery_eff / self.battery_capacity
        soc_penalty = (next_soc - 0.5)**2

        reward = self.m1 * profit_norm - self.m2 * soc_penalty

        return float(reward)

```

该微电网环境是基于上一篇微电网模型进行构建，通过导入微电网数据 rm_data来构建模型，方便与后续构建的控制系统进行交互，该环境主要包含的方法如下：
1. **reset()** ：获取初始状态，当设置为训练模式时，会从测试数据中随机抽取一个时刻作为初始状态，在设置为测试模式时，使用最后两天的数据进行测试。
2. **_normalize()**：对状态进行归一化处理。
3. **_clip_battery_action()**：对控制系统输出的动作进行处理，防止储能系统超过约束限制。
4. **step()**：微电网环境执行控制系统给出的调度指令，生成下一时刻的状态、奖励等。
5. **reward_function()**：基于优化目标构建的奖励函数。

