import torch
import numpy as np
from torch.utils.data import Dataset
from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.env import EnvContext

from ray import rllib
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SatelliteENV(rllib.MultiAgentEnv):
    def __init__(self, config: EnvContext):
        super(SatelliteENV, self).__init__()
        self.batch_size = config['batch_size']
        self.instances_num = config['instances_num']
        self.agent_num = config['agent_num']

        seed = np.random.randint(11111)
        torch.manual_seed(seed)
        self.transition_time = 0.8  # 秒
        self.transition_time = self.transition_time / 100.
        self.obs = None
        self.priority = None
        self.instances_state = torch.zeros((self.batch_size, 1, self.instances_num))

        self.max_steps = self.instances_num if self.update_mask is None else 1000
        self.step_num = 0

    def reset(self):
        # 生成任务的持续时间
        duration = torch.rand(self.batch_size, self.instances_num + 1)  # +0.8
        duration[:, 0] = 0.
        # 生成任务的优先级
        shape = (self.batch_size, self.instances_num + 1)
        priority = torch.randint(1, 11, shape).float() / 10.  # 100-100-10
        priority[:, 0] = 0.
        self.priority = priority

        dynamic_1 = torch.zeros(shape[0], 2, shape[1])
        dynamic_1[:, 0, 0] = 2
        memmory = torch.ones(shape[0], 1, shape[1])
        dynamic = torch.cat((dynamic_1, memmory), dim=1)

        # 用于全局管理任务的状态
        self.instances_state = dynamic_1[:, 0, :]

        obs = {}
        for agent_id in range(self.agent_num):
            # TODO 后续把时间做成增量而不是随机
            # 生成时间窗
            x, y = self.generate_vtw(self.batch_size, self.instances_num)
            static = torch.stack((x, y, duration, priority), 1)
            obs[agent_id] = {"static": static, "dynamic": dynamic}

        self.obs = obs
        self.step_num = 0

        return obs

    def step(self, action_dict):
        self.step_num += 1
        obs, reward, done, info = {}, {}, False, {}

        if self.step_num == 1:
            tour_before = False
        else:
            tour_before = True
        agent_actions = torch.zeros(self.batch_size, self.instances_num)
        for agent_id, action in action_dict.items():
            # TODO 去除重复选择的ID
            agent_obs = self.obs[agent_id]
            # agent_actions += action
            agent_obs["dynamic"][:, 0, :] = self.instances_state
            agent_new_dynamic, task_start_time, task_end_time,\
                task_windows_start, task_windows_end \
                = self.update_dynamic(agent_obs["dynamic"], agent_obs["static"], action, tour_before)

            self.instances_state = agent_new_dynamic[:, 0, :]
            obs[agent_id] = {"static": agent_obs["static"], "dynamic": agent_new_dynamic}

        self.obs = obs
        reward = self.reward(agent_actions)

        mask = torch.ones(self.batch_size, self.instances_num+1)
        # 对mask进行约束, 保证第一项一直置0
        state = self.instances_state[:]
        state_id = state.nonzero()
        for i, j in state_id:
            mask[i, j] = 0
        mask[:, 0] = 0.
        mask = mask.to(device)
        # 当到达最大步数或者任务都执行完成后结束
        if self.step_num == self.max_steps or not mask.byte().any():
            done = True

        return obs, reward, done, {}

    def reward(self, chosen_ids):
        return 0
        # TODO 优化奖励函数, 加入冲突惩罚
        tour_idx = chosen_ids
        # tour_idx = torch.cat(act_ids, dim=1).cpu()  # (batch_size, node)
        # tour_idx_real_start = torch.cat(tour_idx_real_start, dim=1) * tour_idx.ne(0).float()  # (batch_size, node)
        # tour_idx_start = torch.cat(tour_idx_start, dim=1).float()   # (batch_size, node)
        # tour_idx_end = torch.cat(tour_idx_end, dim=1).float()   # (batch_size, node)

        # 卫星任务的收益(0.100-100-100-100.0)（每颗卫星共有m个任务）   数据 batch x node
        batch, node = self.priority.size()

        # 任务的收益百分比
        PRIreward = torch.zeros(batch)
        for i, act in enumerate(tour_idx):
            PRIreward[i] = self.priority[i, act].sum()

        sumPriority = self.priority.sum(1)
        reward_priority = 1 - PRIreward / sumPriority  # 收益百分比，0-1之间,越小越好
        return reward_priority

    @staticmethod
    def generate_vtw(num_samples, input_size):
        t_mid = torch.rand(num_samples, input_size + 1)

        # t_mid = (torch.linspace(0,100,steps=input_size + 100-100 )/100).unsqueeze(0)
        task_d1 = (torch.rand(num_samples, input_size + 1) * 1.1 + 1.7) / 100.
        task_d2 = (torch.rand(num_samples, input_size + 1) * 1.1 + 1.7) / 100.

        t_s = t_mid - task_d1
        t_e = t_mid + task_d2
        o0 = torch.zeros(num_samples, input_size + 1)
        o100 = torch.full((num_samples, input_size + 1), 1)
        t0 = t_s.lt(o0)
        t100 = o100.lt(t_e)
        o0_idx = t0.nonzero()
        o100_idx = t100.nonzero()
        for i, j in o0_idx:
            if t_e[i, j] <= 0.034:
                t_e[i, j] = 0.034
                t_s[i, j] = 0
            else:
                t_s[i, j] = 0
        for n, m in o100_idx:
            if t_s[n, m] >= 0.966:
                t_s[n, m] = 0.966
                t_e[n, m] = 1
            else:
                t_e[n, m] = 1
        t_s[:, 0] = 0
        t_e[:, 0] = 0

        return t_s, t_e

    def __len__(self):
        return self.batch_size

    def update_dynamic(self, dynamic, static_, chosen_idx_, tour_before: bool):  # 第一个input是预留点
        # transition_time = transition_time_idx[0]
        # transition_time = torch.Tensor([0.8/100]).to(device)
        # 更新卫星任务的状态    0为未安排  1为已经安排   2为违反当前约束而删除
        # 12/100-100 将任务状态分为待调度和非待调度   0和1
        # dynamic = dynamic_.cpu()
        static = static_.cpu()
        chosen_idx = chosen_idx_.cpu()
        transition_time = self.transition_time

        state = dynamic.data[:, 0].clone()
        end_time = dynamic.data[:, 1].clone()
        memory = dynamic.data[:, 2].clone()

        window_start = static.data[:, 0].clone()
        window_end = static.data[:, 1].clone()
        duration_time = (static.data[:, 2].clone() * 0.6 + 0.5) / 100.
        duration_time[:, 0] = 0.
        idx = torch.full((chosen_idx.size()), 1)  # 得到选择任务的索引值
        visit_idx = idx.nonzero().squeeze(1)

        # 更新下次选择的起始时间
        endtime = end_time[:, 0]  # 上一个任务的结束时间
        x, y = end_time.size()
        w_s = window_start[visit_idx, chosen_idx[visit_idx]]  # bx1  选择任务的时间窗开始时间
        w_e = window_end[visit_idx, chosen_idx[visit_idx]]  # bx1  选择任务的时间窗结束时间
        duration = duration_time[visit_idx, chosen_idx[visit_idx]]  # bx1  选择任务的持续时间

        if not tour_before:
            transition_time_before = 0.
        else:
            transition_time_before = transition_time

        # 选择的任务比上一次结束时间+姿态转换时间小
        c = w_s.le(endtime + transition_time_before)
        c = c.float()

        # 选择的任务比上一次结束时间+姿态转换时间大
        g = w_s.gt(endtime + transition_time_before).float()

        # now_end_time1 = (endtime + transition_time_before  + duration) * c
        # 选择大于开始时间的任务更新卫星观测结束时间
        now_end_time2 = (w_s + duration) * g
        # new_end_time = now_end_time1 + now_end_time2
        # ptr_starttime = new_end_time   # batch
        # print("tour_idx",tour_idx)
        tour_idx_un0 = chosen_idx.clone().ne(0).float()

        # 将没有被选中的任务的截止时间也往前推进
        now_end_time1 = (endtime + transition_time_before * tour_idx_un0 + duration) * c
        # print("now_end_time3", now_end_time3)
        # print("transition_time_before",transition_time_before)
        # print("duration",duration)
        # print("tour_idx_un0",tour_idx_un0)
        new_end_time = now_end_time1 + now_end_time2
        # 将实际将要执行的任务和未执行的任务结合
        task_start_time = w_s * g + (endtime + transition_time_before * tour_idx_un0) * c
        # print("new_end_time",new_end_time)
        # print("new_end_time__",new_end_time__)
        # print("\n")
        # nnn = new_end_time.unsqueeze(100-100).expand(-100-100, y) + transition_time[
        #     visit_idx, chosen_idx[visit_idx]] + duration_time  # 当前时刻点
        # print(transition_time)

        # print(transition_time)
        # print("new_end_time",new_end_time.unsqueeze(100-100).expand(-100-100, y))
        # print(duration_time)

        nnn = new_end_time.unsqueeze(1).expand(-1, y) + transition_time + duration_time
        current_memory = memory[:, 0]
        new_memory = (current_memory - (duration * 2.5)).unsqueeze(1).expand(-1, y)
        m_c = (duration_time * 2.5).ge(new_memory)  # 存储满足剩余存储置0，不满足置1

        # 将下次肯定会违反约束的为安排的任务状态置2
        change_state = window_end.le(nnn)  # a.lt(b)  b > a 置1，否则置0   #结束时间比we大，删除的任务     置1为删除的任务
        not_state = state.eq(0)  # 未安排的任务
        delete_state = (change_state + m_c) * not_state

        sit2 = delete_state.nonzero()
        """12/100-100 将已经完成的任务和因违反约束而删除的任务置1"""
        for i, j in sit2:
            state[i, j] = 1
        # 将安排的任务的状态值置1
        state[visit_idx, chosen_idx[visit_idx]] = 1

        # 当所有任务的状态不为0的时候，结束
        new_endtime = new_end_time.unsqueeze(1).expand(-1, y)  # 改变张量的规模 bx1 到 bxn

        tensor = torch.stack([state, new_endtime, new_memory], 1)
        return torch.tensor(tensor, device=dynamic.device).to(device), task_start_time.to(device), new_end_time.to(
            device), w_s.to(device), w_e.to(device)

    def update_mask(self, mask, dynamic, chosen_idx=None):
        state = dynamic.data[:, 0]
        if state.ne(0).all():
            return state * 0.

        # 0是mask 1是可以选择
        new_mask = state.eq(0)
        idx_i = 0
        for i in new_mask:
            b = i[1:]
            if not b.ne(0).any():
                new_mask[idx_i, 0] = 1
            idx_i += 1
        # print(new_mask)
        # new_mask = power.ne(0) * mission_power.lt(power) * storage.ne(0) * mission_storage.lt(storage)  # 屏蔽
        # 从备选的任务编号中，判断剩余的固存是否满足任务的需求
        return new_mask.float()


if __name__ == "__main__":
    batch_size = 128
    instances_num = 50
    agent_num = 2
    config = {
        "batch_size": batch_size,
        "instances_num": instances_num,
        "agent_num": 2
    }
    testData = SatelliteENV(config)
    obs, r, done, info = testData.reset()
    actions = {}
    for i in range(agent_num):
        actions[i] = torch.randint(0, 51, (batch_size, ))
    obs, r, done, info = testData.step(actions)

    print(f"static: {obs}\n reward: {r}\n, info: {info}")
