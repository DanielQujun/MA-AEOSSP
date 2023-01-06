from env import SatelliteENV
from agent import MADDPG
import numpy as np
import torch

np.random.seed(1234)
torch.manual_seed(1234)
n_states = 213
n_actions = 2
capacity = 1000000

n_episode = 20000
max_steps = 1000
episodes_before_train = 100

win = None
param = None

STATIC_SIZE = 4
DYNAMIC_SIZE = 3

# TODO need fix action probs nan problems
batch_size = 1
instances_num = 50
agent_num = 2
config = {
    "batch_size": batch_size,
    "instances_num": instances_num,
    "agent_num": 2
}

env = SatelliteENV(config)
maddpg = MADDPG(agent_num, n_states, batch_size)

FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor
x0 = torch.zeros((1, STATIC_SIZE, 1), requires_grad=True)
total_reward = 0.0
for i_episode in range(n_episode):
    obs = env.reset()

    rr = 0.0
    decoder_inputs = {}
    for agent_id in range(agent_num):
        decoder_inputs[agent_id] = x0.expand(batch_size, -1, -1)

    for t in range(max_steps):
        with torch.autograd.set_detect_anomaly(True):
            # RNN 模型需要记录神经网络的上一个隐藏输出
            actions, next_hidden, actions_logp = maddpg.select_action(obs, decoder_inputs)
            obs_, reward, done, _ = env.step(actions)

            decoder_inputs = next_hidden

            if not done:
                next_obs = obs_
            else:
                next_obs = None

            total_reward += reward
            rr += reward
            maddpg.memory.push(obs, actions, actions_logp, next_obs, reward)
            obs = next_obs
            if done:
                # 开始训练这个episode
                c_loss, a_loss = maddpg.update_policy()
                maddpg.episode_done += 1
                print(f'Episode: {i_episode}, reward = {total_reward}, c_loss = {c_loss}, a_loss = {a_loss}')
                break

