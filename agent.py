from models.actor import DRL4SSP
from models.critic import StateCritic
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
import torch.nn as nn
from params import scale_reward


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, batch_size, capacity=1000):
        self.actors = [DRL4SSP(agent_id=i) for i in range(n_agents)]
        self.critics = [StateCritic() for i in range(n_agents)]

        # TODO 后续再引入target模式
        # self.actors_target = deepcopy(self.actors)
        # self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            # for x in self.actors_target:
            #     x.cuda()
            # for x in self.critics_target:
            #     x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(len(self.memory))
            batch = Experience(*zip(*transitions))
            # state_batch: batch_size x n_agents x dim_obs
            for i in range(len(batch.rewards)):
                # TODO 改成完全合作型，中心化crtic的训练
                state_static = batch.states[i][agent]["static"]
                state_dynamic = batch.states[i][agent]["dynamic"]
                tour_logp = batch.actions_logp[i][agent]
                critic_est = self.critics[agent](state_static, state_dynamic).view(-1)

                reward = batch.rewards[i]
                advantage = reward - critic_est

                actor_loss = th.mean(advantage * tour_logp.sum())
                critic_loss = th.mean(advantage ** 2)

                self.actor_optimizer[agent].zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer[agent].step()

                self.critic_optimizer[agent].zero_grad()
                critic_loss.backward()
                self.critic_optimizer[agent].step()

                c_loss.append(critic_loss)
                a_loss.append(actor_loss)

        # if self.steps_done % 100 == 0 and self.steps_done > 0:
        #     for i in range(self.n_agents):
        #         soft_update(self.critics_target[i], self.critics[i], self.tau)
        #         soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch, hiddens):
        actions = {}
        next_hiddens = {}
        logp_s = {}

        for agent_id in range(self.n_agents):
            act, next_hidden, log_p = self.actors[agent_id](state_batch[agent_id], hiddens[agent_id])

            actions[agent_id] = act
            next_hiddens[agent_id] = next_hidden
            logp_s[agent_id] = log_p

        self.steps_done += 1
        print(f"actions: {actions}\n logp_s: {logp_s}")
        return actions, next_hiddens, logp_s
