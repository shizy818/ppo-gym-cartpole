import math
import random
from collections import namedtuple

import torch
from torch import nn, optim
import os

# 参考
# https://github.com/SeeknnDestroy/DQN-CartPole/blob/master/dql-cartpole.ipynb
# https://github.com/jordanlei/deep-reinforcement-learning-cartpole/blob/master/dqn_cartpole.pyq
# https://zhuanlan.zhihu.com/p/466455380

# experience replay
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # if memory isn't full, add a new experience
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)


# deep Q network implementation
class DQN(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        checkpoint_dir="checkpoints/dqn",
        device="cpu",
    ):
        super(DQN, self).__init__()

        # send to device (easier to handle at top level)
        self.device = device
        self.to(device)

        # save checkpoints in case smth happens or I wanna try again with a baseline
        self.checkpoint_file = os.path.join(checkpoint_dir, "dnq")

        # ------------------------------------------------------
        # define layers

        self.layer1 = nn.Linear(input_dims, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dims)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        # x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
        x = nn.functional.relu(self.layer1(x))
        x = self.dropout(nn.functional.relu(self.layer2(x)))
        x = nn.functional.relu(self.layer3(x))
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class DQNAgent:
    # defaults to 'ideal' values from https://arxiv.org/pdf/1707.06347.pdf
    def __init__(
        self,
        num_actions,
        input_dims,
        gamma=0.99,
        eps_start=0.9,
        eps_end=0.1,
        eps_decay=200,
        lr=0.003,
        batch_size=64,
        mem_max=10000,
    ):
        # hyperparams
        self.gamma = gamma
        self.batch_size = batch_size

        self.num_actions = num_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.memory = ExperienceReplay(mem_max)
        self.learner = DQN(input_dims, num_actions)
        self.target = DQN(input_dims, num_actions)
        self.optimizer = optim.Adam(self.learner.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

    def save_models(self, silent=False):
        if not silent:
            print("Saving models...")
        self.learner.save_checkpoint()
        self.target.save_checkpoint()

    def load_models(self, silent=False):
        if not silent:
            print("Loading models...")
        self.learner.load_checkpoint()
        self.target.load_checkpoint()

    # epsilon-greedy策略: 在强化学习中，智能体在选动作时以epsilon概率随机探索，以1-epsilon概率选择当前 Q 网络最优动作。
    # 早期（训练刚开始）要多探索新环境，后期更信赖学到的策略
    def choose_action(self, obs):
        # select an action based on state
        self.steps = self.steps + 1
        sample = random.random()

        # get a decayed epsilon threshold
        eps_thresh = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1 * self.steps / self.eps_decay
        )
        if sample > eps_thresh:
            with torch.no_grad():
                # select the optimal action based on the maximum expected return
                action = torch.argmax(self.learner(obs)).item()
                return action
        else:
            return random.randrange(self.num_actions)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0

        sample_transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*sample_transitions))

        # get a list that is True where the next state is not "done"
        has_next_state = torch.tensor(
            list(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_expected = self.learner(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        # get the max along each row using the target network, then detach
        next_state_values[has_next_state] = self.target(next_states).detach().max(1)[0]

        # Q(s, a) = reward(s, a) + Q(s_t+1, a_t+1) * gamma
        q_target = next_state_values.unsqueeze(1) * self.gamma + reward_batch

        loss = self.loss(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.learner.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.soft_update(1e-3)

        return loss

    def soft_update(self, tau):
        for target_param, learner_param in zip(
            self.target.parameters(), self.learner.parameters()
        ):
            target_param.data.copy_(
                tau * learner_param.data + (1.0 - tau) * target_param.data
            )
