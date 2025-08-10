import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Categorical


class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.rewards = []
        self.probs = []
        self.position = 0

    def push(self, reward, prob):
        if len(self.rewards) < self.capacity:
            self.rewards.append(None)
            self.probs.append(None)

        self.rewards[self.position] = reward
        self.probs[self.position] = prob
        self.position = (self.position + 1) % self.capacity

    def clear(self):
        self.rewards.clear()
        self.probs.clear()
        self.position = 0

    def __len__(self):
        return len(self.rewards)


class PolicyGradient(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        checkpoint_dir="checkpoints/policy_gradient",
        device="cpu",
    ):
        super(PolicyGradient, self).__init__()
        self.checkpoint_file = checkpoint_dir
        self.device = device
        self.to(device)

        self.layer1 = nn.Linear(*input_dims, 128)
        self.layer2 = nn.Linear(128, output_dims)
        self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        # state tensor
        x = self.layer1(state)
        x = F.relu(self.dropout(x))
        x = self.layer2(x)
        x = self.softmax(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class PolicyGradientAgent:
    def __init__(self, num_actions, state_dims, gamma, lr):
        self.policy: PolicyGradient = PolicyGradient(state_dims, num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        self.memory = ExperienceReplay(10000)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save_models(self, silent=False):
        if not silent:
            print("Saving models...")
        self.policy.save_checkpoint()

    def load_models(self, silent=False):
        if not silent:
            print("Loading models...")
        self.policy.load_checkpoint()

    def choose_action(self, obs):
        state = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)

        probs = self.policy(state)
        c = Categorical(probs)
        action = c.sample()

        probs = c.log_prob(action)
        action = torch.squeeze(action).item()
        return action, probs

    # update policy
    def learn(self):
        R = 0
        rewards = []

        for r in self.memory.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (
            rewards.std() + np.finfo(np.float32).eps
        )

        # loss = - sum_t log(\pi(a|s)) * v_t
        probs = torch.stack(self.memory.probs)
        loss = torch.sum(
            torch.mul(probs, Variable(rewards).to(self.device)).mul(-1), -1
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
