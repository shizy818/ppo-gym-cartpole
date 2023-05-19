# PPO for OpenAI Gym Cartpole

## Link

WandB: https://wandb.ai/arth-shukla/PPO%20Gym%20Cart%20Pole

## Papers Used

Proximal Policy Optimization Algorithms: https://arxiv.org/pdf/1707.06347.pdf

## Technologies Used

Algorithms/Concepts: PPO, Experience Replay

AI Development: Pytorch (Torch, Cuda), OpenAI Gym, WandB

## Evaluation and Inference

More Figures available on WandB: https://wandb.ai/arth-shukla/PPO%20Gym%20Cart%20Pole

The PPO Model currently only supports discrete action spaces (categorical distribution). In OpenAI Gym Cartpole, by episode 136, the agent is able to effectively "beat" cartpole:

<figure>
<figcaption><h3>Episode 136</h3></figcaption>
<video style="width:100%" controls>
  <source src="./videos/gym_carpole_ppo_ep_136.mp4" type="video/mp4">
</video>
</figure>

## Future Experiments

First I want to implement algorithms that came before PPO (DQNs or earlier actor-critic algorithms like DDPG, etc) to get a stronger understanding of the math. Also, I'll get a change to make agents for popular environemnts like Mario.

I also want to tackle more challenging game environments, like the DM Control Suite. To do this, I'll explore PPO for continuous action spaces (through normal distributions), other similarly effective models like SAN, and models like RecurrentPPO which offer some implemenation challenges.

Finally, there are some other options for experience replay I'd like to implement, like Prioritized ER.

## About Me

Arth Shukla [Site](https://arth.website) | [GitHub](https://github.com/arth-shukla) | [LinkedIn](https://www.linkedin.com/in/arth-shukla/)
