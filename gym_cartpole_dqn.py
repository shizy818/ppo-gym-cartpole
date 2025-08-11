import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
import numpy as np
from torch.autograd import Variable

from dqn import DQNAgent
import torch
import os
from pathlib import Path


def env_step(env: Env, action):
    state, reward, done, _, _ = env.step(action)
    return torch.FloatTensor([state]), torch.FloatTensor([reward]), done


def train(
    env: Env,
    num_episodes=300,
    batch_size=128,
    lr=0.001,
    print_every=1,
    target_update=40,
    max_score=10000,
):

    # create agent (note this is done in train because it needs hyperparams)
    agent = DQNAgent(
        num_actions=env.action_space.n,
        state_dims=env.observation_space.shape,
        batch_size=batch_size,
        lr=lr,
    )

    # lists to log score and avg score hist during training
    score_hist = []
    avg_score_hist = []

    # data tracking
    num_steps = 0

    # init best score (used to see if model should be saved)
    best_score = -float("inf")

    # episodes is the number of games to play, where each game ends
    # when the agent meets terminal conditions for the env
    for episode in range(num_episodes):
        state = env.reset()[0]
        obs = Variable(torch.from_numpy(state).float().unsqueeze(0))
        done = False
        score = 0
        # c_loss = 0
        # c_samples = 0

        # run loop where model gathers data for "learn_every"
        # steps then learns using that information
        while not done:
            # choose action (actor)
            action = agent.choose_action(obs)
            # get results of action
            next_obs, reward, done = env_step(env, action.item())

            if done:
                next_obs = None

            # save data to memory for experience learning
            agent.memory.push(obs, action, next_obs, reward)
            obs = next_obs

            # learn using "batch_size" many memories
            agent.learn()
            if num_steps % target_update == 0:
                # agent.soft_update(tau=1e-3)
                agent.learner.save_checkpoint()
                agent.target.load_checkpoint()

            # update obs and tracking vars
            score += reward.detach().item()
            num_steps += 1
            # c_samples += batch_size
            # c_loss += loss

            if score > max_score:
                print(f"Agent was too powerful! Score exceeded {max_score}")
                agent.save_models()

                score_hist.append(max_score)
                avg_score = np.mean(score_hist[-100:])
                avg_score_hist.append(avg_score)

                return agent, score_hist, avg_score_hist, episode

        score_hist.append(score)

        # calc running avg score (use prev 100 so that
        # it's not biased by (a) lucky run(s))
        try:
            avg_score = np.mean(score_hist[-100:])
        except:
            avg_score = 0

        # track history
        avg_score_hist.append(avg_score)

        # save best model so far
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models(silent=True)

        if episode % print_every == 0 or episode == num_episodes - 1:
            print(
                f"game: [{episode+1}/{num_episodes}]\tscore:\t{score:.2f}\tavg_score: {avg_score:.2f}"
            )
            # print(
            #     "Loss", c_loss / c_samples, num_steps
            # )

    return agent, score_hist, avg_score_hist, num_episodes


def run_example(env: Env, agent: DQNAgent, max_score=1000):
    with torch.no_grad():
        state = env.reset()[0]
        obs = Variable(torch.from_numpy(state).float().unsqueeze(0))
        done = False
        score = 0

        while not done:
            # choose action (actor)
            action = agent.choose_action(obs)

            # get results of action
            next_state, reward, done, _, _ = env.step(action.item())

            # update obs and tracking vars
            obs = torch.FloatTensor([next_state])
            score += reward

            if score > max_score:
                print(f"Agent was too powerful! Score exceeded {max_score}")
                return


if __name__ == "__main__":

    SAVE_LOC = "recordings/gym-cartpole-v1/dqn"

    # OpenAI Gym Cartpole for Testing
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # record and run train loop
    train_env = RecordVideo(env, SAVE_LOC, name_prefix="dqn-cartpole-train")

    trained_agent, score_hist, avg_score_hist, num_episodes = train(train_env)
    train_env.close()

    # load best model
    print("load best model!!")
    trained_agent.load_models()

    # record and run test example
    test_env = RecordVideo(
        env, SAVE_LOC, name_prefix="dqn-cartpole-test", episode_trigger=lambda x: x == 0
    )
    run_example(test_env, trained_agent)
    test_env.close()

    vids = dict()
    test_file = None
    for file in os.listdir(SAVE_LOC):
        file = Path(file)
        if file.suffix == ".mp4":
            if "test" in file.stem:
                vids[num_episodes] = str(Path(SAVE_LOC) / file)
            else:
                episode_num = file.stem.split("-")[4]
                vids[int(episode_num)] = str(Path(SAVE_LOC) / file)
