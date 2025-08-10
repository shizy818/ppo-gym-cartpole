import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from pg import PolicyGradientAgent
import numpy as np
import torch
import os
from pathlib import Path


def train(
    env: Env,
    print_every=1,
    num_episodes=300,
    lr=0.01,
    gamma=0.99,
    max_score=10000,
):
    agent = PolicyGradientAgent(
        num_actions=env.action_space.n,
        state_dims=env.observation_space.shape,
        lr=lr,
        gamma=gamma,
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
        obs = env.reset()[0]
        done = False
        score = 0
        # c_loss = 0

        while not done:
            # choose action
            action, prob = agent.choose_action(obs)

            # get results of action
            next_obs, reward, done, _, _ = env.step(action)
            agent.memory.push(reward, prob)

            # update obs and tracking vars
            obs = next_obs
            score += reward
            num_steps += 1

        loss = agent.learn()
        # c_loss += loss
        agent.memory.clear()

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
            # print("Loss", c_loss, num_steps)

    return agent, score_hist, avg_score_hist, num_episodes


def run_example(env: Env, agent: PolicyGradientAgent, max_score=1000):
    with torch.no_grad():
        obs = env.reset()[0]
        done = False
        score = 0

        while not done:

            # choose action (actor)
            action, prob = agent.choose_action(obs)

            # get results of action
            next_obs, reward, done, _, _ = env.step(action)

            # update obs and tracking vars
            obs = next_obs
            score += reward

            if score > max_score:
                print(f"Agent was too powerful! Score exceeded {max_score}")
                return


if __name__ == "__main__":

    SAVE_LOC = "recordings/gym-cartpole-v1/pg"

    # OpenAI Gym Cartpole for Testing
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # record and run train loop
    train_env = RecordVideo(env, SAVE_LOC, name_prefix="policy-gradient-cartpole-train")

    trained_agent, score_hist, avg_score_hist, num_episodes = train(train_env)
    train_env.close()

    # load best model
    print("load best model!!")
    trained_agent.load_models()

    # record and run test example
    test_env = RecordVideo(
        env,
        SAVE_LOC,
        name_prefix="policy-gradient-cartpole-test",
        episode_trigger=lambda x: x == 0,
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
                episode_num = file.stem.split("-")[5]
                vids[int(episode_num)] = str(Path(SAVE_LOC) / file)
