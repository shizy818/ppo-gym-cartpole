from dm_control import suite
import numpy as np
from ppo import PPOAgent
from tqdm import tqdm
import torch
import os
from pathlib import Path
import cv2


def train(
    env,
    batch_size=5,
    num_epochs=4,
    lr=0.0003,
    num_games=300,
    learn_every=20,
    print_every=20,
    max_score=10000,
):

    # get action and observation spec
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()

    sample_arrs = [np.zeros(el.shape) for el in observation_spec.values()]
    obs_shape = np.hstack(sample_arrs).shape[0]

    # return

    # create agent (note this is done in train because it needs hyperparams)
    agent = PPOAgent(
        num_actions=action_spec.shape[0],
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        input_dims=[obs_shape],
    )

    # lists to log score and avg score hist during training
    score_hist = []
    avg_score_hist = []

    # list to store video
    video = []
    size = 480

    # data tracking
    avg_score = 0
    num_steps = 0

    # init best score (used to see if model should be saved)
    best_score = 0

    # num_games is the number of games to play, where each game ends
    # when the agent meets terminal conditions for the env
    for i in range(num_games):

        time_step = env.reset()
        obs = np.hstack(time_step.observation.values())

        score = 0

        # run loop where model gathers data for "learn_every"
        # steps then learns using that information
        while not time_step.last():

            # choose action (actor)
            action, prob, val = agent.choose_action(obs)

            # get results of action
            time_step = env.step(action)

            # save data to memory for experience learning
            agent.remember(obs, action, prob, val, time_step.reward, time_step.last())

            # learn using "learn_every" many memories
            if num_steps % learn_every == 0:
                agent.learn()
                agent.clear_memory()

            video.append(
                np.hstack(
                    [
                        env.physics.render(size, size, camera_id=0),
                        env.physics.render(size, size, camera_id=1),
                    ]
                )
            )

            # update obs and tracking vars
            obs = np.hstack(time_step.observation.values())
            score += time_step.reward
            num_steps += 1

            if score > max_score:
                print(f"Agent was too powerful! Score exceeded {max_score}")
                agent.save_models()

                score_hist.append(max_score)
                avg_score = np.mean(score_hist[-100:])
                avg_score_hist.append(avg_score)

                return agent, score_hist, avg_score_hist, video

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

        if i % print_every == 0 or i == num_games - 1:
            print(
                f"game: [{i+1}/{num_games}]\tscore:\t{score:.2f}\tavg_score: {avg_score:.2f}"
            )

    return agent, score_hist, avg_score_hist, video


def run_example(env, agent: PPOAgent, max_score=1000):
    with torch.no_grad():
        obs = env.reset()[0]
        done = False
        score = 0

        # run loop where model gathers data for "learn_every"
        # steps then learns using that information
        while not done:

            # choose action (actor)
            action, prob, val = agent.choose_action(obs)

            # get results of action
            next_obs, reward, done, _, _ = env.step(action)

            # save data to memory for experience learning
            agent.remember(obs, action, prob, val, reward, done)

            # update obs and tracking vars
            obs = next_obs
            score += reward

            if score > max_score:
                print(f"Agent was too powerful! Score exceeded {max_score}")
                return


def numpy_to_vid(fn, video, size):
    import cv2

    size = video[0].shape[0:2]
    out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"mp4v"), 10, (size[1], size[0]))
    for frame in video:
        out.write(frame)
    out.release()


if __name__ == "__main__":

    SAVE_LOC = "recordings/dm-cartpole-v1"

    # DeepMind Control cartpole
    env = suite.load(domain_name="cartpole", task_name="balance")
    # Iterate over a task set:
    for domain_name, task_name in suite.BENCHMARKING:
        env = suite.load(domain_name, task_name)

    # run train loop
    trained_agent, score_hist, avg_score_hist, video = train(
        env, print_every=1, num_games=1
    )

    numpy_to_vid("output.mp4", video, (480, 960))

    # # # load best model
    # # trained_agent.load_models()

    # # # record and run test example
    # # run_example(env, trained_agent)

    # # vids = dict()
    # # test_file = None
    # # for file in os.listdir(SAVE_LOC):
    # #     file = Path(file)
    # #     if file.suffix == '.mp4':
    # #         if 'test' in file.stem:
    # #             vids[num_episodes] = str(Path(SAVE_LOC) / file)
    # #         else:
    # #             episode_num = file.stem.split('-')[4]
    # #             vids[int(episode_num)] = str(Path(SAVE_LOC) / file)

    # # import wandb

    # # wandb.login()

    # # run = wandb.init(project='PPO Gym Cart Pole')

    # # for score, avg_score in zip(score_hist, avg_score_hist):
    # #     wandb.log({ 'Score': score, 'Average Score (Past 100 Episodes)': avg_score })
    # # for key in sorted(vids.keys()):
    # #     wandb.log({f'PPO {key} Episodes': wandb.Video(vids[key], fps=24, format='mp4')})

    # # wandb.finish()
