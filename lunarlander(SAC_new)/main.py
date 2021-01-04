from gym.envs.box2d.lunar_lander import LunarLanderContinuous
import numpy as np
from agent import Agent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = LunarLanderContinuous()
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_games = 5

    filename = 'lunar_lander.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = True

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        timestep = 0
        while not done and timestep < 3000:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)

            env.render()

            if not load_checkpoint:
                agent.learn()
            observation = observation_
            timestep += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

    env.close()