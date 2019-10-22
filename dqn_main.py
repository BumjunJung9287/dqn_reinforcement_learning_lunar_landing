import gym
from simple_dqn_torch import Agent
import numpy as np
#from gym import wrappers

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                    input_dims=[8], lr=0.003)
    scores = []
    eps_history = []
    n_games = 300
    score = 0

    for i in range(n_games):
        if i % 10 == 0 and i>0:
            avg_score = np.mean(scores[max(0,i-10):(i+1)])
            print("episode", i, "score", score,
                    "average score %.3f" % avg_score,
                    "epsilon %.3f" % brain.epsilon)
        else:
            print("episode ", i, "score", score)
        score = 0
        eps_history.append(brain.epsilon)
        observation = env.reset()
        done = False
        while not done:
            if i%20 == 0:
                env.render()
            action = brain.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            brain.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            brain.learn()
        scores.append(score)
