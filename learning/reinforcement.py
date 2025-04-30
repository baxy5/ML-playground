"""
Reinforcement Learning: The model learns by interacting with an environment and getting rewards or penalties.

Models:
- Q-Learning
- Deep Q Networks
"""

import gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))

alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration rate
episodes = 1000

# Training loop
for episode in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:
        # ε-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, done, truncated, info = env.step(action)

        # Q-value update (Bellman Equation)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

# Show learned Q-table
print("Trained Q-Table:")
print(q_table)

state = env.reset()[0]
done = False
total_reward = 0

print("\nRunning Trained Agent...\n")

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, truncated, info = env.step(action)
    env.render()
    total_reward += reward

print(f"\nTotal reward: {total_reward}")
