import numpy as np
import random
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, alpha=0.05, gamma=1,
                 epsilon_init=1, epsilon_decay=0.99):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        print(f'alpha={alpha}, gamma={gamma}, ', end='')
        print(f'epsilon_init={epsilon_init}, epsilon_decay={epsilon_decay}.')
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge using q_learning, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        max_q_of_next_state = max(self.Q[next_state]) if not done else 0
        approx_return = reward + self.gamma * max_q_of_next_state
        self.Q[state][action] += self.alpha * (approx_return - self.Q[state][action])

        if done:  # update epsilon
            self.epsilon *= self.epsilon_decay