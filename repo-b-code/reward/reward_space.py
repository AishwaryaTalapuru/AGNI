
import numpy as np

REWARD_ACTIONS = [
    np.array([1.0, 0.0, 0.0]),     # latency only
    np.array([0.0, 1.0, 0.0]),     # memory only
    np.array([0.0, 0.0, 1.0]),     # compile time only
    np.array([0.5, 0.5, 0.0]),     # latency + memory
    np.array([0.4, 0.3, 0.3]),     # balanced
    np.array([0.33, 0.33, 0.33])   # equally balanced
]

def get_action(index):
    return REWARD_ACTIONS[index]

def num_actions():
    return len(REWARD_ACTIONS)
