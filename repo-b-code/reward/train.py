import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from agent import MetaDQNAgent
from state_encoder import encode_state
from reward_space import REWARD_ACTIONS

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
LR = 1e-3
MEMORY_SIZE = 10000
NUM_EPISODES = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay buffer
memory = deque(maxlen=MEMORY_SIZE)

# Initialize DQN and optimizer
input_dim = 3  # hardware_id, num_nodes, num_unique_ops
num_actions = len(REWARD_ACTIONS)

dqn = MetaDQNAgent(input_dim, num_actions).to(device)
optimizer = torch.optim.Adam(dqn.parameters(), lr=LR)

epsilon = EPSILON_START

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = dqn(state_tensor)
        return q_values.argmax().item()

def simulate_primary_rl_performance(reward_weights, graph_entry):
    """
    TODO: Replace with real Primary RL run.
    For now, simulate performance based on reward weights.
    """
    base_latency = graph_entry.get("xla_perf", {}).get("latency_ms", 200)
    base_memory = graph_entry.get("xla_perf", {}).get("memory_mb", 1000)
    base_compile = graph_entry.get("xla_perf", {}).get("compile_time_ms", 50)

    latency = base_latency - 20 * reward_weights[0] + random.uniform(-5, 5)
    memory = base_memory - 100 * reward_weights[1] + random.uniform(-20, 20)
    compile_time = base_compile - 5 * reward_weights[2] + random.uniform(-2, 2)

    return {"latency": latency, "memory": memory, "compile_time": compile_time}

def compute_meta_reward(primary_perf, xla_perf):
    return (
        (xla_perf["latency_ms"] - primary_perf["latency"]) +
        0.5 * (xla_perf["memory_mb"] - primary_perf["memory"]) +
        1.0 * (xla_perf["compile_time_ms"] - primary_perf["compile_time"])
    )

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states = zip(*batch)

    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions).unsqueeze(1).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)

    q_values = dqn(states_tensor).gather(1, actions_tensor)
    with torch.no_grad():
        next_q_values = dqn(next_states_tensor).max(1)[0].unsqueeze(1)

    expected_q_values = rewards_tensor + (GAMMA * next_q_values)

    loss = F.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(graphs):
    global epsilon
    for episode in range(NUM_EPISODES):
        for graph in graphs:
            graph_entry = graph
            state = encode_state(graph_entry)
            action_idx = select_action(state, epsilon)
            reward_weights = REWARD_ACTIONS[action_idx]

            primary_perf = simulate_primary_rl_performance(reward_weights, graph_entry)
            xla_perf = graph_entry.get("xla_perf", {"latency_ms": 200, "memory_mb": 1000, "compile_time_ms": 50})

            reward = compute_meta_reward(primary_perf, xla_perf)
            
            # For next_state, just reuse current state for simplicity
            next_state = state

            memory.append((state, action_idx, reward, next_state))

            optimize_model()

            # Decay epsilon
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {reward:.2f}, Epsilon: {epsilon:.2f}")

if __name__ == "__main__":
    import json
    # Load dataset of graphs
    data = []
    with open("data/comp_graphs.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            if isinstance(obj, list):
                data.extend(obj)
            else:
                data.append(obj)
        print(len(data))


    train(data)
