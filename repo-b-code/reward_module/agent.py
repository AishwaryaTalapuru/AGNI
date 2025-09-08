import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import networkx as nx

# --- CONFIG ---
ACTIONS = ['tile', 'fuse', 'quantize', 'vectorize', 'prune']
HARDWARE = ['cpu', 'gpu', 'tpu', 'iot']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- ENVIRONMENT SIMULATION ---

def generate_random_graph(num_nodes=8):
    G = nx.DiGraph()
    ops = ['conv', 'matmul', 'relu', 'softmax', 'add', 'reshape']
    for i in range(num_nodes):
        G.add_node(i, op=random.choice(ops))
    for i in range(1, num_nodes):
        G.add_edge(random.randint(0, i - 1), i)
    return G

def extract_features(G):
    op_counts = {op: 0 for op in ['conv', 'matmul', 'relu', 'softmax', 'add', 'reshape']}
    for _, data in G.nodes(data=True):
        op_counts[data['op']] += 1
    features = list(op_counts.values()) + [G.number_of_nodes(), G.number_of_edges()]
    return np.array(features, dtype=np.float32) / 10.0  # normalize

def simulate_performance(graph, hardware, action=None):
    base_latency = random.uniform(20, 100)
    base_memory = random.uniform(100, 500)

    if action:
        if action == 'tile':
            base_latency *= 0.85 if hardware in ['gpu', 'tpu'] else 1.1
        elif action == 'fuse':
            base_latency *= 0.9
            base_memory *= 0.85
        elif action == 'quantize':
            base_memory *= 0.7 if hardware in ['iot', 'tpu'] else 1.0
        elif action == 'vectorize':
            base_latency *= 0.8 if hardware == 'cpu' else 1.0
        elif action == 'prune':
            base_memory *= 0.6

    return base_latency, base_memory

def reward_fn(lat1, mem1, lat2, mem2):
    return max(0.0, (lat1 - lat2) * 0.6 + (mem1 - mem2) * 0.4)

# --- DQN MODEL ---

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- AGENT ---

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size).to(DEVICE)
        self.target = DQN(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=5000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.update_target()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        ns = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        a = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(DEVICE)
        r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        d = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        q_values = self.model(s).gather(1, a)
        next_q = self.target(ns).max(1)[0].detach().unsqueeze(1)
        expected = r + (1 - d) * self.gamma * next_q

        loss = self.loss_fn(q_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

# --- TRAINING LOOP ---

def train_agent(episodes=500):
    state_size = 8  # 6 op-types + num_nodes + num_edges
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)

    for ep in range(episodes):
        G = generate_random_graph()
        state = extract_features(G)
        hardware = random.choice(HARDWARE)

        base_lat, base_mem = simulate_performance(G, hardware)

        action_id = agent.act(state)
        action = ACTIONS[action_id]
        G_opt = G.copy()
        G_opt.graph['applied'] = action

        opt_lat, opt_mem = simulate_performance(G_opt, hardware, action)
        reward = reward_fn(base_lat, base_mem, opt_lat, opt_mem)

        next_state = extract_features(G_opt)
        done = True

        agent.remember(state, action_id, reward, next_state, done)
        agent.train_step()

        if (ep + 1) % 50 == 0:
            agent.update_target()
            print(f"[Episode {ep+1}] Action: {action}, HW: {hardware}, Reward: {reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    return agent

# --- RUN ---

if __name__ == "__main__":
    agent = train_agent()
