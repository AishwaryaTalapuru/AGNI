import numpy as np
import tensorflow as tf
from collections import deque
import random
import time
import psutil

from Actions.ConstantFolder import ConstantFolder
from Actions.DeadCodeElimination import DeadCodeElimination
from Actions.Inlining import Inlining
from Actions.LoopUnrolling import LoopUnrolling


def traverse_graph(graph_def):
    print("\nTensorFlow Computation Graph:")
    for node in graph_def.node:
        print(f"Node Name: {node.name}")
        print(f"  Operation Type: {node.op}")
        print(f"  Inputs: {[inp for inp in node.input]}")
        print("-" * 40)


def read_tf_code_as_string():

    file_path = "../Training/tf_code_1.txt"


    with open(file_path, "r") as file:
        graph_code = file.read()


    return graph_code

def create_graph_from_string(graph_code):


    with tf.compat.v1.Graph().as_default() as g:

        exec(graph_code)

        graph_def = g.as_graph_def()
    return graph_def


graph_code = read_tf_code_as_string()


graph_def = create_graph_from_string(graph_code)


print("The Comp Graph under observation is : ")
traverse_graph(graph_def)



LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
EPSILON = 1.0  # Exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPISODES = 100
MAX_STEPS_PER_EPISODE = 20  # Maximum steps per episode


unoptimized_graphs = [graph_def]


class GraphOptimizationEnv:
    def __init__(self, unoptimized_graphs):
        self.unoptimized_graphs = unoptimized_graphs
        self.current_graph_idx = 0
        self.current_graph = unoptimized_graphs[self.current_graph_idx]
        self.previous_graph = None
        self.xla_optimized_graph = self.optimize_with_xla(self.current_graph)
        self.actions = ["loop_unrolling", "constant_folding", "inlining", "dead_code_elimination"]
        self.action_space = len(self.actions)
        self.step_count = 0  
    def reset(self):
        self.current_graph_idx = 0
        self.current_graph = self.unoptimized_graphs[self.current_graph_idx]
        self.previous_graph = None
        self.xla_optimized_graph = self.optimize_with_xla(self.current_graph)
        self.step_count = 0  
        return self.encode_graph(self.current_graph)

    def step(self, action):

        new_graph = self.apply_action(self.current_graph, action)
        self.previous_graph = self.current_graph
        self.current_graph = new_graph


        xla_time, xla_memory = self.measure_performance(self.xla_optimized_graph)
        rl_time, rl_memory = self.measure_performance(self.current_graph)
        prev_time, prev_memory = self.measure_performance(self.previous_graph) if self.previous_graph else (float('inf'), float('inf'))


        if rl_time < xla_time and rl_memory < xla_memory:
            reward = 100 
            done = True  
        elif rl_time < prev_time and rl_memory < prev_memory:
            reward = 10  
            done = False  
        else:
            reward = -10  
            done = False  

        self.step_count += 1

        
        if self.step_count >= MAX_STEPS_PER_EPISODE:
            done = True

        return self.encode_graph(self.current_graph), reward, done

    def encode_graph(self, graph):
        return np.zeros(128)  

    def apply_action(self, graph, action):
        if self.actions[action] == "loop_unrolling":
            return self.loop_unrolling(graph)
        elif self.actions[action] == "constant_folding":
            return self.constant_folding(graph)
        elif self.actions[action] == "inlining":
            return self.inlining(graph)
        elif self.actions[action] == "dead_code_elimination":
            return self.dead_code_elimination(graph)
        else:
            raise ValueError("Invalid action")

    def optimize_with_xla(self, graph):
        with tf.Graph().as_default():
            tf.import_graph_def(graph, name="")
            optimized_graph_def = tf.function(lambda: None, jit_compile=True).get_concrete_function().graph.as_graph_def()
        return optimized_graph_def

    def measure_performance(self, graph):
        with tf.Graph().as_default():
            tf.import_graph_def(graph, name="")
            with tf.compat.v1.Session() as sess:
                start_time = time.time()
                sess.run(tf.compat.v1.global_variables_initializer())
                end_time = time.time()
                execution_time = end_time - start_time
                memory_usage = psutil.virtual_memory().used / (1024 * 1024)
        return execution_time, memory_usage

    def loop_unrolling(self, graph):
        print("Applying loop unrolling...")
        obj = LoopUnrolling(graph)
        graph = obj.unroll_loops()
        return graph

    def constant_folding(self, graph):
        print("Applying constant folding...")
        obj = ConstantFolder(graph)
        graph = obj.fold_constants()
        return graph

    def inlining(self, graph):
        print("Applying inlining...")
        obj = Inlining(graph)
        graph = obj.inline_functions()
        return graph

    def dead_code_elimination(self, graph):
        print("Applying dead code elimination...")
        obj = DeadCodeElimination(graph)
        graph = obj.eliminate_dead_code()
        return graph


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_SIZE)  
        self.epsilon = EPSILON


        self.model = self.build_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation="relu", input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(self.action_dim, activation="linear")
        ])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)  # Explore
        else:
            q_values = self.model.predict(state[np.newaxis], verbose=0)
            return np.argmax(q_values[0])  # Exploit

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])


        target_q = rewards + GAMMA * np.max(self.model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        target_q_full = self.model.predict(states, verbose=0)
        target_q_full[np.arange(batch_size), actions] = target_q


        self.model.train_on_batch(states, target_q_full)


        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY


env = GraphOptimizationEnv(unoptimized_graphs)
agent = DQNAgent(state_dim=128, action_dim=env.action_space)  

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay(BATCH_SIZE)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")