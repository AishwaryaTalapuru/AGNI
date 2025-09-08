import numpy as np

# ---------------------------
# Hardcoded input
# ---------------------------
graph_entry = {
    "hardware": {
        "type": "gpu",
        "num_cores": 16
    },
    "graph": {
        "num_nodes": 24,
        "num_ops": 7,
        "op_types": ["MatMul", "ReLU", "Add", "Softmax"]
    },
    "xla_perf": {
        "latency_ms": 110,
        "memory_mb": 800,
        "compile_time_ms": 55
    }
}

# ---------------------------
# Helper: encode inputs
# ---------------------------
def encode_graph_and_hw(entry):
    hw_type_map = {"cpu": 0, "gpu": 1, "tpu": 2}
    hw = entry["hardware"]
    graph = entry["graph"]
    perf = entry["xla_perf"]

    state_vector = [
        hw_type_map.get(hw["type"], 0),  # hardware type id
        hw.get("num_cores", 4),
        graph["num_nodes"],
        graph["num_ops"],
        len(graph["op_types"]),
        perf["latency_ms"],
        perf["memory_mb"],
        perf["compile_time_ms"]
    ]
    return np.array(state_vector, dtype=np.float32)

# ---------------------------
# Scoring function
# ---------------------------
def compute_fitness_score(state_vector, reward_weights):
    """
    Lower latency/memory/compile_time → higher score.
    """
    # Normalize the performance-related parts
    norm_latency = state_vector[5] / 1000   # assume max 1000ms
    norm_memory = state_vector[6] / 16000   # assume max 16GB
    norm_compile = state_vector[7] / 300    # assume max 300ms

    perf_vector = np.array([norm_latency, norm_memory, norm_compile])
    score = 1.0 - np.dot(reward_weights, perf_vector)  # higher is better

    return round(score, 4)

# ---------------------------
# Run example
# ---------------------------
state_vec = encode_graph_and_hw(graph_entry)

# Choose reward weights: prioritize latency > memory > compile_time
reward_weights = np.array([0.6, 0.3, 0.1])  # should sum to 1

score = compute_fitness_score(state_vec, reward_weights)

print("Fitness Score for Graph on this Hardware:", score)
