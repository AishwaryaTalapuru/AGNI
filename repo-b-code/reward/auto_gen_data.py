import random
import json

OP_TYPES = ["MatMul", "ReLU", "Add", "Softmax", "Sigmoid", "Conv2D", "Tanh", "MaxPool"]
HARDWARE_TYPES = ["cpu", "gpu", "tpu"]

def simulate_perf(graph, hardware):
    base_latency = graph["num_nodes"] * 2 + graph["num_ops"] * 5
    base_memory = graph["num_nodes"] * 10 + graph["num_ops"] * 20
    base_compile = graph["num_ops"] * 3

    hw_type = hardware["type"]
    cores = hardware["num_cores"]

    # Simulate effects based on hardware type
    if hw_type == "cpu":
        latency = base_latency * 1.5
        memory = base_memory * 0.8
        compile_time = base_compile * 1.2
    elif hw_type == "gpu":
        latency = base_latency * 0.8
        memory = base_memory * 1.2
        compile_time = base_compile
    elif hw_type == "tpu":
        latency = base_latency * 0.6
        memory = base_memory * 1.5
        compile_time = base_compile * 1.5

    # Add noise
    latency += random.uniform(-10, 10)
    memory += random.uniform(-50, 50)
    compile_time += random.uniform(-5, 5)

    return {
        "latency_ms": max(1, round(latency, 2)),
        "memory_mb": max(1, round(memory, 2)),
        "compile_time_ms": max(1, round(compile_time, 2))
    }

def generate_random_graph():
    num_nodes = random.randint(10, 100)
    num_ops = random.randint(2, 10)
    op_types = random.choices(OP_TYPES, k=random.randint(1, 5))
    return {
        "num_nodes": num_nodes,
        "num_ops": num_ops,
        "op_types": op_types
    }

def generate_random_hardware():
    hw_type = random.choice(HARDWARE_TYPES)
    if hw_type == "cpu":
        cores = random.choice([2, 4, 8])
    elif hw_type == "gpu":
        cores = random.choice([8, 16, 32])
    else:  # TPU
        cores = random.choice([32, 64])
    return {"type": hw_type, "num_cores": cores}

def generate_dataset(num_samples=1000, out_file="comp_graphs_with_perf.jsonl"):
    with open(out_file, "w") as f:
        for _ in range(num_samples):
            graph = generate_random_graph()
            hardware = generate_random_hardware()
            xla_perf = simulate_perf(graph, hardware)

            entry = {
                "hardware": hardware,
                "graph": graph,
                "xla_perf": xla_perf
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Dataset saved to {out_file}")

# Run this to generate
if __name__ == "__main__":
    generate_dataset(num_samples=1000)
