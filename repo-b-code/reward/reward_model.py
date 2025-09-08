import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features_and_labels(file_path):
    data = []
    labels = []

    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)

            hw = entry.get("hardware", {})
            graph = entry.get("graph", {})
            perf = entry.get("xla_perf", {})

            # Skip if key info missing
            if not hw or not graph or not perf:
                continue

            hw_type = hw.get("type", "cpu")
            num_cores = hw.get("num_cores", 1)

            num_nodes = graph.get("num_nodes", 0)
            num_ops = graph.get("num_ops", 0)
            op_types = graph.get("op_types", [])

            # Count each op type
            op_counts = {
                "MatMul": 0,
                "ReLU": 0,
                "Add": 0,
                "Softmax": 0
            }
            for op in op_types:
                if op in op_counts:
                    op_counts[op] += 1

            latency = perf.get("latency_ms", 9999)
            memory = perf.get("memory_mb", 9999)
            compile_time = perf.get("compile_time_ms", 9999)

            # Handcrafted reward as initial label
            score = latency + compile_time + 0.1 * memory
            reward = 1000 / (score + 1)

            data.append([
                hw_type, num_cores, num_nodes, num_ops,
                op_counts["MatMul"], op_counts["ReLU"],
                op_counts["Add"], op_counts["Softmax"],
                latency, memory, compile_time
            ])
            labels.append(reward)

    return np.array(data), np.array(labels)

# -----------------------------
# Model Training Pipeline
# -----------------------------
def train_reward_model(file_path):
    raw_X, y = extract_features_and_labels(file_path)

    # Extract hardware type and one-hot encode it
    hw_types = raw_X[:, 0].reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    hw_encoded = encoder.fit_transform(hw_types)

    # Convert the rest of the columns to float
    X_numeric = raw_X[:, 1:].astype(float)

    # Final input matrix
    X = np.hstack([hw_encoded, X_numeric])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model trained. RMSE on test set: {rmse:.4f}")

    return model, encoder

# -----------------------------
# Reward Prediction Function
# -----------------------------
def predict_reward(graph_entry, model, encoder):
    hw = graph_entry.get("hardware", {})
    graph = graph_entry.get("graph", {})
    perf = graph_entry.get("xla_perf", {})

    hw_type = hw.get("type", "cpu")
    num_cores = hw.get("num_cores", 1)

    num_nodes = graph.get("num_nodes", 0)
    num_ops = graph.get("num_ops", 0)
    op_types = graph.get("op_types", [])

    op_counts = {
        "MatMul": 0,
        "ReLU": 0,
        "Add": 0,
        "Softmax": 0
    }
    for op in op_types:
        if op in op_counts:
            op_counts[op] += 1

    latency = perf.get("latency_ms", 9999)
    memory = perf.get("memory_mb", 9999)
    compile_time = perf.get("compile_time_ms", 9999)

    hw_encoded = encoder.transform([[hw_type]])
    numeric_features = np.array([[
        num_cores, num_nodes, num_ops,
        op_counts["MatMul"], op_counts["ReLU"],
        op_counts["Add"], op_counts["Softmax"],
        latency, memory, compile_time
    ]])
    full_input = np.hstack([hw_encoded, numeric_features])
    return model.predict(full_input)[0]



# Train the model
model, encoder = train_reward_model("comp_graphs_with_perf.jsonl")

# Predict for a single graph
graph_entry = {
    "hardware": {"type": "gpu", "num_cores": 16},
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

reward = predict_reward(graph_entry, model, encoder)
print(f"Predicted Reward: {reward:.2f}")
