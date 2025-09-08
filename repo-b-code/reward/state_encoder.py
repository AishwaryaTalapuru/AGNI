# meta_rl/state_encoder.py

def hardware_type_to_id(hw_type):
    mapping = {"cpu": 0, "gpu": 1, "tpu": 2}
    return mapping.get(hw_type.lower(), -1)

def encode_state(entry):
    """
    Converts one JSON graph entry into a fixed-length state vector.
    
    Args:
      entry (dict): JSON-parsed dictionary with keys 'hardware' and 'graph'.
    
    Returns:
      list: State vector [hardware_id, num_nodes, num_unique_ops].
    """
    hw_type = entry.get("hardware", {}).get("type", "cpu")
    hardware_id = hardware_type_to_id(hw_type)
    
    nodes = entry.get("graph", {}).get("nodes", [])
    num_nodes = len(nodes)
    
    op_types = set()
    for node in nodes:
        op = node.get("op", "Unknown")
        op_types.add(op)
    num_unique_ops = len(op_types)
    
    return [hardware_id, num_nodes, num_unique_ops]

# Example usage if run standalone:
if __name__ == "__main__":
    # Example JSON-like dict
    example_entry = {
        "hardware": {"type": "gpu", "cores": 4},
        "graph": {
            "nodes": [{"op": "MatMul"}, {"op": "Relu"}, {"op": "Add"}],
            "edges": [["x", "z"], ["y", "z"]]
        }
    }
    state_vector = encode_state(example_entry)
    print("Encoded state vector:", state_vector)
