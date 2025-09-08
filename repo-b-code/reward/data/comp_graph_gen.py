import tensorflow as tf
import random
import json
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)
tf.random.set_seed(42)

OPS = ['add', 'mul', 'matmul', 'relu', 'sigmoid']

def random_shape(rank=2):
    return [random.randint(2, 4) for _ in range(rank)]

def random_tensor(shape):
    return tf.random.normal(shape, dtype=tf.float32)


def apply_random_op(op, inputs):
    if op == 'add':
        return tf.add(inputs[0], inputs[1])
    elif op == 'mul':
        return tf.multiply(inputs[0], inputs[1])
    elif op == 'matmul':
        return tf.matmul(inputs[0], inputs[1])
    elif op == 'relu':
        return tf.nn.relu(inputs[0])
    elif op == 'sigmoid':
        return tf.nn.sigmoid(inputs[0])
    else:
        raise ValueError(f"Unknown op: {op}")

def generate_tf_graph(num_nodes=10):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        nodes = []  # each element: {"tensor": tf.Tensor, "shape": list[int]}
        for _ in range(num_nodes):
            op = random.choice(OPS)
            node = None

            if op in ['add', 'mul'] and len(nodes) >= 2:
                # Find compatible shapes
                compatible_pairs = [(a, b) for a in nodes for b in nodes
                                    if a != b and a['shape'] == b['shape']]
                if compatible_pairs:
                    input1, input2 = random.choice(compatible_pairs)
                    result = apply_random_op(op, [input1['tensor'], input2['tensor']])
                    node = {"tensor": result, "shape": input1['shape']}
            elif op == 'matmul' and len(nodes) >= 2:
                # Match A: [m, k], B: [k, n]
                compatible_pairs = [(a, b) for a in nodes for b in nodes
                                    if a != b and len(a['shape']) == 2 and len(b['shape']) == 2
                                    and a['shape'][1] == b['shape'][0]]
                if compatible_pairs:
                    input1, input2 = random.choice(compatible_pairs)
                    m, _ = input1['shape']
                    _, n = input2['shape']
                    result = tf.matmul(input1['tensor'], input2['tensor'])
                    node = {"tensor": result, "shape": [m, n]}
            elif op in ['relu', 'sigmoid'] and len(nodes) >= 1:
                input1 = random.choice(nodes)
                result = apply_random_op(op, [input1['tensor']])
                node = {"tensor": result, "shape": input1['shape']}

            if node is None:
                # Fallback: generate new tensor
                shape = random_shape()
                t = random_tensor(shape)
                node = {"tensor": t, "shape": shape}

            nodes.append(node)
    return tf_graph


def extract_graph_structure(tf_graph):
    return [{"name": op.name, "type": op.type} for op in tf_graph.get_operations()]

def generate_dataset(num_graphs=100000, output_file="comp_graphs.jsonl"):
    with open(output_file, 'w') as f:
        for _ in tqdm(range(num_graphs), desc="Generating Graphs"):
            graph = generate_tf_graph(num_nodes=random.randint(6, 12))
            structure = extract_graph_structure(graph)
            json_line = json.dumps(structure)
            f.write(json_line + "\n")

if __name__ == "__main__":
    generate_dataset()
