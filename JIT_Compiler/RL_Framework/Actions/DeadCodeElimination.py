import tensorflow as tf

class DeadCodeElimination:
    def __init__(self, graph_def):
        self.graph_def = graph_def

    def eliminate_dead_code(self):

        graph_def = self.graph_def
        optimized_graph_def = tf.compat.v1.GraphDef()


        node_map = {node.name: node for node in graph_def.node}
        used_nodes = set()


        consumers = {node.name: set() for node in graph_def.node}

        for node in graph_def.node:
            for inp in node.input:
                inp_name = inp.split(":")[0]  
                if inp_name in consumers:
                    consumers[inp_name].add(node.name)

        final_output_nodes = [node.name for node in graph_def.node if len(consumers[node.name]) == 0]


        def mark_used(node_name):
            if node_name in used_nodes:
                return
            used_nodes.add(node_name)
            if node_name in node_map:
                for inp in node_map[node_name].input:
                    mark_used(inp.split(":")[0])

        for output in final_output_nodes:
            mark_used(output)

        for node in graph_def.node:
            if node.name in used_nodes:
                optimized_graph_def.node.append(node)

        optimized_graph_def.library.CopyFrom(graph_def.library)

        return optimized_graph_def



"""
# Example: Define a computation graph with dead nodes
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Graph().as_default() as g:
    a = tf.constant(2, name="a")
    b = tf.constant(3, name="b")
    c = tf.multiply(a, b, name="c")  # Useful node
    d = tf.constant(5, name="d")  # Dead node (unused)
    e = tf.add(c, a, name="e")  # Useful node
    f = tf.constant(7, name="f")  # Another dead node (unused)
    
    graph_def = g.as_graph_def()

# Print unoptimized graph
print("\nUnoptimized TensorFlow Computation Graph:")
for node in graph_def.node:
    print(f"Node Name: {node.name}, Operation: {node.op}")

# Perform dead code elimination
dce = DeadCodeElimination(graph_def)
optimized_graph_def = dce.eliminate_dead_code()

# Print optimized graph
print("\nOptimized TensorFlow Computation Graph:")
for node in optimized_graph_def.node:
    print(f"Node Name: {node.name}, Operation: {node.op}")
"""