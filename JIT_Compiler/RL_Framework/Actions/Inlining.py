import tensorflow as tf

class Inlining:
    def __init__(self, graph_def):
        self.graph_def = graph_def

    def inline_functions(self):
        graph_def = self.graph_def
        optimized_graph_def = tf.compat.v1.GraphDef()

        for node in graph_def.node:
            if node.op == "StatefulPartitionedCall" or node.op == "PartitionedCall":
                function_name = node.attr["f"].func.name

                function_def = None
                for func in graph_def.library.function:
                    if func.signature.name == function_name:
                        function_def = func
                        break

                if function_def is None:
                    raise ValueError(f"Function {function_name} not found in the graph's library.")

                for func_node in function_def.node_def:
                    new_node = tf.compat.v1.NodeDef()
                    new_node.CopyFrom(func_node)


                    new_node.name = f"{node.name}/{func_node.name}"

                    new_inputs = [
                        f"{node.name}/{inp}" if ":" in inp else inp
                        for inp in func_node.input
                    ]
                    new_node.input[:] = new_inputs  

                    optimized_graph_def.node.append(new_node)
            else:
               
                optimized_graph_def.node.append(node)

       
        optimized_graph_def.library.CopyFrom(graph_def.library)

        return optimized_graph_def







"""
# Example usage
@tf.function
def add_and_multiply(a, b):
    c = tf.add(a, b)
    d = tf.multiply(c, b)
    return d

tf.compat.v1.reset_default_graph()
with tf.compat.v1.Graph().as_default() as g:
    a = tf.constant(2.0, name="a")
    b = tf.constant(3.0, name="b")
    result = add_and_multiply(a, b)

        # Get computation graph
    graph_def = g.as_graph_def()

    print("\nunOptimized TensorFlow Computation Graph:")
    for node in graph_def.node:
        print(f"Node Name: {node.name}, Operation: {node.op}")

    # Perform inlining
    obj = Inlining(graph_def)
    optimized_graph_def = obj.inline_functions()

        # Print optimized graph
    print("\nOptimized TensorFlow Computation Graph:")
    for node in optimized_graph_def.node:
        print(f"Node Name: {node.name}, Operation: {node.op}")
"""

