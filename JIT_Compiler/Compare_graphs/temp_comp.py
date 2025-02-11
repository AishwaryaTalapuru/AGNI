import tensorflow as tf


tensorflow_code = """
import tensorflow as tf

# Define a computation graph
a = tf.constant(5.0, name="a")
b = tf.constant(3.0, name="b")
c = tf.add(a, b, name="c")  # Addition
d = tf.multiply(c, tf.constant(2.0), name="d")  # Multiplication
"""


un_op_graph, op_graph = tf.Graph(), tf.Graph()


with un_op_graph.as_default():
    exec(tensorflow_code, globals())  


with op_graph.as_default():
    exec(tensorflow_code, globals())  


un_op_graph_def = un_op_graph.as_graph_def()
op_graph_def = op_graph.as_graph_def()

print("\nComputation Graph Nodes")
for node in un_op_graph_def.node:
    print(f"Node Name: {node.name}, Operation: {node.op}, Inputs: {node.input}")


@tf.function(jit_compile=True)  
def optimized_function():
    with op_graph.as_default():
        return tf.import_graph_def(op_graph_def, name="")

print("\nRunning XLA-Optimized Computation")
optimized_function()  

optimized_graph_def = op_graph.as_graph_def()

print("\nXLA-Optimized Computation Graph Nodes")
for node in optimized_graph_def.node:
    print(f"Node Name: {node.name}, Operation: {node.op}, Inputs: {node.input}")



