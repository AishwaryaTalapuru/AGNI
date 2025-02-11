import tensorflow as tf

class LoopUnrolling:
    def __init__(self, graph_def):
        self.graph_def = graph_def

    def unroll_loops(self, max_iterations=10):
        graph_def = self.graph_def
        optimized_graph_def = tf.compat.v1.GraphDef()

        for node in graph_def.node:
            if node.op == "While" or node.op == "StatelessWhile":
                cond_fn_name = node.attr["cond"].func.name
                body_fn_name = node.attr["body"].func.name

                cond_fn_def, body_fn_def = None, None
                for func in graph_def.library.function:
                    if func.signature.name == cond_fn_name:
                        cond_fn_def = func
                    elif func.signature.name == body_fn_name:
                        body_fn_def = func

                if cond_fn_def is None or body_fn_def is None:
                    raise ValueError("Loop condition or body function not found in graph.")

                loop_inputs = node.input
                current_inputs = loop_inputs

                for i in range(max_iterations):
                    scope = f"{node.name}/unrolled_{i}"
                    for body_node in body_fn_def.node_def:
                        new_node = tf.compat.v1.NodeDef()
                        new_node.CopyFrom(body_node)
                        new_node.name = f"{scope}/{body_node.name}"
                        new_node.input[:] = [f"{scope}/{inp}" if inp in loop_inputs else inp for inp in body_node.input]
                        optimized_graph_def.node.append(new_node)

                    current_inputs = [f"{scope}/{output}" for output in body_fn_def.signature.output_arg]

                for output in current_inputs:
                    out_node = tf.compat.v1.NodeDef()
                    out_node.name = output
                    out_node.op = "Identity"
                    out_node.input.append(output)
                    optimized_graph_def.node.append(out_node)
            else:
                optimized_graph_def.node.append(node)

        optimized_graph_def.library.CopyFrom(graph_def.library)
        return optimized_graph_def



"""
def loop_body(i, x):
    return i + 1, x * 2

def loop_cond(i, x):
    return i < 5

tf.compat.v1.reset_default_graph()
with tf.compat.v1.Graph().as_default() as g:
    i = tf.constant(0, name="i")
    x = tf.constant(1, name="x")
    result = tf.while_loop(loop_cond, loop_body, [i, x], name="while_loop")

    graph_def = g.as_graph_def()

print("\nUnoptimized TensorFlow Computation Graph:")
for node in graph_def.node:
    print(f"Node Name: {node.name}, Operation: {node.op}")

obj = LoopUnrolling(graph_def)
op_graph = obj.unroll_loops(max_iterations=5)

print("\nOptimized TensorFlow Computation Graph:")
for node in op_graph.node:
    print(f"Node Name: {node.name}, Operation: {node.op}")
"""
