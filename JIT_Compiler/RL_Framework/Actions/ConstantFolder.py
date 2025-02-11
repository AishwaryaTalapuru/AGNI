import tensorflow as tf

class ConstantFolder:
    def __init__(self, graph_def):
        self.graph_def = graph_def
        self.constant_values = {}  
        self.new_nodes = []  
    def is_constant(self, node):
        return node.op == "Const"

    def get_const_value(self, node):
        tensor_attr = node.attr["value"].tensor
        dtype = tensor_attr.dtype

        if dtype == tf.float32.as_datatype_enum:
            return tensor_attr.float_val[0]
        elif dtype == tf.int32.as_datatype_enum:
            return tensor_attr.int_val[0]
        elif dtype == tf.bool.as_datatype_enum:
            return tensor_attr.bool_val
        else:
            raise ValueError(f"Unsupported tensor type: {dtype}")

    def evaluate_operation(self, node, input_values):
        op_type = node.op

        if op_type in ["Add", "AddV2"]:
            return input_values[0] + input_values[1]
        elif op_type == "Sub":
            return input_values[0] - input_values[1]
        elif op_type == "Mul":
            return input_values[0] * input_values[1]
        elif op_type == "Div":
            return input_values[0] / input_values[1] if input_values[1] != 0 else None
        elif op_type == "Pow":
            return input_values[0] ** input_values[1]
        else:
            return None  

    def fold_constants(self):

        for node in self.graph_def.node:
            if node.op in ["Add", "AddV2", "Sub", "Mul", "Div", "Pow"]:
                
                input_names = [inp.split(":")[0] for inp in node.input]

                if all(inp in self.constant_values for inp in input_names):
                    input_values = [self.constant_values[inp] for inp in input_names]


                    new_value = self.evaluate_operation(node, input_values)
                    if new_value is not None:

                        new_const_node = tf.compat.v1.NodeDef()
                        new_const_node.op = "Const"
                        new_const_node.name = node.name


                        new_const_node.attr["dtype"].type = tf.float32.as_datatype_enum
                        tensor_proto = tf.make_tensor_proto(new_value, dtype=tf.float32)
                        new_const_node.attr["value"].tensor.CopyFrom(tensor_proto)


                        self.constant_values[node.name] = new_value
                        self.new_nodes.append(new_const_node)
                        continue  


            self.new_nodes.append(node)
            if self.is_constant(node):
                self.constant_values[node.name] = self.get_const_value(node)


        optimized_graph = tf.compat.v1.GraphDef()
        optimized_graph.node.extend(self.new_nodes)
        return optimized_graph



"""
# Example: Creating a TensorFlow computation graph
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Graph().as_default() as g:
    a = tf.constant(5.0, name="a")
    b = tf.constant(3.0, name="b")
    c = tf.multiply(a, b, name="c")  # 5 * 3 = 15
    d = tf.add(c, tf.constant(2.0), name="d")  # 15 + 2 = 17

    # Get computation graph
    graph_def = g.as_graph_def()
for node in graph_def.node:
  print(f"Node Name: {node.name}, Operation: {node.op}, Value: {node}")
# Perform constant folding
folder = ConstantFolder(graph_def)
optimized_graph = folder.fold_constants()

# Print optimized graph
print("\nOptimized TensorFlow Computation Graph:")
for node in optimized_graph.node:
    print(f"Node Name: {node.name}, Operation: {node.op}, Value: {node}")
"""
