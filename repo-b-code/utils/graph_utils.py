def get_all_ops(graph):
    return [op for op in graph.get_operations()]

def get_op_by_name(graph, name):
    return graph.get_operation_by_name(name)
