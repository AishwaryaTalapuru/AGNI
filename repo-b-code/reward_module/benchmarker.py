import tensorflow as tf
from hardware_simulator import run_on_device, simulate_iot

def benchmark_graph(graph_fn, device, use_xla=False):
    if use_xla:
        graph_fn = tf.function(graph_fn, jit_compile=True)
    if device == 'iot':
        return simulate_iot(graph_fn)
    else:
        return run_on_device(graph_fn, device)
