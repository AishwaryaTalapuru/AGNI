import tensorflow as tf
import time

def run_on_device(graph_fn, device):
    with tf.device(device):
        start = time.time()
        result = graph_fn()
        _ = result.numpy()
        end = time.time()
    return end - start

def simulate_iot(graph_fn):
    start = time.time()
    result = graph_fn()
    _ = result.numpy()
    time.sleep(0.05)  # Artificial delay for IoT
    end = time.time()
    return end - start

def get_available_devices():
    devices = ['/CPU:0']
    if tf.config.list_physical_devices('GPU'):
        devices.append('/GPU:0')
    # TPU detection (if available)
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        devices.append('/TPU:0')
    except Exception:
        pass
    devices.append('iot')
    return devices
