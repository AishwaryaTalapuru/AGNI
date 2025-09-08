
import tensorflow as tf
import time
import json
import resource

start_compile = time.time()
@tf.function(jit_compile=True)
def run_graph():
    out = tf.random.normal((128, 128))
    return out

compile_done = time.time()
out = run_graph()
run_done = time.time()

latency = (run_done - compile_done) * 1000
compile_time = (compile_done - start_compile) * 1000
memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

result = {
    "latency_ms": round(latency, 2),
    "compile_time_ms": round(compile_time, 2),
    "memory_mb": round(memory, 2)
}

with open("result.json", "w") as f:
    json.dump(result, f)
