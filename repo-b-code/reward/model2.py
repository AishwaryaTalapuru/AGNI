import tensorflow as tf
import time
import psutil
import os

def run_graph_and_profile(graph_ops):
    process = psutil.Process(os.getpid())

    # Define the TF function without XLA
    @tf.function
    def run():
        out = tf.random.normal((128, 128))
        for op in graph_ops:
            if op == "MatMul":
                out = tf.matmul(out, tf.random.normal(out.shape))
            elif op == "ReLU":
                out = tf.nn.relu(out)
            elif op == "Add":
                out = tf.add(out, tf.ones_like(out))
            elif op == "Softmax":
                out = tf.nn.softmax(out)
        return out

    # Measure compile time (first call triggers tracing and compilation)
    mem_before_compile = process.memory_info().rss / (1024 * 1024)
    start_compile = time.time()
    run()  # first call compiles the graph
    compile_time_ms = (time.time() - start_compile) * 1000
    mem_after_compile = process.memory_info().rss / (1024 * 1024)

    # Measure latency (run the function again, now compiled)
    mem_before_run = process.memory_info().rss / (1024 * 1024)
    start_run = time.time()
    run()
    latency_ms = (time.time() - start_run) * 1000
    mem_after_run = process.memory_info().rss / (1024 * 1024)

    # Estimate memory increase during run (rough estimate)
    mem_usage_during_run = max(mem_after_compile - mem_before_compile, mem_after_run - mem_before_run)

    print(f"Compile time (ms): {compile_time_ms:.2f}")
    print(f"Latency (ms): {latency_ms:.2f}")
    print(f"Estimated memory increase (MB): {mem_usage_during_run:.2f}")

    return {
        "compile_time_ms": compile_time_ms,
        "latency_ms": latency_ms,
        "memory_mb": mem_usage_during_run
    }


#Useful when used as standlone
if __name__ == "__main__":
    ops = ["MatMul", "ReLU", "Add", "Softmax"]
    profile = run_graph_and_profile(ops)
