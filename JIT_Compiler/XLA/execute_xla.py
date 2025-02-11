import tensorflow as tf
import time
import psutil
import os

def create_computation_graph():

    a = tf.constant(2.0, name="a")
    b = tf.constant(3.0, name="b")
    c = tf.add(a, b, name="c")  # 2 + 3 = 5
    d = tf.multiply(c, tf.constant(4.0), name="d")  # 5 * 4 = 20
    return d


"""

if __name__ == "__main__":
   
    output_tensor = create_computation_graph()

    
    @tf.function
    def original_function():
        return output_tensor

    original_metrics = measure_time_and_memory(original_function)
    print("Original Graph Metrics:")
    print(f"Execution Time: {original_metrics['execution_time']:.4f} seconds")
    print(f"Memory Usage: {original_metrics['memory_usage']:.2f} MB")


    @tf.function(jit_compile=True)
    def xla_optimized_function():
        return output_tensor

    xla_metrics = measure_time_and_memory(xla_optimized_function)
    print("\nOptimized Graph Metrics (XLA):")
    print(f"Execution Time: {xla_metrics['execution_time']:.4f} seconds")
    print(f"Memory Usage: {xla_metrics['memory_usage']:.2f} MB")

    # Compare results
    print("\nComparison:")
    print(f"Execution Time Reduction: {original_metrics['execution_time'] - xla_metrics['execution_time']:.4f} seconds")
    print(f"Memory Usage Difference: {original_metrics['memory_usage'] - xla_metrics['memory_usage']:.2f} MB")
"""