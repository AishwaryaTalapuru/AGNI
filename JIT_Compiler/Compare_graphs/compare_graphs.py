import tensorflow as tf
import time
import psutil
import os

def measure_time_and_memory(func, *args):

   
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / (1024 * 1024)  

    start_time = time.time()
    outputs = func(*args)
    end_time = time.time()


    memory_after = process.memory_info().rss / (1024 * 1024)  

    return {
        "execution_time": end_time - start_time,
        "memory_usage": memory_after - memory_before,
    }