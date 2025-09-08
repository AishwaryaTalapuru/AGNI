def reward_speedup(native_time, xla_time):
    return native_time / xla_time if xla_time > 0 else 0

def reward_latency(xla_time):
    return -xla_time

def reward_custom(native_time, xla_time, memory_usage=None, energy=None, weights=None):
    # Example: weighted sum of speedup, memory, energy
    w = weights or {'speedup': 1.0, 'memory': 0.0, 'energy': 0.0}
    speedup = native_time / xla_time if xla_time > 0 else 0
    reward = w['speedup'] * speedup
    if memory_usage is not None:
        reward -= w['memory'] * memory_usage
    if energy is not None:
        reward -= w['energy'] * energy
    return reward
