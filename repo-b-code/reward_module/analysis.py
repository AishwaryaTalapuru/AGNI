import numpy as np

def analyze_results(results):
    # Group by device and compute average speedup
    device_stats = {}
    for r in results:
        device = r['device']
        speedup = r['native_time'] / r['xla_time'] if r['xla_time'] > 0 else 0
        if device not in device_stats:
            device_stats[device] = []
        device_stats[device].append(speedup)
    for device, speedups in device_stats.items():
        print(f"Device: {device}, Avg Speedup: {np.mean(speedups):.2f}, Max: {np.max(speedups):.2f}, Min: {np.min(speedups):.2f}")
    # Further analysis: cluster graphs by op types, depth, etc.
