from graph_generator import generate_graph_batch
from hardware_simulator import get_available_devices
from benchmarker import benchmark_graph
from reward import reward_speedup
from analysis import analyze_results

def main():
    graph_batch = generate_graph_batch(batch_size=50)
    devices = get_available_devices()
    results = []

    for i, graph in enumerate(graph_batch):
        for device in devices:
            t_native = benchmark_graph(lambda: graph, device, use_xla=False)
            t_xla = benchmark_graph(lambda: graph, device, use_xla=True)
            reward = reward_speedup(t_native, t_xla)
            results.append({
                'graph_id': i,
                'device': device,
                'native_time': t_native,
                'xla_time': t_xla,
                'reward': reward
            })
            print(f"Graph {i}, Device {device}: Native {t_native:.4f}s, XLA {t_xla:.4f}s, Reward {reward:.2f}")

    analyze_results(results)

if __name__ == "__main__":
    main()
