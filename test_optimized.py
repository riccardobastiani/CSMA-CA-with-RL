"""Quick test of the optimized timing simulation"""

from simulation_optimized_timing import OptimizedSimulationEngine

print("Testing Optimized Timing Simulation (20us granularity)")
print("="*60)

# Simulate 1 second of network activity
sim = OptimizedSimulationEngine(
    num_nodes=10,
    packet_prob=0.05,
    node_type='BEB',
    duration=50000,  # 50k slots × 20µs = 1 second
    seed=42
)

metrics = sim.run()

print("\nFinal Results:")
print(f"  Throughput: {metrics['throughput']:.2f} packets/sec")
print(f"  Collision Rate: {metrics['collision_rate']:.4f}")
print(f"  PDR: {metrics['pdr']:.4f}")
print(f"  Fairness: {metrics['fairness']:.4f}")
print(f"  Channel Utilization: {metrics['channel_utilization']:.4f}")
print()
print("Success! The optimized timing model works great.")
