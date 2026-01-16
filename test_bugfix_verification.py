"""
Test to verify the bug fix for optimized timing simulation.

The bug was: backoff/DIFS counters were not accounting for time granularity,
causing nodes to wait 20x longer than they should in the optimized (20µs) mode.

Expected behavior after fix:
- BEB and RL should have similar performance
- RL should potentially outperform BEB (not be 80% worse!)
- Throughput should make sense given network density
"""

from simulation_optimized_timing import OptimizedSimulationEngine

print("="*70)
print("OPTIMIZED TIMING - BUG FIX VERIFICATION")
print("="*70)
print()

# Test with 10 nodes - where the bug was most obvious
NUM_NODES = 10
PACKET_PROB = 0.05
DURATION = 50000  # 1 second
SEED = 42

print(f"Configuration: {NUM_NODES} nodes, p={PACKET_PROB}, 1 second simulation")
print()

# Test BEB
print("-" * 70)
print("Testing BEB (Baseline)")
print("-" * 70)
sim_beb = OptimizedSimulationEngine(
    num_nodes=NUM_NODES,
    packet_prob=PACKET_PROB,
    node_type='BEB',
    duration=DURATION,
    seed=SEED
)
metrics_beb = sim_beb.run()

print()
print(f"BEB Results:")
print(f"  Throughput: {metrics_beb['throughput']:.4f} packets/sec")
print(f"  Collision Rate: {metrics_beb['collision_rate']:.4f}")
print(f"  PDR: {metrics_beb['pdr']:.4f}")
print(f"  Channel Utilization: {metrics_beb['channel_utilization']:.4f}")
print()

# Test RL
print("-" * 70)
print("Testing RL (Standard epsilon=0.1)")
print("-" * 70)
sim_rl = OptimizedSimulationEngine(
    num_nodes=NUM_NODES,
    packet_prob=PACKET_PROB,
    node_type='RL',
    duration=DURATION,
    seed=SEED,
    epsilon=0.1,
    alpha=0.1,
    gamma=0.9
)
metrics_rl = sim_rl.run()

print()
print(f"RL Results:")
print(f"  Throughput: {metrics_rl['throughput']:.4f} packets/sec")
print(f"  Collision Rate: {metrics_rl['collision_rate']:.4f}")
print(f"  PDR: {metrics_rl['pdr']:.4f}")
print(f"  Channel Utilization: {metrics_rl['channel_utilization']:.4f}")
print()

# Comparison
print("="*70)
print("COMPARISON")
print("="*70)
print()

throughput_ratio = metrics_rl['throughput'] / metrics_beb['throughput'] if metrics_beb['throughput'] > 0 else 0
collision_diff = metrics_beb['collision_rate'] - metrics_rl['collision_rate']

print(f"Throughput:")
print(f"  BEB: {metrics_beb['throughput']:.4f} packets/sec")
print(f"  RL:  {metrics_rl['throughput']:.4f} packets/sec")
print(f"  Ratio: {throughput_ratio:.2f}x")

if throughput_ratio < 0.5:
    print(f"  ❌ PROBLEM: RL is {(1-throughput_ratio)*100:.0f}% worse than BEB!")
    print(f"     This suggests the bug is still present.")
elif throughput_ratio > 0.8:
    print(f"  ✓ GOOD: RL is competitive with BEB")
else:
    print(f"  ⚠ MARGINAL: RL is somewhat worse, might need tuning")

print()
print(f"Collision Rate:")
print(f"  BEB: {metrics_beb['collision_rate']:.4f}")
print(f"  RL:  {metrics_rl['collision_rate']:.4f}")
print(f"  Difference: {collision_diff:+.4f} ({'RL better' if collision_diff > 0 else 'BEB better'})")

print()
print("="*70)
print("EXPECTED vs BUGGY BEHAVIOR")
print("="*70)
print()
print("BUGGY (before fix):")
print("  - BEB throughput: ~0.40, RL throughput: ~0.08-0.14")
print("  - RL is 60-80% worse than BEB (makes no sense!)")
print("  - Caused by: backoff counter not decrementing properly")
print()
print("EXPECTED (after fix):")
print("  - Both BEB and RL should have similar throughput")
print("  - RL might be slightly better or worse depending on tuning")
print("  - Throughput ratio should be 0.8-1.2x")
print()

if throughput_ratio > 0.8:
    print("✓✓✓ BUG FIX SUCCESSFUL! ✓✓✓")
else:
    print("❌❌❌ BUG STILL PRESENT OR NEW ISSUE ❌❌❌")
print()
