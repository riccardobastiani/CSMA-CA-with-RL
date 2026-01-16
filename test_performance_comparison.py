"""
Performance Comparison: Different Time Granularities

Compares three approaches:
1. Original: Abstract slots, instantaneous TX
2. Realistic (1µs): Very fine granularity, slow but accurate
3. Optimized (20µs): SlotTime granularity, fast and accurate
"""

import time
from simulation import SimulationEngine as OriginalEngine
from simulation_realistic_timing import SimulationEngine as RealisticEngine
from simulation_optimized_timing import OptimizedSimulationEngine

print("="*70)
print("PERFORMANCE COMPARISON: Time Granularity Trade-offs")
print("="*70)
print()

# Configuration
NUM_NODES = 10
PACKET_PROB = 0.05
SEED = 42

# Durations for fair comparison (~1 second of simulated time each)
ORIGINAL_SLOTS = 50000
OPTIMIZED_SLOTS = 50000  # 50k slots * 20µs = 1 second
REALISTIC_US = 1000000   # 1 million µs = 1 second

print(f"Configuration: {NUM_NODES} nodes, p={PACKET_PROB}, BEB protocol")
print()

# =============================================================================
# Test 1: Original Simulation
# =============================================================================
print("-" * 70)
print("TEST 1: ORIGINAL SIMULATION (Abstract Slots)")
print("-" * 70)
print(f"Duration: {ORIGINAL_SLOTS:,} slots")
print()

start = time.time()
sim1 = OriginalEngine(num_nodes=NUM_NODES, packet_prob=PACKET_PROB,
                      node_type='BEB', duration=ORIGINAL_SLOTS, seed=SEED)
metrics1 = sim1.run()
elapsed1 = time.time() - start

print(f"\nResults:")
print(f"  Execution Time: {elapsed1:.2f}s")
print(f"  Throughput: {metrics1['throughput']:.4f}")
print(f"  Collision Rate: {metrics1['collision_rate']:.4f}")
print(f"  PDR: {metrics1['pdr']:.4f}")
print()

# =============================================================================
# Test 2: Optimized Simulation (20µs granularity)
# =============================================================================
print("-" * 70)
print("TEST 2: OPTIMIZED TIMING (20µs Slot Granularity)")
print("-" * 70)
print(f"Duration: {OPTIMIZED_SLOTS:,} slots = {OPTIMIZED_SLOTS * 20 / 1000:.1f}ms")
print()

start = time.time()
sim2 = OptimizedSimulationEngine(num_nodes=NUM_NODES, packet_prob=PACKET_PROB,
                                  node_type='BEB', duration=OPTIMIZED_SLOTS, seed=SEED)
metrics2 = sim2.run()
elapsed2 = time.time() - start

print(f"\nResults:")
print(f"  Execution Time: {elapsed2:.2f}s")
print(f"  Throughput: {metrics2['throughput']:.2f} packets/sec")
print(f"  Collision Rate: {metrics2['collision_rate']:.4f}")
print(f"  PDR: {metrics2['pdr']:.4f}")
print(f"  Channel Utilization: {metrics2['channel_utilization']:.4f}")
print()

# =============================================================================
# Test 3: Realistic Simulation (1µs granularity) - SHORT VERSION
# =============================================================================
print("-" * 70)
print("TEST 3: REALISTIC TIMING (1µs Granularity)")
print("-" * 70)
print(f"Duration: {REALISTIC_US:,}µs = {REALISTIC_US / 1000:.1f}ms")
print("NOTE: This is 20x slower due to finer granularity!")
print()

start = time.time()
sim3 = RealisticEngine(num_nodes=NUM_NODES, packet_prob=PACKET_PROB,
                       node_type='BEB', duration=REALISTIC_US, seed=SEED)
metrics3 = sim3.run()
elapsed3 = time.time() - start

print(f"\nResults:")
print(f"  Execution Time: {elapsed3:.2f}s")
print(f"  Throughput: {metrics3['throughput']:.2f} packets/sec")
print(f"  Collision Rate: {metrics3['collision_rate']:.4f}")
print(f"  PDR: {metrics3['pdr']:.4f}")
print(f"  Channel Utilization: {metrics3['channel_utilization']:.4f}")
print()

# =============================================================================
# Comparison Summary
# =============================================================================
print("="*70)
print("PERFORMANCE SUMMARY")
print("="*70)
print()

print(f"{'Model':<25} {'Time Steps':<15} {'Exec Time':<12} {'Speedup':<10}")
print("-" * 70)
print(f"{'Original (abstract)':<25} {ORIGINAL_SLOTS:>14,} {elapsed1:>11.2f}s {'1.0x':>9}")
print(f"{'Optimized (20µs)':<25} {OPTIMIZED_SLOTS:>14,} {elapsed2:>11.2f}s {elapsed1/elapsed2:>9.1f}x")
print(f"{'Realistic (1µs)':<25} {REALISTIC_US:>14,} {elapsed3:>11.2f}s {elapsed1/elapsed3:>9.1f}x")
print()

print("="*70)
print("RECOMMENDATION")
print("="*70)
print()
print("For most purposes, use the OPTIMIZED (20µs) model:")
print("  + Fast execution (similar to original)")
print("  + Accurate MAC timing (DIFS, SIFS, TX duration)")
print("  + Easy to understand (1 slot = 1 SlotTime)")
print()
print("Use REALISTIC (1µs) model only when:")
print("  - You need extremely precise timing analysis")
print("  - Simulating very short durations (<100ms)")
print("  - Validating timing-critical protocol behavior")
print()
print("Use ORIGINAL model when:")
print("  - You don't need realistic timing")
print("  - Running very long simulations (hours of network time)")
print("  - Focusing on relative protocol performance")
print("="*70)
