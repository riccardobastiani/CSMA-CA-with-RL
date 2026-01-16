"""
Comparison test between original and realistic timing simulations.

This script runs both simulations with the same parameters and compares:
- Throughput
- Collision rate
- Packet delivery ratio
- Fairness

The realistic timing version should show similar relative performance
but with more accurate time-domain behavior.
"""

import sys
import time
import numpy as np

# Original implementation
from simulation import SimulationEngine as OriginalEngine

# Realistic timing implementation
from simulation_realistic_timing import SimulationEngine as RealisticEngine
from models_realistic_timing import MACTiming


def run_comparison():
    """Compare both simulation approaches"""
    
    print("="*80)
    print("CSMA/CA SIMULATION COMPARISON")
    print("Original (slot-based) vs. Realistic Timing (IEEE 802.11 DCF)")
    print("="*80)
    print()
    
    # Test configuration
    NUM_NODES = 10
    PACKET_PROB = 0.05
    SEED = 42
    
    # Duration configuration
    # Original: 10,000 "abstract" time slots
    ORIGINAL_DURATION = 10000
    
    # Realistic: We want approximately the same "simulation time"
    # If original slot ~ 20µs (backoff slot time), then 10k slots = 200ms
    # But transmissions are instantaneous in original
    # For fair comparison, let's run realistic for 200ms = 200,000 µs
    REALISTIC_DURATION = 200000  # 200ms in microseconds
    
    print(f"Configuration:")
    print(f"  Nodes: {NUM_NODES}")
    print(f"  Packet Probability: {PACKET_PROB}")
    print(f"  Random Seed: {SEED}")
    print(f"  Node Type: BEB (Binary Exponential Backoff)")
    print()
    
    # ========================================================================
    # Run Original Simulation
    # ========================================================================
    print("-" * 80)
    print("RUNNING ORIGINAL SIMULATION (Slot-based)")
    print("-" * 80)
    
    start_time = time.time()
    
    sim_original = OriginalEngine(
        num_nodes=NUM_NODES,
        packet_prob=PACKET_PROB,
        node_type='BEB',
        duration=ORIGINAL_DURATION,
        seed=SEED
    )
    
    metrics_original = sim_original.run()
    
    elapsed_original = time.time() - start_time
    
    print(f"\nOriginal Results:")
    print(f"  Throughput: {metrics_original['throughput']:.4f}")
    print(f"  Collision Rate: {metrics_original['collision_rate']:.4f}")
    print(f"  PDR: {metrics_original['pdr']:.4f}")
    print(f"  Fairness: {metrics_original['fairness']:.4f}")
    print(f"  Execution Time: {elapsed_original:.2f}s")
    print()
    
    # ========================================================================
    # Run Realistic Timing Simulation
    # ========================================================================
    print("-" * 80)
    print("RUNNING REALISTIC TIMING SIMULATION (IEEE 802.11 DCF)")
    print("-" * 80)
    
    start_time = time.time()
    
    sim_realistic = RealisticEngine(
        num_nodes=NUM_NODES,
        packet_prob=PACKET_PROB,
        node_type='BEB',
        duration=REALISTIC_DURATION,
        seed=SEED
    )
    
    metrics_realistic = sim_realistic.run()
    
    elapsed_realistic = time.time() - start_time
    
    print(f"\nRealistic Timing Results:")
    print(f"  Throughput: {metrics_realistic['throughput']:.4f} packets/sec")
    print(f"  Collision Rate: {metrics_realistic['collision_rate']:.4f}")
    print(f"  PDR: {metrics_realistic['pdr']:.4f}")
    print(f"  Fairness: {metrics_realistic['fairness']:.4f}")
    print(f"  Channel Utilization: {metrics_realistic['channel_utilization']:.4f}")
    print(f"  Avg Backoff Time: {metrics_realistic['avg_backoff_time']:.4f}s")
    print(f"  Execution Time: {elapsed_realistic:.2f}s")
    print()
    
    # ========================================================================
    # Comparison Analysis
    # ========================================================================
    print("="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)
    print()
    
    print("Key Differences:")
    print(f"  * Throughput units: Original is 'slots', Realistic is 'packets/sec'")
    print(f"  * Realistic model includes DIFS ({MACTiming.DIFS}us), SIFS ({MACTiming.SIFS}us)")
    print(f"  * Realistic model includes transmission time ({MACTiming.DATA_DURATION}us per packet)")
    print(f"  * Realistic model includes ACK time ({MACTiming.ACK_DURATION}us)")
    print()
    
    print("Relative Performance:")
    print(f"  Collision Rate - Original: {metrics_original['collision_rate']:.4f}, "
          f"Realistic: {metrics_realistic['collision_rate']:.4f}")
    
    collision_diff = abs(metrics_original['collision_rate'] - metrics_realistic['collision_rate'])
    print(f"    -> Difference: {collision_diff:.4f} ({'similar' if collision_diff < 0.1 else 'different'})")
    print()
    
    print(f"  PDR - Original: {metrics_original['pdr']:.4f}, "
          f"Realistic: {metrics_realistic['pdr']:.4f}")
    pdr_diff = abs(metrics_original['pdr'] - metrics_realistic['pdr'])
    print(f"    -> Difference: {pdr_diff:.4f} ({'similar' if pdr_diff < 0.1 else 'different'})")
    print()
    
    print(f"  Fairness - Original: {metrics_original['fairness']:.4f}, "
          f"Realistic: {metrics_realistic['fairness']:.4f}")
    fairness_diff = abs(metrics_original['fairness'] - metrics_realistic['fairness'])
    print(f"    -> Difference: {fairness_diff:.4f} ({'similar' if fairness_diff < 0.1 else 'different'})")
    print()
    
    print("Computational Performance:")
    print(f"  Original: {ORIGINAL_DURATION:,} steps in {elapsed_original:.2f}s "
          f"({ORIGINAL_DURATION/elapsed_original:.0f} steps/sec)")
    print(f"  Realistic: {REALISTIC_DURATION:,} steps in {elapsed_realistic:.2f}s "
          f"({REALISTIC_DURATION/elapsed_realistic:.0f} steps/sec)")
    
    speedup = elapsed_realistic / elapsed_original
    print(f"  -> Realistic is {speedup:.1f}x {'slower' if speedup > 1 else 'faster'} "
          f"(expected due to {REALISTIC_DURATION/ORIGINAL_DURATION:.0f}x more steps)")
    print()
    
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("The realistic timing model provides:")
    print("  + Accurate MAC layer timing (DIFS, SIFS, ACK)")
    print("  + Real-world time domain (microseconds)")
    print("  + Channel utilization metrics")
    print("  + More faithful representation of IEEE 802.11 DCF")
    print()
    print("Trade-off:")
    print("  - Slower execution due to finer time granularity")
    print("  - Requires longer simulation duration for same number of events")
    print()
    print("Both models should show similar *relative* performance trends,")
    print("but the realistic model provides more accurate *absolute* metrics.")
    print()


if __name__ == "__main__":
    run_comparison()
