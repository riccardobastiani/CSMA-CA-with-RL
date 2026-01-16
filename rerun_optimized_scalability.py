"""
Re-run Optimized Scalability Experiment with BUG FIX

This script re-runs the scalability experiment using the FIXED
OptimizedSimulationEngine to replace the buggy results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation_optimized_timing import OptimizedSimulationEngine
import time

print("="*80)
print("RE-RUNNING OPTIMIZED SCALABILITY EXPERIMENT (BUG FIX)")
print("="*80)
print()
print("This will replace the buggy results in:")
print("  - results/optimized_scalability_raw.csv")
print("  - results/optimized_scalability_averaged.csv")
print("  - results/optimized_scalability.png")
print()

# Configuration (same as original experiment)
NODE_COUNTS = [10, 50, 100, 200]
PACKET_PROB = 0.05
DURATION = 50000  # 50k slots = 1 second of simulated time
NUM_SEEDS = 10
SEED_START = 42

configs = [
    {'name': 'BEB', 'type': 'BEB', 'kwargs': {}},
    {'name': 'RL (Standard eps=0.1)', 'type': 'RL', 'kwargs': {'epsilon': 0.1}},
    {'name': 'RL (Optimized)', 'type': 'RL', 
     'kwargs': {'epsilon': 1.0, 'epsilon_decay': 0.001, 'epsilon_min': 0.01}},
]

print(f"Configuration:")
print(f"  Node counts: {NODE_COUNTS}")
print(f"  Packet probability: {PACKET_PROB}")
print(f"  Duration: {DURATION} slots (1 second simulated time)")
print(f"  Seeds: {NUM_SEEDS} (starting from {SEED_START})")
print()

# Storage
raw_data = []
averaged_data = []
results = {cfg['name']: {
    'throughput': [], 
    'pdr': [], 
    'collision': [], 
    'efficiency': []
} for cfg in configs}

total_runs = len(NODE_COUNTS) * len(configs) * NUM_SEEDS
current_run = 0
start_time = time.time()

# Run experiments
for N in NODE_COUNTS:
    print()
    print("="*80)
    print(f"NODE COUNT: {N}")
    print("="*80)
    
    for cfg in configs:
        print(f"\n  Configuration: {cfg['name']}")
        print(f"  " + "-"*70)
        
        throughputs = []
        pdrs = []
        collisions = []
        
        for i, seed in enumerate(range(SEED_START, SEED_START + NUM_SEEDS)):
            current_run += 1
            progress = (current_run / total_runs) * 100
            
            print(f"    [{current_run}/{total_runs}] Seed {seed}...", end=" ")
            
            # Run simulation with FIXED OptimizedSimulationEngine
            sim = OptimizedSimulationEngine(
                num_nodes=N,
                packet_prob=PACKET_PROB,
                node_type=cfg['type'],
                duration=DURATION,
                seed=seed,
                **cfg['kwargs']
            )
            
            # Suppress simulation output
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            metrics = sim.run()
            
            sys.stdout = old_stdout
            
            # Store results
            throughputs.append(metrics['throughput'])
            pdrs.append(metrics['pdr'])
            collisions.append(metrics['collision_rate'])
            
            efficiency = metrics['throughput'] * (1 - metrics['collision_rate'])
            
            raw_data.append({
                'Experiment': 'OptimizedScalability',
                'Nodes': N,
                'Config': cfg['name'],
                'Seed': seed,
                'Throughput': metrics['throughput'],
                'PDR': metrics['pdr'],
                'CollisionRate': metrics['collision_rate'],
                'Efficiency': efficiency
            })
            
            print(f"Thr={metrics['throughput']:.4f}, Coll={metrics['collision_rate']:.4f}, Eff={efficiency:.4f}")
        
        # Calculate averages
        avg_throughput = np.mean(throughputs)
        avg_pdr = np.mean(pdrs)
        avg_collision = np.mean(collisions)
        avg_efficiency = avg_throughput * (1 - avg_collision)
        
        results[cfg['name']]['throughput'].append(avg_throughput)
        results[cfg['name']]['pdr'].append(avg_pdr)
        results[cfg['name']]['collision'].append(avg_collision)
        results[cfg['name']]['efficiency'].append(avg_efficiency)
        
        averaged_data.append({
            'Experiment': 'OptimizedScalability',
            'Nodes': N,
            'Config': cfg['name'],
            'Avg_Throughput': avg_throughput,
            'Avg_PDR': avg_pdr,
            'Avg_CollisionRate': avg_collision,
            'Avg_Efficiency': avg_efficiency
        })
        
        print(f"\n  AVERAGE: Thr={avg_throughput:.4f}, PDR={avg_pdr:.4f}, " +
              f"Coll={avg_collision:.4f}, Eff={avg_efficiency:.4f}")

elapsed = time.time() - start_time
print()
print("="*80)
print(f"EXPERIMENT COMPLETE! ({elapsed:.1f} seconds)")
print("="*80)
print()

# Save to CSV
print("Saving results to CSV...")
df_raw = pd.DataFrame(raw_data)
df_averaged = pd.DataFrame(averaged_data)

df_raw.to_csv('results/optimized_scalability_raw.csv', index=False)
df_averaged.to_csv('results/optimized_scalability_averaged.csv', index=False)
print("  ✓ results/optimized_scalability_raw.csv")
print("  ✓ results/optimized_scalability_averaged.csv")
print()

# Create visualization
print("Creating visualization...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: Throughput
ax1 = fig.add_subplot(gs[0, 0])
for cfg in configs:
    marker = 'o' if 'BEB' in cfg['name'] else ('s' if 'Standard' in cfg['name'] else '^')
    ax1.plot(NODE_COUNTS, results[cfg['name']]['throughput'], 
             marker=marker, label=cfg['name'], linewidth=2, markersize=8)
ax1.set_xlabel('Number of Nodes', fontsize=12)
ax1.set_ylabel('Throughput (packets/sec)', fontsize=12)
ax1.set_title('Scalability: Throughput vs Network Density', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: PDR
ax2 = fig.add_subplot(gs[0, 1])
for cfg in configs:
    marker = 'o' if 'BEB' in cfg['name'] else ('s' if 'Standard' in cfg['name'] else '^')
    ax2.plot(NODE_COUNTS, results[cfg['name']]['pdr'], 
             marker=marker, label=cfg['name'], linewidth=2, markersize=8)
ax2.set_xlabel('Number of Nodes', fontsize=12)
ax2.set_ylabel('Packet Delivery Ratio (PDR)', fontsize=12)
ax2.set_title('Scalability: PDR vs Network Density', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Collision Rate
ax3 = fig.add_subplot(gs[1, 0])
for cfg in configs:
    marker = 'o' if 'BEB' in cfg['name'] else ('s' if 'Standard' in cfg['name'] else '^')
    ax3.plot(NODE_COUNTS, results[cfg['name']]['collision'], 
             marker=marker, label=cfg['name'], linewidth=2, markersize=8)
ax3.set_xlabel('Number of Nodes', fontsize=12)
ax3.set_ylabel('Collision Rate', fontsize=12)
ax3.set_title('Scalability: Collision Rate vs Network Density', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Efficiency (Throughput × (1 - Collision Rate))
ax4 = fig.add_subplot(gs[1, 1])
for cfg in configs:
    marker = 'o' if 'BEB' in cfg['name'] else ('s' if 'Standard' in cfg['name'] else '^')
    ax4.plot(NODE_COUNTS, results[cfg['name']]['efficiency'], 
             marker=marker, label=cfg['name'], linewidth=2, markersize=8)
ax4.set_xlabel('Number of Nodes', fontsize=12)
ax4.set_ylabel('Efficiency Score', fontsize=12)
ax4.set_title('Scalability: Efficiency vs Network Density', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

fig.suptitle('Optimized Scalability Comparison (FIXED - With Realistic MAC Timing)', 
             fontsize=16, fontweight='bold')

plt.savefig('results/optimized_scalability.png', dpi=150, bbox_inches='tight')
print("  ✓ results/optimized_scalability.png")
print()

# Print summary comparison
print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print()

for N in NODE_COUNTS:
    print(f"Network Size: {N} nodes")
    print("-"*80)
    
    idx = NODE_COUNTS.index(N)
    for cfg in configs:
        thr = results[cfg['name']]['throughput'][idx]
        coll = results[cfg['name']]['collision'][idx]
        pdr = results[cfg['name']]['pdr'][idx]
        eff = results[cfg['name']]['efficiency'][idx]
        
        print(f"  {cfg['name']:<30} Thr: {thr:.4f}  Coll: {coll:.4f}  PDR: {pdr:.4f}  Eff: {eff:.4f}")
    
    # Show RL vs BEB comparison
    beb_thr = results['BEB']['throughput'][idx]
    rl_std_thr = results['RL (Standard eps=0.1)']['throughput'][idx]
    rl_opt_thr = results['RL (Optimized)']['throughput'][idx]
    
    print(f"\n  RL Standard vs BEB: {(rl_std_thr/beb_thr - 1)*100:+.1f}%")
    print(f"  RL Optimized vs BEB: {(rl_opt_thr/beb_thr - 1)*100:+.1f}%")
    print()

print("="*80)
print("✓ EXPERIMENT COMPLETE!")
print("="*80)
print()
print("The buggy results have been replaced with corrected data.")
print("You can now use these results in your report/analysis.")
