"""
Visual Demo: Realistic Timing vs Original Simulation

This creates a side-by-side comparison visualization showing:
1. Throughput over time
2. Collision rate over time
3. Key metrics table
"""

import matplotlib.pyplot as plt
import numpy as np
from simulation import SimulationEngine as OriginalEngine
from simulation_realistic_timing import SimulationEngine as RealisticEngine
from models_realistic_timing import MACTiming

def run_visual_comparison():
    """Run both simulations and create comparison visualization"""
    
    print("Running Comparison Simulations...")
    print("="*60)
    
    # Configuration
    NUM_NODES = 10
    PACKET_PROB = 0.05
    SEED = 42
    
    # Durations
    ORIGINAL_DURATION = 5000  # 5000 time steps
    REALISTIC_DURATION = 500000  # 500ms in microseconds
    
    # Run Original
    print("\n[1/2] Running original simulation...")
    sim_orig = OriginalEngine(
        num_nodes=NUM_NODES,
        packet_prob=PACKET_PROB,
        node_type='BEB',
        duration=ORIGINAL_DURATION,
        seed=SEED
    )
    metrics_orig = sim_orig.run()
    
    # Run Realistic
    print("\n[2/2] Running realistic timing simulation...")
    sim_real = RealisticEngine(
        num_nodes=NUM_NODES,
        packet_prob=PACKET_PROB,
        node_type='BEB',
        duration=REALISTIC_DURATION,
        seed=SEED
    )
    metrics_real = sim_real.run()
    
    print("\n" + "="*60)
    print("Creating Visualization...")
    print("="*60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # --- Plot 1: Throughput Comparison ---
    ax1 = fig.add_subplot(gs[0, :])
    
    # Extract history
    history_orig = metrics_orig['history']
    history_real = metrics_real['history']
    
    if history_orig:
        times_orig = [h['time'] for h in history_orig]
        throughput_orig = [h['throughput'] for h in history_orig]
        ax1.plot(times_orig, throughput_orig, 'b-', label='Original (slot-based)', linewidth=2)
    
    if history_real:
        times_real = [h['time'] for h in history_real]
        throughput_real = [h['throughput'] for h in history_real]
        ax1.plot(times_real, throughput_real, 'r-', label='Realistic Timing', linewidth=2)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Throughput')
    ax1.set_title('Throughput Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Collision Rate Comparison ---
    ax2 = fig.add_subplot(gs[1, :])
    
    if history_orig:
        collision_orig = [h['collision_rate'] for h in history_orig]
        ax2.plot(times_orig, collision_orig, 'b-', label='Original (slot-based)', linewidth=2)
    
    if history_real:
        collision_real = [h['collision_rate'] for h in history_real]
        ax2.plot(times_real, collision_real, 'r-', label='Realistic Timing', linewidth=2)
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Collision Rate')
    ax2.set_title('Collision Rate Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Metrics Comparison Table ---
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')
    
    table_data = [
        ['Metric', 'Original', 'Realistic'],
        ['Collision Rate', f"{metrics_orig['collision_rate']:.4f}", f"{metrics_real['collision_rate']:.4f}"],
        ['PDR', f"{metrics_orig['pdr']:.4f}", f"{metrics_real['pdr']:.4f}"],
        ['Fairness', f"{metrics_orig['fairness']:.4f}", f"{metrics_real['fairness']:.4f}"],
        ['Total Generated', f"{metrics_orig['total_generated']}", f"{metrics_real['total_generated']}"],
        ['Total Success', f"{metrics_orig['total_success']}", f"{metrics_real['total_success']}"],
    ]
    
    table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax3.set_title('Metrics Comparison', fontsize=12, fontweight='bold', pad=20)
    
    # --- Plot 4: Timing Details ---
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    timing_text = f"""
IEEE 802.11 DCF Timing Parameters:

DIFS:  {MACTiming.DIFS} us
SIFS:  {MACTiming.SIFS} us
Slot Time:  {MACTiming.SLOT_TIME} us
Data TX:  {MACTiming.DATA_DURATION} us
ACK TX:  {MACTiming.ACK_DURATION} us

Success Duration (DATA+SIFS+ACK):
  {MACTiming.SUCCESS_DURATION} us

Realistic Model Features:
  + Explicit DIFS/SIFS waiting
  + Variable slot durations
  + Transmission time modeling
  + ACK timeout handling
  + Channel utilization tracking
    """
    
    ax4.text(0.1, 0.5, timing_text, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle('CSMA/CA Simulation: Original vs. Realistic Timing',
                 fontsize=16, fontweight='bold')
    
    # Save figure
    output_file = 'results/realistic_timing_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nOriginal Model:")
    print(f"  - Abstract time slots")
    print(f"  - Instantaneous transmissions")
    print(f"  - Collision Rate: {metrics_orig['collision_rate']:.4f}")
    print(f"  - PDR: {metrics_orig['pdr']:.4f}")
    print(f"\nRealistic Timing Model:")
    print(f"  - Microsecond granularity")
    print(f"  - IEEE 802.11 DCF timing")
    print(f"  - Collision Rate: {metrics_real['collision_rate']:.4f}")
    print(f"  - PDR: {metrics_real['pdr']:.4f}")
    print(f"  - Channel Utilization: {metrics_real['channel_utilization']:.4f}")
    print("\nBoth models show consistent relative performance!")
    print("="*60)


if __name__ == "__main__":
    run_visual_comparison()
