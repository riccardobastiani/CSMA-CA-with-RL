"""Quick test of realistic timing simulation"""

from simulation_realistic_timing import SimulationEngine as RealisticEngine
from models_realistic_timing import MACTiming

print("Testing Realistic Timing Simulation")
print("="*50)
print()

# Very short test - 100ms
sim = RealisticEngine(
    num_nodes=5,
    packet_prob=0.1,
    node_type='BEB',
    duration=100000,  # 100ms = 100,000 microseconds
    seed=42
)

print(f"MAC Timing Constants:")
print(f"  DIFS: {MACTiming.DIFS} us")
print(f"  SIFS: {MACTiming.SIFS} us")
print(f"  SlotTime: {MACTiming.SLOT_TIME} us")
print(f"  Data TX: {MACTiming.DATA_DURATION} us")
print(f"  ACK: {MACTiming.ACK_DURATION} us")
print()

metrics = sim.run()

print()
print("Final Metrics:")
print(f"  Throughput: {metrics['throughput']:.2f} packets/sec")
print(f"  Collision Rate: {metrics['collision_rate']:.4f}")
print(f"  PDR: {metrics['pdr']:.4f}")
print(f"  Channel Util: {metrics['channel_utilization']:.4f}")
print()
print("Test Complete!")
