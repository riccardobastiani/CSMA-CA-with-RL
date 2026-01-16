from simulation_optimized_timing import OptimizedSimulationEngine

print("Running TRUE SPARSE Test (N=10, p=0.01)...")

# 1. BEB
sim_beb = OptimizedSimulationEngine(num_nodes=10, packet_prob=0.001, node_type='BEB', duration=5000000)
metrics_beb = sim_beb.run()

# 2. RL
sim_rl = OptimizedSimulationEngine(num_nodes=10, packet_prob=0.001, node_type='RL', duration=5000000)
metrics_rl = sim_rl.run()

print("\nRESULTS:")
print(f"BEB Throughput: {metrics_beb['throughput']:.4f} pkt/s")
print(f"RL  Throughput: {metrics_rl['throughput']:.4f} pkt/s")

if metrics_beb['throughput'] > metrics_rl['throughput']:
    print("\n✅ CONFIRMED: BEB wins in truly sparse traffic!")
else:
    print("\n❓ RL still wins? (Maybe exploration is very efficient)")


# anche provando con valori estremi BEB non ha mai battuto RL. Il che implica che ci siano dei problemi con uno dei due. 
# BEB dovrebbe outperformare RL in questo scenario, ma non lo fa. 