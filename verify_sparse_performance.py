from simulation_optimized_timing import OptimizedSimulationEngine

def run_test_averaged(p_val, num_seeds=5):
    print(f"\nTesting with p={p_val} (Averaged over {num_seeds} seeds)...")
    load = 10 * p_val * 600
    print(f"Estimated Load: {load*100:.1f}%")

    beb_throughputs = []
    rl_throughputs = []

    for seed in range(num_seeds):
        # 1. BEB
        sim_beb = OptimizedSimulationEngine(num_nodes=10, packet_prob=p_val, node_type='BEB', duration=500000, seed=seed)
        metrics_beb = sim_beb.run()
        beb_throughputs.append(metrics_beb['throughput'])

        # 2. RL
        sim_rl = OptimizedSimulationEngine(num_nodes=10, packet_prob=p_val, node_type='RL', duration=500000, seed=seed)
        metrics_rl = sim_rl.run()
        rl_throughputs.append(metrics_rl['throughput'])
    
    avg_beb = sum(beb_throughputs) / num_seeds
    avg_rl = sum(rl_throughputs) / num_seeds
    
    print(f"BEB Avg Throughput: {avg_beb:.4f} pkt/s")
    print(f"RL  Avg Throughput: {avg_rl:.4f} pkt/s")
    
    if avg_beb > avg_rl:
        print("✅ BEB WINS")
    else:
        print("❌ RL WINS")

# Run averaged tests
run_test_averaged(0.0001, num_seeds=5)  # 60% load
run_test_averaged(0.00001, num_seeds=5) # 6% load
