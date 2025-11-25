from experiments import ExperimentRunner
import time

def main():
    runner = ExperimentRunner()
    start_time = time.time()
    
    print("=== Starting CSMA/CA Simulations with Seeding (10 seeds) ===")
    
    # 1. Scalability (Baseline BEB vs RL)
    print("\n[1/6] Running Scalability Experiment...")
    runner.run_scalability(node_counts=[10, 50, 100, 200], seeds=10)
    
    # 2. Reward Comparison (Tuning RL)
    print("\n[2/6] Running Reward Comparison Experiment...")
    runner.run_reward_comparison(seeds=10)
    
    # 3. Learning Stability (Convergence)
    print("\n[3/6] Running Learning Stability Experiment...")
    runner.run_learning_stability(seeds=10)
    
    # 4. Retry Comparison (Impact of Retry Limits)
    print("\n[4/6] Running Retry Comparison Experiment...")
    runner.run_retry_comparison(seeds=10)
    
    # 5. Optimized Scalability (Best RL vs BEB)
    print("\n[5/6] Running Optimized Scalability Experiment...")
    runner.run_optimized_scalability(node_counts=[10, 50, 100, 200], seeds=10)
    
    # 6. Epsilon Comparison
    print("\n[6/6] Running Epsilon Comparison Experiment...")
    runner.run_epsilon_comparison(seeds=10)

    elapsed = time.time() - start_time
    print(f"\n=== All Simulations Completed in {elapsed:.2f} seconds ===")

if __name__ == "__main__":
    main()
