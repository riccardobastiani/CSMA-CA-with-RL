from experiments import ExperimentRunner
import time

def main():
    runner = ExperimentRunner()
    start_time = time.time()
    
    print("=== Starting Remaining CSMA/CA Simulations with Seeding (10 seeds) ===")
    
    # 1. Load Response (Collision vs Load)
    print("\n[1/3] Running Load Response Experiment...")
    # Using fewer node counts to save time, but keeping high seed count
    runner.run_load_response(packet_probs=[0.1, 0.3, 0.5, 0.7, 0.9], node_counts=[10, 50, 100], seeds=10)
    
    # 2. Retry Scalability
    print("\n[2/3] Running Retry Scalability Experiment...")
    runner.run_retry_scalability(node_counts=[10, 50, 100, 200], seeds=10)
    
    # 3. Retry Heatmap
    print("\n[3/3] Running Retry Heatmap Analysis...")
    # Reduced node counts slightly for speed, but kept seeds=10
    runner.run_retry_heatmap(node_counts=[10, 50, 100, 200], packet_probs=[0.3, 0.5, 0.7, 0.9], seeds=10)

    elapsed = time.time() - start_time
    print(f"\n=== Remaining Simulations Completed in {elapsed:.2f} seconds ===")

if __name__ == "__main__":
    main()
