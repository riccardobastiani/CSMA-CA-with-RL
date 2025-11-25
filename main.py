import argparse
import os
from experiments import ExperimentRunner


'''
Scalability Experiment: It tests with 5, 10, 20, and 30 nodes by default.
This is defined in line 8: default=[5, 10, 20, 30].
This experiment varies the number of nodes to see how performance degrades as the network gets denser.
Load Response Experiment: It uses a fixed number of 10 nodes (default in experiments.py)
but varies the traffic load.
This experiment tests how the network handles different levels of traffic 
(Packet Generation Probability: 0.1, 0.3, 0.5, 0.7, 0.9).
'''

def main():
    parser = argparse.ArgumentParser(description="CSMA/CA RL Simulation")
    parser.add_argument('--experiment', type=str, choices=['scalability', 'load', 'stability', 'reward', 'retry', 'retry_viz', 'epsilon', 'optimized_scalability', 'all'], default='all', help='Experiment to run')
    parser.add_argument('--nodes', type=int, nargs='+', default=[10, 50, 100, 200, 500], help='List of node counts for scalability')
    parser.add_argument('--probs', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9], help='List of packet probs for load response')
    parser.add_argument('--duration', type=int, default=10000, help='Simulation duration in slots')
    parser.add_argument('--seeds', type=int, default=10, help='Number of repetitions per scenario')
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
        
    runner = ExperimentRunner(output_dir='results')
    
    if args.experiment in ['scalability', 'all']:
        runner.run_scalability(node_counts=args.nodes, duration=args.duration, seeds=args.seeds)
        
    if args.experiment in ['load', 'all']:
        runner.run_load_response(packet_probs=args.probs, node_counts=args.nodes, duration=args.duration, seeds=args.seeds)

    if args.experiment in ['stability', 'all']:
        runner.run_learning_stability(duration=5000)

    if args.experiment in ['reward', 'all']:
        runner.run_reward_comparison(duration=args.duration, seeds=args.seeds)

    if args.experiment in ['retry']:
        runner.run_retry_comparison(duration=args.duration, seeds=args.seeds)

    if args.experiment in ['retry_viz']:
        print("Running all retry visualizations...")
        runner.run_retry_scalability(node_counts=args.nodes, seeds=args.seeds)
        runner.run_retry_convergence()
        runner.run_retry_heatmap(seeds=args.seeds)

    if args.experiment in ['epsilon']:
        runner.run_epsilon_comparison(seeds=args.seeds)

    if args.experiment in ['optimized_scalability']:
        runner.run_optimized_scalability(node_counts=args.nodes, seeds=args.seeds)

if __name__ == '__main__':
    main()
