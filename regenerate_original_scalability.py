from experiments import ExperimentRunner

print("Regenerating Original (Abstract) Scalability Plot...")
runner = ExperimentRunner()
runner.run_scalability(node_counts=[10, 50, 100, 200], packet_prob=0.5, duration=5000, seeds=10)
print("Done!")
