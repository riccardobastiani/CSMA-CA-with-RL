import unittest
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import SimulationEngine

class TestSimulation(unittest.TestCase):
    def test_beb_simulation_run(self):
        # Run a short simulation with BEB nodes
        sim = SimulationEngine(num_nodes=5, packet_prob=0.5, node_type='BEB', duration=100)
        metrics = sim.run()
        
        self.assertIn('throughput', metrics)
        self.assertIn('collision_rate', metrics)
        self.assertGreaterEqual(metrics['throughput'], 0)
        self.assertLessEqual(metrics['throughput'], 1)
        print(f"BEB Metrics: {metrics}")

    def test_rl_simulation_run(self):
        # Run a short simulation with RL nodes
        sim = SimulationEngine(num_nodes=5, packet_prob=0.5, node_type='RL', duration=100, alpha=0.1, epsilon=0.1)
        metrics = sim.run()
        
        self.assertIn('throughput', metrics)
        self.assertIn('collision_rate', metrics)
        print(f"RL Metrics: {metrics}")

if __name__ == '__main__':
    unittest.main()
