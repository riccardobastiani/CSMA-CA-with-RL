import unittest
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Packet, Channel, Node, BEBNode, RLNode

class TestModels(unittest.TestCase):
    def test_channel_logic(self):
        channel = Channel()
        self.assertEqual(channel.resolve_slot([]), Channel.IDLE)
        self.assertEqual(channel.resolve_slot([1]), Channel.SUCCESS)
        self.assertEqual(channel.resolve_slot([1, 2]), Channel.COLLISION)

    def test_beb_node_backoff(self):
        node = BEBNode(1, 1.0) # Always generate packet
        node.generate_packet(0)
        node.check_new_packet()
        
        initial_cw = node.cw
        self.assertEqual(initial_cw, 4)
        
        # Simulate Collision
        node.handle_feedback(Channel.COLLISION)
        self.assertEqual(node.cw, 8)
        
        # Simulate Success
        node.handle_feedback(Channel.SUCCESS)
        self.assertEqual(node.cw, 4)

    def test_rl_node_learning(self):
        node = RLNode(1, 1.0, alpha=0.5, gamma=0.9, epsilon=0.0) # No exploration for deterministic test
        node.generate_packet(0)
        node.check_new_packet()
        
        # Force action index 0 (CW=8)
        node.last_state = 0
        node.last_action_idx = 0
        initial_q = node.q_table[0][0]
        
        # Simulate Success
        node.handle_feedback(Channel.SUCCESS)
        
        # Q should increase: Q = Q + alpha * (Reward + gamma*maxQ' - Q)
        # Reward = 10. maxQ' = 0 (terminal). Q = 0 + 0.5 * (10 + 0 - 0) = 5.0
        self.assertEqual(node.q_table[0][0], 5.0)

if __name__ == '__main__':
    unittest.main()
