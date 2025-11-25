from models import Channel, BEBNode, RLNode, BEBRetryNode, RLRetryNode

import random
import numpy as np

class SimulationEngine:
    def __init__(self, num_nodes, packet_prob, node_type='BEB', duration=1000, seed=None, **kwargs):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.duration = duration
        self.channel = Channel()
        self.nodes = []
        self.current_time = 0
        
        # Initialize nodes
        for i in range(num_nodes):
            if node_type == 'BEB':
                self.nodes.append(BEBNode(i, packet_prob))
            elif node_type == 'BEB_RETRY':
                self.nodes.append(BEBRetryNode(i, packet_prob))
            elif node_type == 'RL':
                self.nodes.append(RLNode(i, packet_prob, **kwargs))
            elif node_type == 'RL_RETRY':
                self.nodes.append(RLRetryNode(i, packet_prob, **kwargs))
            else:
                raise ValueError(f"Unknown node type: {node_type}")

        # Metrics
        self.metrics = {
            'throughput': 0,
            'collision_rate': 0,
            'total_generated': 0,
            'total_success': 0,
            'total_collisions': 0,
            'history': [] # For time-series plots
        }
        
        # State counters
        self.successful_slots = 0
        self.collision_slots = 0

    def step(self):
        t = self.current_time
        
        # 1. Traffic Generation
        for node in self.nodes:
            node.generate_packet(t)
            node.check_new_packet()

        # 2. Determine Transmissions
        transmitting_nodes = []
        for node in self.nodes:
            node.decrement_backoff()
            if node.ready_to_transmit():
                transmitting_nodes.append(node)

        # 3. Channel Resolution
        status = self.channel.resolve_slot(transmitting_nodes)
        
        # 4. Feedback & Metrics
        if status == Channel.SUCCESS:
            self.successful_slots += 1
            # Only the successful node gets success feedback
            transmitting_nodes[0].handle_feedback(Channel.SUCCESS)
        elif status == Channel.COLLISION:
            self.collision_slots += 1
            for node in transmitting_nodes:
                node.handle_feedback(Channel.COLLISION)
        
        # 5. Record History (every 100 slots)
        if (t + 1) % 100 == 0:
            window_throughput = (self.successful_slots - self.metrics.get('last_success', 0)) / 100
            window_collision = (self.collision_slots - self.metrics.get('last_collision', 0)) / 100
            self.metrics['history'].append({
                'time': t + 1,
                'throughput': window_throughput,
                'collision_rate': window_collision
            })
            self.metrics['last_success'] = self.successful_slots
            self.metrics['last_collision'] = self.collision_slots
        
        self.current_time += 1
        return status, transmitting_nodes

    def run(self):
        self.successful_slots = 0
        self.collision_slots = 0
        
        for _ in range(self.duration):
            self.step()

        # Final Metrics Calculation
        total_generated = sum(n.total_generated for n in self.nodes)
        total_success = sum(n.total_success for n in self.nodes)
        total_collisions = sum(n.total_collisions for n in self.nodes) # Note: this counts individual node collisions
        total_dropped = sum(n.total_dropped for n in self.nodes)

        self.metrics['throughput'] = self.successful_slots / self.duration
        self.metrics['collision_rate'] = self.collision_slots / self.duration # Slot collision rate
        # Alternatively: total_collisions / (total_success + total_collisions) for packet collision rate
        
        self.metrics['total_generated'] = total_generated
        self.metrics['total_success'] = total_success
        self.metrics['total_collisions'] = total_collisions
        self.metrics['total_dropped'] = total_dropped
        self.metrics['pdr'] = total_success / total_generated if total_generated > 0 else 0
        
        # Jain's Fairness Index
        # (Sum x_i)^2 / (n * Sum x_i^2) where x_i is throughput of node i
        throughputs = [n.total_success / self.duration for n in self.nodes]
        if sum(throughputs) > 0:
            numerator = sum(throughputs) ** 2
            denominator = len(self.nodes) * sum(x**2 for x in throughputs)
            self.metrics['fairness'] = numerator / denominator if denominator > 0 else 0
        else:
            self.metrics['fairness'] = 0

        return self.metrics
