"""
Optimized Realistic Timing Simulation

Instead of 1µs granularity, this uses SlotTime (20µs) as the basic unit.
This reduces iterations by 20x while maintaining accurate MAC timing.

Key Changes:
- Time advances in 20µs chunks (SlotTime)
- DIFS = 2.5 slots, SIFS = 0.5 slots
- Transmission duration = 600 slots (12,000µs / 20µs)
"""

from models_realistic_timing import (
    Channel, BEBNode, RLNode, BEBRetryNode, RLRetryNode, MACTiming
)
import random
import numpy as np


class OptimizedSimulationEngine:
    """
    Time-aware simulation with SlotTime (20µs) granularity.
    
    This is 20x faster than 1µs simulation while maintaining
    accurate MAC layer timing.
    """
    
    def __init__(self, num_nodes, packet_prob, node_type='BEB', 
                 duration=50000, seed=None, **kwargs):
        """
        Args:
            num_nodes: Number of nodes
            packet_prob: Packet generation probability (per slot)
            node_type: 'BEB', 'BEB_RETRY', 'RL', 'RL_RETRY'
            duration: Simulation duration in SLOT_TIME units (default 50k = 1 second)
            seed: Random seed
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.duration = duration
        self.channel = Channel()
        self.nodes = []
        self.current_slot = 0
        self.slot_duration_us = MACTiming.SLOT_TIME  # 20µs
        
        # Packet generation probability
        self.packet_prob = packet_prob
        
        # MAC timing in slots (20µs each)
        self.DIFS_SLOTS = int(MACTiming.DIFS / MACTiming.SLOT_TIME)  # 50/20 = 2.5 ≈ 3
        self.SIFS_SLOTS = max(1, int(MACTiming.SIFS / MACTiming.SLOT_TIME))  # 10/20 = 0.5 ≈ 1
        self.TX_SLOTS = int(MACTiming.DATA_DURATION / MACTiming.SLOT_TIME)  # 12000/20 = 600
        
        # Initialize nodes
        for i in range(num_nodes):
            if node_type == 'BEB':
                self.nodes.append(BEBNode(i, packet_prob, time_granularity_us=self.slot_duration_us))
            elif node_type == 'BEB_RETRY':
                self.nodes.append(BEBRetryNode(i, packet_prob, time_granularity_us=self.slot_duration_us))
            elif node_type == 'RL':
                self.nodes.append(RLNode(i, packet_prob, time_granularity_us=self.slot_duration_us, **kwargs))
            elif node_type == 'RL_RETRY':
                self.nodes.append(RLRetryNode(i, packet_prob, time_granularity_us=self.slot_duration_us, **kwargs))
            else:
                raise ValueError(f"Unknown node type: {node_type}")
        
        # Metrics
        self.metrics = {
            'throughput': 0,
            'collision_rate': 0,
            'total_generated': 0,
            'total_success': 0,
            'total_collisions': 0,
            'total_dropped': 0,
            'channel_utilization': 0,
            'history': []
        }
        
        # State
        self.successful_transmissions = 0
        self.collision_transmissions = 0
        self.busy_slots = 0
        self.transmitting_nodes = []
        self.transmission_end_slot = None
        
        print(f"Optimized Simulation Engine Initialized")
        print(f"  Time granularity: {self.slot_duration_us}µs per slot")
        print(f"  DIFS: {self.DIFS_SLOTS} slots ({self.DIFS_SLOTS * self.slot_duration_us}µs)")
        print(f"  SIFS: {self.SIFS_SLOTS} slots ({self.SIFS_SLOTS * self.slot_duration_us}µs)")
        print(f"  TX Duration: {self.TX_SLOTS} slots ({self.TX_SLOTS * self.slot_duration_us}µs)")
    
    def step(self):
        """Single time step (1 SlotTime = 20µs)"""
        t = self.current_slot
        current_time_us = t * self.slot_duration_us
        
        # Check if transmission has ended
        if self.transmission_end_slot is not None and t >= self.transmission_end_slot:
            # Transmission just ended
            if len(self.transmitting_nodes) == 1:
                # Success
                self.successful_transmissions += 1
                for node in self.transmitting_nodes:
                    node.handle_feedback(Channel.SUCCESS, current_time_us)
            else:
                # Collision
                self.collision_transmissions += 1
                for node in self.transmitting_nodes:
                    node.handle_feedback(Channel.COLLISION, current_time_us)
            
            self.transmitting_nodes = []
            self.transmission_end_slot = None
        
        # Check if channel is busy
        channel_busy = self.transmission_end_slot is not None and t < self.transmission_end_slot
        
        if channel_busy:
            self.busy_slots += 1
        
        # Packet generation (every slot)
        for node in self.nodes:
            node.generate_packet(current_time_us)
            node.check_new_packet(current_time_us)
        
        # Update all nodes
        ready_nodes = []
        for node in self.nodes:
            if node.update(current_time_us, channel_busy):
                ready_nodes.append(node)
        
        # Start transmissions if nodes are ready and channel is free
        if ready_nodes and not channel_busy:
            self.transmitting_nodes = ready_nodes
            self.transmission_end_slot = t + self.TX_SLOTS
            
            for node in ready_nodes:
                node.start_transmission(current_time_us)
        
        # Record history every 1000 slots (~20ms)
        if (t + 1) % 1000 == 0:
            self._record_history(t)
        
        self.current_slot += 1
        return len(self.transmitting_nodes) > 0
    
    def _record_history(self, slot):
        """Record periodic metrics"""
        window_slots = 1000
        
        last_success = self.metrics.get('_last_success_count', 0)
        last_collision = self.metrics.get('_last_collision_count', 0)
        
        window_success = self.successful_transmissions - last_success
        window_collision = self.collision_transmissions - last_collision
        
        # Throughput in packets/sec
        window_duration_sec = (window_slots * self.slot_duration_us) / 1e6
        window_throughput = window_success / window_duration_sec if window_duration_sec > 0 else 0
        
        # Collision rate
        total_attempts = window_success + window_collision
        window_collision_rate = window_collision / total_attempts if total_attempts > 0 else 0
        
        self.metrics['history'].append({
            'time': (slot * self.slot_duration_us) / 1000,  # Convert to ms
            'throughput': window_throughput,
            'collision_rate': window_collision_rate
        })
        
        self.metrics['_last_success_count'] = self.successful_transmissions
        self.metrics['_last_collision_count'] = self.collision_transmissions
    
    def run(self):
        """Run full simulation"""
        print(f"\nStarting optimized simulation for {self.duration:,} slots "
              f"({self.duration * self.slot_duration_us / 1e6:.2f} seconds)...")
        
        milestone = max(1, self.duration // 10)
        for step in range(self.duration):
            self.step()
            
            if (step + 1) % milestone == 0:
                progress = ((step + 1) / self.duration) * 100
                print(f"Progress: {progress:.0f}%")
        
        self._calculate_final_metrics()
        return self.metrics
    
    def _calculate_final_metrics(self):
        """Calculate final aggregate metrics"""
        total_generated = sum(n.total_generated for n in self.nodes)
        total_success = sum(n.total_success for n in self.nodes)
        total_collisions = sum(n.total_collisions for n in self.nodes)
        total_dropped = sum(n.total_dropped for n in self.nodes)
        
        duration_sec = (self.duration * self.slot_duration_us) / 1e6
        
        self.metrics['throughput'] = self.successful_transmissions / duration_sec
        
        total_events = self.successful_transmissions + self.collision_transmissions
        self.metrics['collision_rate'] = (
            self.collision_transmissions / total_events if total_events > 0 else 0
        )
        
        self.metrics['channel_utilization'] = self.busy_slots / self.duration
        self.metrics['pdr'] = total_success / total_generated if total_generated > 0 else 0
        
        self.metrics['total_generated'] = total_generated
        self.metrics['total_success'] = total_success
        self.metrics['total_collisions'] = total_collisions
        self.metrics['total_dropped'] = total_dropped
        
        # Fairness
        throughputs = [n.total_success / duration_sec for n in self.nodes]
        if sum(throughputs) > 0:
            numerator = sum(throughputs) ** 2
            denominator = len(self.nodes) * sum(x**2 for x in throughputs)
            self.metrics['fairness'] = numerator / denominator if denominator > 0 else 0
        else:
            self.metrics['fairness'] = 0
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        print(f"Simulated Time: {duration_sec:.2f} seconds")
        print(f"Successful Transmissions: {self.successful_transmissions}")
        print(f"Collision Events: {self.collision_transmissions}")
        print(f"Throughput: {self.metrics['throughput']:.2f} packets/sec")
        print(f"Collision Rate: {self.metrics['collision_rate']:.4f}")
        print(f"Channel Utilization: {self.metrics['channel_utilization']:.4f}")
        print(f"PDR: {self.metrics['pdr']:.4f}")
        print(f"Fairness: {self.metrics['fairness']:.4f}")
        print("="*60)
