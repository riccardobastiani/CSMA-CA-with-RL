# this is the time used 

from models_realistic_timing import (
    Channel, BEBNode, RLNode, BEBRetryNode, RLRetryNode, MACTiming
)
import random
import numpy as np


class SimulationEngine:
    """
    Time-aware simulation engine with µs-level granularity.
    
    This hybrid approach maintains a time loop but accounts for realistic
    MAC layer timing (DIFS, SIFS, ACK, transmission durations).
    """
    
    def __init__(self, num_nodes, packet_prob, node_type='BEB', 
                 duration=1000000, seed=None, **kwargs):
        """
        Args:
            num_nodes: Number of nodes in the network
            packet_prob: Packet generation probability (per µs)
            node_type: 'BEB', 'BEB_RETRY', 'RL', 'RL_RETRY'
            duration: Simulation duration in microseconds (default 1s = 1M µs)
            seed: Random seed for reproducibility
            **kwargs: Additional parameters for RL nodes (alpha, gamma, etc.)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.duration = duration
        self.channel = Channel()
        self.nodes = []
        self.current_time = 0
        
        # Adjust packet_prob for µs-level simulation
        # Original slot was abstract; now we generate per µs
        # To keep same average rate: prob_per_µs = prob_per_slot / slot_duration
        # We'll generate packets every SLOT_TIME µs with the given probability
        self.packet_prob = packet_prob
        self.packet_check_interval = MACTiming.SLOT_TIME  # Check every 20µs
        
        
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
            'total_dropped': 0,
            'channel_utilization': 0,
            'history': []
        }
        
        # State counters
        self.successful_transmissions = 0
        self.collision_transmissions = 0
        self.busy_time = 0  # Time channel was busy (µs)
        
        # Track nodes currently transmitting
        self.transmitting_nodes = []
    
    def step(self):
        """
        Single time step (1 µs advancement).
        Returns: (channel_status, transmitting_nodes)
        """
        t = self.current_time
        channel_busy = self.channel.is_busy(t)
        
        # 1. Check if any ongoing transmission has ended
        ended, final_status = self.channel.check_transmission_end(t)
        if ended:
            
            if final_status == Channel.SUCCESS:
                self.successful_transmissions += 1
                for node in self.transmitting_nodes:
                    node.handle_feedback(Channel.SUCCESS, t)
            elif final_status == Channel.COLLISION:
                self.collision_transmissions += 1
                for node in self.transmitting_nodes:
                    node.handle_feedback(Channel.COLLISION, t)
            
            self.transmitting_nodes = []
            channel_busy = False  # Channel now free
        
        # 2. Packet generation (check every SLOT_TIME µs)
        if t % self.packet_check_interval == 0:
            for node in self.nodes:
                node.generate_packet(t)
                node.check_new_packet(t)
        
        # 3. Update all nodes (state machines, backoff counting)
        ready_nodes = []
        for node in self.nodes:
            if node.update(t, channel_busy):
                ready_nodes.append(node)
        
        # 4. If any nodes are ready to transmit and channel is free, start transmission
        if ready_nodes and not channel_busy:

            status, collision = self.channel.start_transmissions(ready_nodes, t)
            self.transmitting_nodes = ready_nodes
            
            
            for node in ready_nodes:
                node.start_transmission(t)
            
            
            if status in [Channel.TRANSMITTING, Channel.COLLISION]:
                self.busy_time += MACTiming.DATA_DURATION
        else:
            status = self.channel.status
        
        # 5. every 100ms = 100,000 µs
        if (t + 1) % 100000 == 0:
            self._record_history(t)
        
        self.current_time += 1
        return status, self.transmitting_nodes
    
    def _record_history(self, t):
        """Record periodic metrics"""
        window_duration = 100000  # 100ms in µs
        
        
        last_success = self.metrics.get('_last_success_count', 0)
        last_collision = self.metrics.get('_last_collision_count', 0)
        
        window_success = self.successful_transmissions - last_success
        window_collision = self.collision_transmissions - last_collision
        
        # Throughput = successful transmissions per second
        # Each success = 1 packet transmitted in ~12ms
        # So throughput is packets/sec in this window
        window_throughput = (window_success / window_duration) * 1e6  # packets/sec
        
        # Collision rate = collisions / (successes + collisions)
        total_attempts = window_success + window_collision
        window_collision_rate = window_collision / total_attempts if total_attempts > 0 else 0
        
        self.metrics['history'].append({
            'time': t / 1000,  # Convert to ms for easier reading
            'throughput': window_throughput,
            'collision_rate': window_collision_rate
        })
        
        self.metrics['_last_success_count'] = self.successful_transmissions
        self.metrics['_last_collision_count'] = self.collision_transmissions
    
    def run(self):
        """
        Run the full simulation.
        Returns: metrics dictionary
        """
        print(f"Starting simulation for {self.duration / 1e6:.2f} seconds...")
        print(f"Time granularity: 1 µs")
        print(f"MAC Timing - DIFS: {MACTiming.DIFS}µs, SIFS: {MACTiming.SIFS}µs, "
              f"SlotTime: {MACTiming.SLOT_TIME}µs")
        print(f"Packet TX: {MACTiming.DATA_DURATION}µs, ACK: {MACTiming.ACK_DURATION}µs")
        print()
        
        # Main simulation loop
        milestone = self.duration // 10
        for step in range(self.duration):
            self.step()
            
            # Progress indicator
            if (step + 1) % milestone == 0:
                progress = ((step + 1) / self.duration) * 100
                print(f"Progress: {progress:.0f}%")
        
        # Calculate final metrics
        self._calculate_final_metrics()
        return self.metrics
    
    def _calculate_final_metrics(self):
        """Calculate aggregate metrics at end of simulation"""
        total_generated = sum(n.total_generated for n in self.nodes)
        total_success = sum(n.total_success for n in self.nodes)
        total_collisions = sum(n.total_collisions for n in self.nodes)
        total_dropped = sum(n.total_dropped for n in self.nodes)
        total_pending = sum(len(n.queue) + (1 if n.current_packet is not None else 0) for n in self.nodes)
        
        # Throughput: successful packets per second
        self.metrics['throughput'] = (self.successful_transmissions / self.duration) * 1e6
        
        # Collision rate: collision events / total events
        total_events = self.successful_transmissions + self.collision_transmissions
        self.metrics['collision_rate'] = (
            self.collision_transmissions / total_events if total_events > 0 else 0
        )
        
        # Channel utilization: fraction of time channel was busy
        self.metrics['channel_utilization'] = self.busy_time / self.duration
        
        # Packet delivery ratio
        self.metrics['pdr'] = total_success / total_generated if total_generated > 0 else 0
        # PDR_mod: only count packets that actually finished (success or dropped)
        total_completed = total_success + total_dropped
        self.metrics['pdr_mod'] = total_success / total_completed if total_completed > 0 else 0
        
        # Node-level metrics
        self.metrics['total_generated'] = total_generated
        self.metrics['total_success'] = total_success
        self.metrics['total_collisions'] = total_collisions
        self.metrics['total_dropped'] = total_dropped
        self.metrics['total_pending'] = total_pending
        
        # Average backoff time per node
        avg_backoff = np.mean([n.total_backoff_time / 1e6 for n in self.nodes])  # in seconds
        self.metrics['avg_backoff_time'] = avg_backoff
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        print(f"Total Duration: {self.duration / 1e6:.2f} seconds")
        print(f"Successful Transmissions: {self.successful_transmissions}")
        print(f"Collision Events: {self.collision_transmissions}")
        print(f"Throughput: {self.metrics['throughput']:.2f} packets/sec")
        print(f"Collision Rate: {self.metrics['collision_rate']:.4f}")
        print(f"Channel Utilization: {self.metrics['channel_utilization']:.4f}")
        print(f"Packet Delivery Ratio: {self.metrics['pdr']:.4f}")
        print(f"Packet Delivery Ratio (mod): {self.metrics['pdr_mod']:.4f}")
        print(f"Avg Backoff Time: {avg_backoff:.3f} seconds")
        print("="*60)
