import random

# IEEE 802.11b DCF Timing (microseconds)
class MACTiming:
    SIFS = 10        # Short Inter-Frame Space
    SLOT_TIME = 20   # Backoff slot duration (µs)
    DIFS = SIFS + 2 * SLOT_TIME  # 50 µs - DCF IFS
    
    # Transmission durations 
    PACKET_SIZE = 1500  # bytes
    DATA_RATE = 1e6     # 1 Mbps = 
    DATA_DURATION = int((PACKET_SIZE * 8) / DATA_RATE * 1e6)  # ~12,000 µs
    
    # ACK frame (14 bytes header + 4 bytes FCS at 1 Mbps)
    ACK_SIZE = 14 + 4  # bytes
    ACK_DURATION = int((ACK_SIZE * 8) / DATA_RATE * 1e6)  # ~144 µs
    
    # Total successful transmission time
    # DATA + SIFS + ACK
    SUCCESS_DURATION = DATA_DURATION + SIFS + ACK_DURATION  # ~12,154 µs
    
    # Timing in slot units for optimized simulation
    DIFS_SLOTS = int(DIFS / SLOT_TIME)  # 50/20 = 2.5 ≈ 3 slots
    SIFS_SLOTS = max(1, int(SIFS / SLOT_TIME))  # 10/20 = 0.5 ≈ 1 slot
    DATA_SLOTS = int(DATA_DURATION / SLOT_TIME)  # 12000/20 = 600 slots


class Packet:
    def __init__(self, creation_time):
        self.creation_time = creation_time
        self.collisions = 0
        self.transmission_start_time = None


class Channel:
    """Channel with MAC timing states"""
    IDLE = 0
    SUCCESS = 1
    COLLISION = 2
    TRANSMITTING = 3  #ongoing transmission
    
    def __init__(self):
        self.status = Channel.IDLE
        self.transmitting_nodes = []
        self.transmission_end_time = None  # When current transmission ends
        self.collision_detected = False
        
    def is_busy(self, current_time):
        """Check if channel is busy (transmission in progress)"""
        if self.transmission_end_time is not None:
            return current_time < self.transmission_end_time
        return False
    
    def start_transmissions(self, transmitting_nodes, current_time):
        """
        Start transmissions and detect collisions.
        Returns: (status, collision_detected)
        """
        self.transmitting_nodes = transmitting_nodes
        num_tx = len(transmitting_nodes)
        
        if num_tx == 0:
            self.status = Channel.IDLE
            self.transmission_end_time = None
            self.collision_detected = False
            return Channel.IDLE, False
            
        elif num_tx == 1:
            # Single transmission. succeed after DATA + SIFS + ACK
            self.status = Channel.TRANSMITTING
            self.transmission_end_time = current_time + MACTiming.DATA_DURATION
            self.collision_detected = False
            #  transmission start
            transmitting_nodes[0].current_packet.transmission_start_time = current_time
            return Channel.TRANSMITTING, False
            
        else:
            # Multiple transmissions is a COLLISION
            # In reality, collision detected during preamble/header
            self.status = Channel.COLLISION
            self.transmission_end_time = current_time + MACTiming.DATA_DURATION
            self.collision_detected = True
            for node in transmitting_nodes:
                node.current_packet.transmission_start_time = current_time
            return Channel.COLLISION, True
    
    def check_transmission_end(self, current_time):
        """
        Check if ongoing transmission has ended.
        Returns: (ended, final_status)
        """
        if self.transmission_end_time is None:
            return False, Channel.IDLE
            
        if current_time >= self.transmission_end_time:
            # Transmission just ended
            if self.collision_detected:
                final_status = Channel.COLLISION
            else:
                final_status = Channel.SUCCESS
                
            # Clear transmission state
            self.transmission_end_time = None
            self.collision_detected = False
            self.status = Channel.IDLE
            
            return True, final_status
        
        return False, self.status


class Node:
    """Base node with MAC timing state machine"""
    STATE_IDLE = 'idle'
    STATE_DIFS_WAIT = 'difs_wait'
    STATE_BACKOFF = 'backoff'
    STATE_TRANSMITTING = 'transmitting'
    
    def __init__(self, node_id, packet_prob, time_granularity_us=1):
        """
        Args:
            node_id: Node identifier
            packet_prob: Packet generation probability
            time_granularity_us: Time step size in microseconds (1 for realistic, 20 for optimized)
        """
        self.id = node_id
        self.packet_prob = packet_prob
        self.time_granularity_us = time_granularity_us
        self.queue = []
        self.current_packet = None
        self.state = Node.STATE_IDLE
        
        # Timing state (in time_granularity_us units)
        self.backoff_counter = 0
        self.difs_counter = 0
        
        # Contention window
        self.cw = 0
        
        # Metrics
        self.total_success = 0
        self.total_collisions = 0
        self.total_generated = 0
        self.total_dropped = 0
        
        # Timing metrics
        self.total_backoff_time = 0
        self.total_transmission_time = 0
    
    def generate_packet(self, current_time):
        """Bernoulli packet generation"""
        if random.random() < self.packet_prob:
            self.queue.append(Packet(current_time))
            self.total_generated += 1
    
    def check_new_packet(self, current_time):
        """Check if we should start processing a new packet"""
        if self.current_packet is None and self.queue:
            self.current_packet = self.queue.pop(0)
            self.state = Node.STATE_DIFS_WAIT
            # DIFS counter in units
            self.difs_counter = int(MACTiming.DIFS / self.time_granularity_us)
            self.init_backoff()
    
    def init_backoff(self):
        """Initialize backoff counter - implemented by subclasses"""
        raise NotImplementedError
    
    def update(self, current_time, channel_busy):
        """
        Update node state based on channel status.
        Returns: True if ready to transmit, False otherwise
        """
        if self.current_packet is None:
            return False
        
        # State machine
        if self.state == Node.STATE_DIFS_WAIT:
            if channel_busy:
                # Channel busy, reset DIFS counter
                self.difs_counter = int(MACTiming.DIFS / self.time_granularity_us)
            else:
                # Channel idle, count down DIFS
                self.difs_counter -= 1
                if self.difs_counter <= 0:
                    # DIFS completed, enter backoff
                    self.state = Node.STATE_BACKOFF
        
        elif self.state == Node.STATE_BACKOFF:
            if channel_busy:
                # Channel busy, freeze backoff
                pass
            else:
                # Channel idle, count down backoff
                self.backoff_counter -= 1
                self.total_backoff_time += self.time_granularity_us
                
                if self.backoff_counter <= 0:
                    # Backoff expired, ready to transmit!
                    return True
        
        return False
    
    def start_transmission(self, current_time):
        """Node begins transmission"""
        self.state = Node.STATE_TRANSMITTING
        self.total_transmission_time += MACTiming.DATA_DURATION
    
    def handle_feedback(self, status, current_time):
        """Handle transmission result - implemented by subclasses"""
        raise NotImplementedError


class BEBNode(Node):
    """Binary Exponential Backoff"""
    CW_MIN = 4
    CW_MAX = 1024
    
    def __init__(self, node_id, packet_prob, time_granularity_us=1):
        super().__init__(node_id, packet_prob, time_granularity_us)
        self.cw = self.CW_MIN
    
    def init_backoff(self):
        """Set backoff counter in SlotTime units"""
        # Backoff is always in units of SlotTime (20µs)
        backoff_slots = random.randint(0, self.cw - 1)
        # Convert to time_granularity_us 
        self.backoff_counter = backoff_slots * int(MACTiming.SLOT_TIME / self.time_granularity_us)
    
    def handle_feedback(self, status, current_time):
        if status == Channel.SUCCESS:
            self.total_success += 1
            self.current_packet = None
            self.cw = self.CW_MIN
            self.state = Node.STATE_IDLE
            self.check_new_packet(current_time)
            
        elif status == Channel.COLLISION:
            self.total_collisions += 1
            self.current_packet.collisions += 1
            self.cw = min(self.cw * 2, self.CW_MAX)
            # Return to DIFS wait
            self.state = Node.STATE_DIFS_WAIT
            self.difs_counter = int(MACTiming.DIFS / self.time_granularity_us)
            self.init_backoff()


class BEBRetryNode(BEBNode):
    """BEB with retry limit (max 7 retries)"""
    MAX_RETRIES = 7
    
    def handle_feedback(self, status, current_time):
        if status == Channel.SUCCESS:
            super().handle_feedback(status, current_time)
        elif status == Channel.COLLISION:
            self.total_collisions += 1
            self.current_packet.collisions += 1
            
            if self.current_packet.collisions > self.MAX_RETRIES:
                # Drop packet
                self.total_dropped += 1
                self.current_packet = None
                self.cw = self.CW_MIN
                self.state = Node.STATE_IDLE
                self.check_new_packet(current_time)
            else:
                self.cw = min(self.cw * 2, self.CW_MAX)
                self.state = Node.STATE_DIFS_WAIT
                self.difs_counter = int(MACTiming.DIFS / self.time_granularity_us)
                self.init_backoff()


class RLNode(Node):
    """Q-Learning based backoff selection"""
    ACTIONS = [8, 16, 32, 64, 128, 256, 512, 1024]
    
    def __init__(self, node_id, packet_prob, time_granularity_us=1, alpha=0.1, gamma=0.9, epsilon=0.1,
                 reward_success=10, reward_collision=-10, epsilon_decay=0.0, epsilon_min=0.01):
        super().__init__(node_id, packet_prob, time_granularity_us)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reward_success = reward_success
        self.reward_collision = reward_collision
        
        # Q-table: State -> Action
        self.q_table = [[0.0 for _ in range(len(self.ACTIONS))] for _ in range(6)]
        self.last_state = 0
        self.last_action_idx = 0
    
    def get_state(self):
        if not self.current_packet:
            return 0
        return min(self.current_packet.collisions, 5)
    
    def choose_action(self):
        state = self.get_state()
        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.ACTIONS) - 1)
        else:
            max_val = max(self.q_table[state])
            best_actions = [i for i, v in enumerate(self.q_table[state]) if v == max_val]
            action_idx = random.choice(best_actions)
        
        self.last_state = state
        self.last_action_idx = action_idx
        return self.ACTIONS[action_idx]
    
    def init_backoff(self):
        cw = self.choose_action()
        self.cw = cw
        # Backoff is in SlotTime units, convert to time_granularity_us
        backoff_slots = random.randint(0, self.cw - 1)
        self.backoff_counter = backoff_slots * int(MACTiming.SLOT_TIME / self.time_granularity_us)
    
    def handle_feedback(self, status, current_time):
        reward = 0
        if status == Channel.SUCCESS:
            self.total_success += 1
            reward = self.reward_success
            self.update_q(reward)
            self.current_packet = None
            if self.epsilon_decay > 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))
            self.state = Node.STATE_IDLE
            self.check_new_packet(current_time)
            
        elif status == Channel.COLLISION:
            self.total_collisions += 1
            self.current_packet.collisions += 1
            reward = self.reward_collision
            self.update_q(reward)
            self.state = Node.STATE_DIFS_WAIT
            self.difs_counter = int(MACTiming.DIFS / self.time_granularity_us)
            self.init_backoff()
    
    def update_q(self, reward):
        current_q = self.q_table[self.last_state][self.last_action_idx]
        
        if self.current_packet is None:
            max_next_q = 0.0
        else:
            next_state = self.get_state()
            max_next_q = max(self.q_table[next_state])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[self.last_state][self.last_action_idx] = new_q


class RLRetryNode(RLNode):
    """RL with retry limit (max 7 retries)"""
    MAX_RETRIES = 7
    
    def handle_feedback(self, status, current_time):
        if status == Channel.SUCCESS:
            self.total_success += 1
            reward = self.reward_success
            self.update_q(reward)
            self.current_packet = None
            if self.epsilon_decay > 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))
            self.state = Node.STATE_IDLE
            self.check_new_packet(current_time)
            
        elif status == Channel.COLLISION:
            self.total_collisions += 1
            self.current_packet.collisions += 1
            
            if self.current_packet.collisions > self.MAX_RETRIES:
                # Drop packet
                self.total_dropped += 1
                self.current_packet = None
                self.state = Node.STATE_IDLE
                self.check_new_packet(current_time)
                reward = self.reward_collision
                self.update_q(reward)
            else:
                reward = self.reward_collision
                self.update_q(reward)
                self.state = Node.STATE_DIFS_WAIT
                self.difs_counter = int(MACTiming.DIFS / self.time_granularity_us)
                self.init_backoff()
