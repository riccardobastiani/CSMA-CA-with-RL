import random
import math

class Packet:
    def __init__(self, creation_time):
        self.creation_time = creation_time
        self.collisions = 0

class Channel:
    IDLE = 0
    SUCCESS = 1
    COLLISION = 2

    def __init__(self):
        self.status = Channel.IDLE
        self.transmitting_nodes = []

    def resolve_slot(self, transmitting_nodes):
        self.transmitting_nodes = transmitting_nodes
        if len(transmitting_nodes) == 0:
            self.status = Channel.IDLE
        elif len(transmitting_nodes) == 1:
            self.status = Channel.SUCCESS
        else:
            self.status = Channel.COLLISION
        return self.status

class Node:
    def __init__(self, node_id, packet_prob):
        self.id = node_id
        self.packet_prob = packet_prob
        self.queue = []
        self.current_packet = None
        self.cw = 0
        self.backoff_counter = 0
        self.total_success = 0
        self.total_collisions = 0
        self.total_generated = 0
        self.total_dropped = 0

    def generate_packet(self, current_time):
        if random.random() < self.packet_prob:
            self.queue.append(Packet(current_time))
            self.total_generated += 1

    def check_new_packet(self):
        if self.current_packet is None and self.queue:
            self.current_packet = self.queue.pop(0)
            self.init_backoff()

    def init_backoff(self):
        raise NotImplementedError

    def decrement_backoff(self):
        # Only decrement if we have a packet and are waiting
        if self.current_packet:
            self.backoff_counter -= 1

    def ready_to_transmit(self):
        return self.current_packet and self.backoff_counter < 0

    def handle_feedback(self, status):
        raise NotImplementedError

class BEBNode(Node): # this is the Binary Exponentiation Backoff

    '''
    1. Base Model: 
BEBNode(Binary Exponential Backoff)
This class implements the standard CSMA/CA protocol
Logic: It uses a fixed rule to adjust its Contention Window (CW).
Start: It starts with a minimum window (CW_MIN = 4).
Collision: If a collision occurs, it doubles the window size (up to CW_MAX = 1024) to reduce the chance of colliding again.
Success: If transmission is successful, it resets the window back to the minimum (CW_MIN).
Key Characteristic: It is "reactive" and "memoryless" regarding long-term trends. It simply reacts to the immediate previous outcome (success or collision).
    '''
    CW_MIN = 4
    CW_MAX = 1024

    def __init__(self, node_id, packet_prob):
        super().__init__(node_id, packet_prob)
        self.cw = self.CW_MIN

    def init_backoff(self):
        self.backoff_counter = random.randint(0, self.cw - 1)

    def handle_feedback(self, status):
        if status == Channel.SUCCESS:
            self.total_success += 1
            self.current_packet = None
            self.cw = self.CW_MIN
            self.check_new_packet()
        elif status == Channel.COLLISION:
            self.total_collisions += 1
            self.current_packet.collisions += 1
            self.cw = min(self.cw * 2, self.CW_MAX)
            self.init_backoff()
        # If IDLE, do nothing (counter was already decremented)

class BEBRetryNode(BEBNode):
    MAX_RETRIES = 7

    def handle_feedback(self, status):
        if status == Channel.SUCCESS:
            super().handle_feedback(status)
        elif status == Channel.COLLISION:
            self.total_collisions += 1
            self.current_packet.collisions += 1
            
            if self.current_packet.collisions > self.MAX_RETRIES:
                # Drop packet
                self.total_dropped += 1
                self.current_packet = None
                self.cw = self.CW_MIN # Reset CW after drop
                self.check_new_packet()
            else:
                self.cw = min(self.cw * 2, self.CW_MAX)
                self.init_backoff()

class RLNode(Node):

    '''
Improved Model: 
RLNode
This class implements an adaptive agent using Q-Learning.
Logic: It learns a policy to choose the best CW size based on the current "State".
State: The state is defined as the number of consecutive collisions for the current packet (capped at 5).
Actions: Instead of just doubling, it can choose any specific window size from the set [8, 16, 32, ..., 1024].
Learning (Q-Table):
It maintains a table q_table[state][action] representing the expected reward for taking
a specific action in a specific state.
Reward: It gets +10 for success and -10 for collision.
Update: After every attempt, it updates the Q-value using the 
Bellman equation: $Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max Q(s', a') - Q(s,a)]$
Key Characteristic: It is "proactive" and "adaptive".
Over time, it learns which window size works best for a given congestion level
 (represented by collision count) to maximize long-term rewards.
    '''
    ACTIONS = [8, 16, 32, 64, 128, 256, 512, 1024]
    
    def __init__(self, node_id, packet_prob, alpha=0.1, gamma=0.9, epsilon=0.1, reward_success=10, reward_collision=-10, epsilon_decay=0.0, epsilon_min=0.01):
        super().__init__(node_id, packet_prob)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.epsilon_decay = epsilon_decay  # Decay rate per successful transmission
        self.epsilon_min = epsilon_min
        self.reward_success = reward_success
        self.reward_collision = reward_collision
        # Q-table: State -> Action. State is num_collisions (capped at 5)
        # Rows: States 0 to 5. Cols: Actions.
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
            # Argmax with random tie-breaking
            max_val = max(self.q_table[state])
            best_actions = [i for i, v in enumerate(self.q_table[state]) if v == max_val]
            action_idx = random.choice(best_actions)
        
        self.last_state = state
        self.last_action_idx = action_idx
        return self.ACTIONS[action_idx]

    def init_backoff(self):
        # In RL, we choose CW based on policy
        cw = self.choose_action()
        self.cw = cw
        self.backoff_counter = random.randint(0, self.cw - 1)

    def handle_feedback(self, status):
        reward = 0
        if status == Channel.SUCCESS:
            self.total_success += 1
            reward = self.reward_success # Positive reward
            self.update_q(reward)
            self.current_packet = None
            # Decay epsilon after success
            if self.epsilon_decay > 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))
            self.check_new_packet()
        elif status == Channel.COLLISION:
            self.total_collisions += 1
            self.current_packet.collisions += 1
            reward = self.reward_collision # Negative penalty
            self.update_q(reward)
            # Pick new backoff
            self.init_backoff()
        # If IDLE, maybe small penalty? For now 0.

    def update_q(self, reward):
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s,a))
        
        current_q = self.q_table[self.last_state][self.last_action_idx]
        
        if self.current_packet is None: # Success, terminal for this packet
             max_next_q = 0.0
        else:
            next_state = self.get_state()
            max_next_q = max(self.q_table[next_state])
            
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[self.last_state][self.last_action_idx] = new_q

class RLRetryNode(RLNode):
    MAX_RETRIES = 7

    def handle_feedback(self, status):
        if status == Channel.SUCCESS:
            self.total_success += 1
            reward = self.reward_success
            self.update_q(reward)
            self.current_packet = None
            # Decay epsilon after success
            if self.epsilon_decay > 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))
            self.check_new_packet()
        elif status == Channel.COLLISION:
            self.total_collisions += 1
            self.current_packet.collisions += 1
            
            # Check for retry limit
            if self.current_packet.collisions > self.MAX_RETRIES:
                # Drop packet
                self.total_dropped += 1
                self.current_packet = None
                self.check_new_packet()
                # We still apply the penalty for the collision that caused the drop
                reward = self.reward_collision
                self.update_q(reward)
            else:
                reward = self.reward_collision # Negative penalty
                self.update_q(reward)
                # Pick new backoff
                self.init_backoff()
