import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from simulation import SimulationEngine
from models import Channel

def run_demo():
    # Configuration
    NUM_NODES = 20
    PACKET_PROB = 0.1 # Low load to see individual transmissions clearly
    DURATION = 1000
    
    print("Initializing Simulation Demo...")
    sim = SimulationEngine(num_nodes=NUM_NODES, packet_prob=PACKET_PROB, node_type='RL', duration=DURATION)
    
    # Setup Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Node Positions (Circle)
    angles = np.linspace(0, 2*np.pi, NUM_NODES, endpoint=False)
    radius = 10
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    # Scatter Plot for Nodes
    scat = ax1.scatter(x, y, s=200, c='grey', edgecolors='black')
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-15, 15)
    ax1.set_title(f"CSMA/CA Network State (N={NUM_NODES})")
    ax1.axis('off')
    
    # Add node labels
    for i in range(NUM_NODES):
        ax1.text(x[i]*1.2, y[i]*1.2, str(i), ha='center', va='center')

    # Status Text
    status_text = ax1.text(0, 0, "IDLE", ha='center', va='center', fontsize=20, fontweight='bold')

    # Throughput Plot
    time_data = []
    throughput_data = []
    line, = ax2.plot([], [], lw=2)
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (Slots)')
    ax2.set_ylabel('Throughput (Moving Avg)')
    ax2.set_title('Live Throughput')
    ax2.grid(True)

    def update(frame):
        # Step Simulation
        status, transmitting_nodes = sim.step()
        
        # Update Node Colors
        # Default: Grey (Idle/Backoff)
        colors = ['lightgrey'] * NUM_NODES
        edgecolors = ['black'] * NUM_NODES
        
        # Highlight Transmitting Nodes
        transmitting_ids = [n.id for n in transmitting_nodes]
        
        for i in range(NUM_NODES):
            node = sim.nodes[i]
            if i in transmitting_ids:
                if status == Channel.SUCCESS:
                    colors[i] = 'limegreen' # Success
                    edgecolors[i] = 'green'
                else:
                    colors[i] = 'red' # Collision
                    edgecolors[i] = 'darkred'
            elif node.current_packet:
                colors[i] = 'skyblue' # Has packet, backing off
            else:
                colors[i] = 'lightgrey' # Idle, no packet

        scat.set_facecolors(colors)
        scat.set_edgecolors(edgecolors)
        
        # Update Center Text
        if status == Channel.SUCCESS:
            status_text.set_text("SUCCESS")
            status_text.set_color("green")
        elif status == Channel.COLLISION:
            status_text.set_text("COLLISION")
            status_text.set_color("red")
        else:
            status_text.set_text("IDLE")
            status_text.set_color("grey")

        # Update Throughput Graph
        if len(sim.metrics['history']) > 0:
            history = sim.metrics['history']
            times = [h['time'] for h in history]
            throughputs = [h['throughput'] for h in history]
            
            line.set_data(times, throughputs)
            ax2.set_xlim(0, max(200, times[-1] + 10))

        return scat, status_text, line

    print("Starting Animation...")
    ani = animation.FuncAnimation(fig, update, frames=range(DURATION), interval=100, blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    run_demo()
