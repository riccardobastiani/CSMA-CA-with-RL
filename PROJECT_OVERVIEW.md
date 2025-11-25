# Project Overview: CSMA/CA Reinforcement Learning Simulation

## 1. Project Goal
The objective was to design, implement, and analyze a **Reinforcement Learning (RL)** based approach to the **CSMA/CA (Carrier Sense Multiple Access with Collision Avoidance)** protocol. The goal was to see if intelligent agents could learn to optimize their Contention Window (CW) to minimize collisions and maximize throughput in a decentralized wireless network, outperforming the standard **Binary Exponential Backoff (BEB)** algorithm.

## 2. Implementation Details

### Core Components (`models.py`)
*   **`Packet`**: Represents a data packet with creation time and collision count.
*   **`Channel`**: Simulates the shared medium.
    *   States: `IDLE` (0 nodes), `SUCCESS` (1 node), `COLLISION` (>1 nodes).
*   **`Node` (Base Class)**: Handles packet generation (Bernoulli process) and queue management.
*   **`BEBNode`**: Implements standard IEEE 802.11 DCF logic.
    *   Doubles CW on collision ($CW \times 2$), resets to minimum ($CW_{min}=4$) on success.
*   **`RLNode`**: Implements Q-Learning agents.
    *   **State**: Number of consecutive collisions (0-5).
    *   **Action**: Choose CW from $\{8, 16, 32, \dots, 1024\}$.
    *   **Reward**: $+10$ for Success, $-10$ for Collision.
    *   **Policy**: $\epsilon$-greedy with exponential decay.
*   **Retry Variants**: `BEBRetryNode` and `RLRetryNode` implement a retry limit (drop packet after 7 collisions) to prevent latency spikes.

### Simulation Engine (`simulation.py`)
*   **Discrete Time Slot System**: The engine advances time in discrete slots.
*   **Phases per Slot**:
    1.  **Traffic Generation**: Nodes generate packets based on probability $p$.
    2.  **Backoff**: Nodes decrement counters.
    3.  **Resolution**: Channel determines if 0, 1, or >1 nodes transmitted.
    4.  **Feedback**: Nodes receive immediate feedback to update CW (BEB) or Q-Table (RL).
    5.  **Metrics**: Tracks throughput, collisions, PDR, and fairness.
*   **Refactoring**: Added a `step()` method to allow for interactive real-time visualization.

### Experiments (`experiments.py`)
We conducted a suite of experiments to validate the model:
1.  **Scalability**: Tested $N=10$ to $500$ nodes.
    *   *Result*: RL significantly outperforms BEB in dense networks, maintaining stable throughput while BEB degrades.
2.  **Load Response**: Tested packet probability $p=0.1$ to $0.9$.
    *   *Result*: RL adapts to high load by proactively increasing CW.
3.  **Epsilon Strategy**: Compared fixed $\epsilon$ vs. decay.
    *   *Result*: Decay strategy ($\epsilon \to 0.01$) is essential for stability.
4.  **Reward Tuning**: Compared Aggressive vs. Balanced vs. Throughput rewards.
    *   *Result*: Balanced ($+10/-10$) works best.
5.  **Retry Limits**: Analyzed impact of dropping packets.
    *   *Result*: Prevents "congestive collapse" (infinite latency) at the cost of lower raw PDR.

### Visualization & Reporting
*   **Real-Time Demo (`demo.py`)**: A Tkinter/Matplotlib animation showing nodes (circles) changing color based on state (Idle/Backoff/Success/Collision) with a live throughput graph.
*   **Final Report (`REPORT.tex`)**: A comprehensive LaTeX document covering:
    *   System Model & Assumptions.
    *   Detailed Algorithm Descriptions (BEB vs. RL).
    *   Parameter Optimization (Exploration, Rewards).
    *   Results (Collision Reduction, Throughput, PDR).
    *   Formal mathematical formulations for Q-Learning.

## 3. Key Findings
1.  **Proactive vs. Reactive**: RL agents learn to be *proactive* (starting with high CW in dense networks), whereas BEB is *reactive* (must collide to increase CW).
2.  **Thundering Herd Solution**: RL solves the "thundering herd" problem where BEB nodes all reset to small CWs simultaneously after success.
3.  **Scalability**: RL is far superior for large networks ($N > 50$).
4.  **Stability**: Epsilon decay is critical for converging to a stable, low-variance policy.

## 4. File Structure
*   `models.py`: Class definitions for Channel, Packet, and Nodes.
*   `simulation.py`: Main engine logic.
*   `experiments.py`: Scripts to run batch simulations and generate plots.
*   `demo.py`: Interactive visualization script.
*   `main.py`: Entry point for simple CLI tests.
*   `results/`: Directory containing generated plots (`.png`).
*   `REPORT.tex`: Final project report in LaTeX.

## 5. Status
**Complete.**
*   All core logic implemented and tested.
*   Interactive demo functional.
*   Final experiments run with **10-seed averaging** for statistical robustness.
*   Report updated with final empirical data.
