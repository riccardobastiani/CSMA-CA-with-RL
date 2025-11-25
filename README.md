# CSMA/CA with Reinforcement Learning

A Q-learning based approach to optimize the CSMA/CA MAC protocol, demonstrating superior performance over Binary Exponential Backoff (BEB) in dense wireless networks.

## Quick Start

### Requirements
- Python 3.7+
- `numpy`, `matplotlib` (install via `pip install numpy matplotlib`)

### Run Experiments
```bash
# Run all main experiments (scalability, reward comparison, etc.)
python run_simulations.py

# Run additional experiments (load response, retry analysis)
python run_remaining_experiments.py

# Interactive visualization
python demo.py
```

## Key Files

- **`REPORT_FINAL.tex`** - Complete academic report with results and analysis
- **`models.py`** - Node implementations (BEBNode, RLNode)
- **`simulation.py`** - Discrete-event simulation engine
- **`experiments.py`** - All experiment definitions
- **`results/`** - Generated plots and CSV data

## Project Structure

```
├── models.py              # BEB and RL agent implementations
├── simulation.py          # Time-slotted network simulator
├── experiments.py         # Experiment suite (scalability, reward tuning, etc.)
├── demo.py                # Real-time visualization
├── run_simulations.py     # Main experiment runner
└── results/               # Plots (.png) and data (.csv)
```

## Results Summary

**Key Findings:**
- RL outperforms BEB in dense networks (N ≥ 100 nodes)
- BEB dominates in sparse networks (N < 50) due to RL exploration overhead
- RL reduces collision rate by 19.3% at N=200 while maintaining higher throughput
- All results averaged over 10 random seeds for statistical validity

See `REPORT_FINAL.tex` for comprehensive analysis.

## Reproducibility

All experiments use seeds 42-51. Raw per-seed data available in `results/*_raw.csv`. Simulation logs preserved in `simulation_output_detailed_v5.txt` and `simulation_output_remaining_v2.txt`.

## Citation

```
@misc{csma_rl_2024,
  title={CSMA/CA with Reinforcement Learning: A Performance Analysis},
  author={Your Name},
  year={2024}
}
```
