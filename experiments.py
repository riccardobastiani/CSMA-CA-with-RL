import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from simulation import SimulationEngine
from simulation_optimized_timing import OptimizedSimulationEngine
from models_realistic_timing import MACTiming

class ExperimentRunner:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_scalability_4plots_from_csv(self, raw_csv='scalability_raw.csv', node_counts=None, title_suffix='', out_png='scalability_4plots.png'):
        """Create a 4-panel summary (Throughput, PDR, Collision Rate, Efficiency) from an existing raw CSV.

        Expects the CSV to have columns: Experiment, Nodes, Algorithm, Seed, Throughput, PDR, CollisionRate.
        If present, PDR_mod is ignored here and can be plotted separately.
        This avoids rerunning simulations and ensures plots reflect the exact same samples.
        """
        filepath = os.path.join(self.output_dir, raw_csv)
        df = pd.read_csv(filepath)

        if 'Experiment' in df.columns:
            df = df[df['Experiment'] == 'Scalability']

        if node_counts is not None:
            df = df[df['Nodes'].isin(node_counts)]

        if df.empty:
            raise ValueError(f"No matching rows found in {filepath} for Experiment='Scalability' and node_counts={node_counts}")

        df = df.copy()
        df['Efficiency'] = df['Throughput'] * (1 - df['CollisionRate'])

        def summarize(metric_name):
            grouped = df.groupby(['Algorithm', 'Nodes'])[metric_name]
            stats = grouped.agg(['mean', 'std', 'count']).reset_index()
            stats['ci'] = 0.0
            mask = stats['count'] > 1
            stats.loc[mask, 'ci'] = 1.96 * (stats.loc[mask, 'std'] / np.sqrt(stats.loc[mask, 'count']))
            return stats

        stats_throughput = summarize('Throughput')
        stats_pdr = summarize('PDR')
        stats_collision = summarize('CollisionRate')
        stats_eff = summarize('Efficiency')

        if node_counts is None:
            node_counts = sorted(df['Nodes'].unique().tolist())

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

        def plot_panel(ax, stats, ylabel, title):
            for algo, marker in [('BEB', 'o'), ('RL', 's')]:
                sub = stats[stats['Algorithm'] == algo].set_index('Nodes').reindex(node_counts)
                ax.errorbar(node_counts, sub['mean'], yerr=sub['ci'], label=f'{algo} (95% CI)', marker=marker, capsize=4, linewidth=2)
            ax.set_xlabel('Number of Nodes')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        ax1 = fig.add_subplot(gs[0, 0])
        plot_panel(ax1, stats_throughput, 'Aggregate Throughput', 'Scalability: Throughput vs Network Density')

        ax2 = fig.add_subplot(gs[0, 1])
        plot_panel(ax2, stats_pdr, 'Packet Delivery Ratio (PDR)', 'Scalability: PDR vs Network Density')

        ax3 = fig.add_subplot(gs[1, 0])
        plot_panel(ax3, stats_collision, 'Collision Rate', 'Scalability: Collision Rate vs Network Density')

        ax4 = fig.add_subplot(gs[1, 1])
        plot_panel(ax4, stats_eff, 'Efficiency (throughput × (1 - collision))', 'Scalability: Efficiency vs Network Density')

        suptitle = 'Scalability: 4-Metric Summary'
        if title_suffix:
            suptitle += f' {title_suffix}'
        plt.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.995)

        outpath = os.path.join(self.output_dir, out_png)
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"Scalability 4-plot figure saved: {outpath}")
            
    def save_results_to_csv(self, filename, data, mode='w'):
        """Helper to save results to CSV"""
        filepath = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(data)
        # Append if file exists and mode is 'a', otherwise write
        if mode == 'a' and os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, mode=mode, index=False)
        print(f"Saved results to {filepath}")

    def run_scalability(self, node_counts, packet_prob=0.5, duration=1000, seeds=10, rl_kwargs=None, output_prefix='scalability'):
        print("Running Scalability Experiment...")
        results = {'BEB': [], 'RL': []}
        results_full = {'BEB': [], 'RL': []}
        results_full_mod = {'BEB': [], 'RL': []}
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        n_seeds = len(seed_list)

        rl_kwargs = rl_kwargs or {}
        
        raw_data = []
        averaged_data = []

        for N in node_counts:
            print(f"  Nodes: {N}")
            for algo in ['BEB', 'RL']:
                throughputs = []
                pdrs = []
                pdr_mods = []
                for seed in seed_list:
                    sim_kwargs = rl_kwargs if algo == 'RL' else {}
                    # Use retry variants for realistic PDR_mod (packets drop after MAX_RETRIES)
                    node_type = 'BEB_RETRY' if algo == 'BEB' else 'RL_RETRY'
                    sim = SimulationEngine(num_nodes=N, packet_prob=packet_prob, node_type=node_type, duration=duration, seed=seed, **sim_kwargs)
                    metrics = sim.run()
                    throughputs.append(metrics['throughput'])
                    pdrs.append(metrics['pdr'])
                    pdr_mods.append(metrics['pdr_mod'])
                    
                    # Collect raw data
                    raw_data.append({
                        'Experiment': 'Scalability',
                        'Nodes': N,
                        'Algorithm': algo,
                        'Seed': seed,
                        'Throughput': metrics['throughput'],
                        'PDR': metrics['pdr'],
                        'PDR_mod': metrics['pdr_mod'],
                        'CollisionRate': metrics['collision_rate']
                    })
                    print(
                        f"    Seed {seed}: Throughput={metrics['throughput']:.4f}, "
                        f"PDR={metrics['pdr']:.4f}, PDR_mod={metrics['pdr_mod']:.4f}"
                    )
                
                avg_throughput = np.mean(throughputs)
                std_throughput = np.std(throughputs, ddof=1) if n_seeds > 1 else 0.0
                ci_throughput = 1.96 * std_throughput / np.sqrt(n_seeds) if n_seeds > 1 else 0.0  # 95% CI
                
                results[algo].append({'mean': avg_throughput, 'ci': ci_throughput})
                
                avg_pdr = np.mean(pdrs)
                std_pdr = np.std(pdrs, ddof=1) if n_seeds > 1 else 0.0
                ci_pdr = 1.96 * std_pdr / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
                
                results_full[algo].append({'mean': avg_pdr, 'ci': ci_pdr})

                avg_pdr_mod = np.mean(pdr_mods)
                std_pdr_mod = np.std(pdr_mods, ddof=1) if n_seeds > 1 else 0.0
                ci_pdr_mod = 1.96 * std_pdr_mod / np.sqrt(n_seeds) if n_seeds > 1 else 0.0

                results_full_mod[algo].append({'mean': avg_pdr_mod, 'ci': ci_pdr_mod})
                
                averaged_data.append({
                    'Experiment': 'Scalability',
                    'Nodes': N,
                    'Algorithm': algo,
                    'Avg_Throughput': avg_throughput,
                    'Avg_PDR': avg_pdr,
                    'Avg_PDR_mod': avg_pdr_mod
                })

        self.save_results_to_csv(f'{output_prefix}_raw.csv', raw_data)
        self.save_results_to_csv(f'{output_prefix}_averaged.csv', averaged_data)
        
        # Plot 
        plt.figure()
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results[algo]]
            cis = [r['ci'] for r in results[algo]]
            plt.errorbar(node_counts, means, yerr=cis, label=f'{algo} (95% CI)', marker=marker, capsize=5)
            
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Aggregate Throughput')
        plt.title('Scalability: Throughput vs Network Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/{output_prefix}.png')
        print("Scalability plot saved.")

        # Plot 
        
        plt.figure()
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results_full[algo]]
            cis = [r['ci'] for r in results_full[algo]]
            plt.errorbar(node_counts, means, yerr=cis, label=f'{algo} (95% CI)', marker=marker, capsize=5)

        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Packet Delivery Ratio (PDR)')
        plt.title('Scalability: PDR vs Network Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/{output_prefix}_pdr.png')
        print("Scalability PDR plot saved.")

        # Plot PDR_mod
        plt.figure()
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results_full_mod[algo]]
            cis = [r['ci'] for r in results_full_mod[algo]]
            plt.errorbar(node_counts, means, yerr=cis, label=f'{algo} (95% CI)', marker=marker, capsize=5)

        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Packet Delivery Ratio (PDR_mod)')
        plt.title('Scalability: PDR_mod vs Network Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/{output_prefix}_pdr_mod.png')
        print("Scalability PDR_mod plot saved.")

        # Plot PDR vs PDR_mod comparison
        plt.figure()
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            pdr_means = [r['mean'] for r in results_full[algo]]
            pdr_cis = [r['ci'] for r in results_full[algo]]
            pdr_mod_means = [r['mean'] for r in results_full_mod[algo]]
            pdr_mod_cis = [r['ci'] for r in results_full_mod[algo]]
            plt.errorbar(node_counts, pdr_means, yerr=pdr_cis, label=f'{algo} PDR', marker=marker, capsize=5)
            plt.errorbar(node_counts, pdr_mod_means, yerr=pdr_mod_cis, label=f'{algo} PDR_mod', marker=marker, linestyle='--', capsize=5)

        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Packet Delivery Ratio')
        plt.title('Scalability: PDR vs PDR_mod (Network Density)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/{output_prefix}_pdr_compare.png')
        print("Scalability PDR comparison plot saved.")

    def run_optimized_timing_scalability(self, node_counts, packet_prob=0.5, duration=5000, seeds=10):
        """Scalability using the optimized realistic-timing engine (SlotTime=20µs)."""
        print("Running Scalability Experiment (Optimized Timing Engine)...")
        print("Note: duration is in SlotTime units (20µs each).")

        simulated_seconds = (duration * MACTiming.SLOT_TIME) / 1e6

        results = {'BEB': [], 'RL': []}  # throughput
        results_full = {'BEB': [], 'RL': []}  # pdr
        results_full_mod = {'BEB': [], 'RL': []}  # pdr_mod
        results_collision = {'BEB': [], 'RL': []}
        results_efficiency = {'BEB': [], 'RL': []}

        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        n_seeds = len(seed_list)

        raw_data = []
        averaged_data = []

        for N in node_counts:
            print(f"  Nodes: {N}")
            for algo in ['BEB', 'RL']:
                throughputs = []
                pdrs = []
                pdr_mods = []
                collisions = []
                efficiencies = []
                for seed in seed_list:
                    # Use retry variants for realistic PDR_mod (packets drop after MAX_RETRIES)
                    node_type = 'BEB_RETRY' if algo == 'BEB' else 'RL_RETRY'
                    sim = OptimizedSimulationEngine(
                        num_nodes=N,
                        packet_prob=packet_prob,
                        node_type=node_type,
                        duration=duration,
                        seed=seed,
                    )
                    metrics = sim.run()
                    throughputs.append(metrics['throughput'])
                    pdrs.append(metrics['pdr'])
                    pdr_mods.append(metrics['pdr_mod'])
                    collisions.append(metrics['collision_rate'])
                    efficiencies.append(metrics['throughput'] * (1 - metrics['collision_rate']))

                    raw_data.append({
                        'Experiment': 'OptimizedTimingScalability',
                        'Nodes': N,
                        'Algorithm': algo,
                        'Seed': seed,
                        'Throughput': metrics['throughput'],
                        'PDR': metrics['pdr'],
                        'PDR_mod': metrics['pdr_mod'],
                        'CollisionRate': metrics['collision_rate'],
                        'ChannelUtilization': metrics.get('channel_utilization', None),
                        'TotalGenerated': metrics.get('total_generated', None),
                        'TotalSuccess': metrics.get('total_success', None),
                        'Efficiency': metrics['throughput'] * (1 - metrics['collision_rate']),
                    })
                    print(
                        f"    Seed {seed}: Throughput={metrics['throughput']:.4f}, "
                        f"PDR={metrics['pdr']:.6e}, PDR_mod={metrics['pdr_mod']:.6e}"
                    )

                avg_throughput = np.mean(throughputs)
                std_throughput = np.std(throughputs, ddof=1) if n_seeds > 1 else 0.0
                ci_throughput = 1.96 * std_throughput / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
                results[algo].append({'mean': avg_throughput, 'ci': ci_throughput})

                avg_pdr = np.mean(pdrs)
                std_pdr = np.std(pdrs, ddof=1) if n_seeds > 1 else 0.0
                ci_pdr = 1.96 * std_pdr / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
                results_full[algo].append({'mean': avg_pdr, 'ci': ci_pdr})

                avg_pdr_mod = np.mean(pdr_mods)
                std_pdr_mod = np.std(pdr_mods, ddof=1) if n_seeds > 1 else 0.0
                ci_pdr_mod = 1.96 * std_pdr_mod / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
                results_full_mod[algo].append({'mean': avg_pdr_mod, 'ci': ci_pdr_mod})

                avg_collision = np.mean(collisions)
                std_collision = np.std(collisions, ddof=1) if n_seeds > 1 else 0.0
                ci_collision = 1.96 * std_collision / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
                results_collision[algo].append({'mean': avg_collision, 'ci': ci_collision})

                avg_efficiency = np.mean(efficiencies)
                std_efficiency = np.std(efficiencies, ddof=1) if n_seeds > 1 else 0.0
                ci_efficiency = 1.96 * std_efficiency / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
                results_efficiency[algo].append({'mean': avg_efficiency, 'ci': ci_efficiency})

                averaged_data.append({
                    'Experiment': 'OptimizedTimingScalability',
                    'Nodes': N,
                    'Algorithm': algo,
                    'Avg_Throughput': avg_throughput,
                    'Avg_PDR': avg_pdr,
                    'Avg_PDR_mod': avg_pdr_mod,
                    'Avg_CollisionRate': avg_collision,
                    'Avg_Efficiency': avg_efficiency,
                })

        self.save_results_to_csv('optimized_timing_scalability_raw.csv', raw_data)
        self.save_results_to_csv('optimized_timing_scalability_averaged.csv', averaged_data)

        # Plot Throughput with CI
        plt.figure()
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results[algo]]
            cis = [r['ci'] for r in results[algo]]
            plt.errorbar(node_counts, means, yerr=cis, label=f'{algo} (95% CI)', marker=marker, capsize=5)

        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Throughput (packets/sec)')
        plt.title(f'Scalability (Optimized Timing): Throughput vs Network Density\n(p={packet_prob}, T={simulated_seconds:.2f}s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/optimized_timing_scalability.png')
        print("Optimized timing scalability plot saved.")

        # Plot PDR with CI
        plt.figure()
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results_full[algo]]
            cis = [r['ci'] for r in results_full[algo]]
            plt.errorbar(node_counts, means, yerr=cis, label=f'{algo} (95% CI)', marker=marker, capsize=5)

        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Packet Delivery Ratio (PDR)')
        plt.title(f'Scalability (Optimized Timing): PDR vs Network Density\n(p={packet_prob}, T={simulated_seconds:.2f}s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/optimized_timing_scalability_pdr.png')
        print("Optimized timing scalability PDR plot saved.")

        # Plot PDR_mod with CI
        plt.figure()
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results_full_mod[algo]]
            cis = [r['ci'] for r in results_full_mod[algo]]
            plt.errorbar(node_counts, means, yerr=cis, label=f'{algo} (95% CI)', marker=marker, capsize=5)

        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Packet Delivery Ratio (PDR_mod)')
        plt.title(f'Scalability (Optimized Timing): PDR_mod vs Network Density\n(p={packet_prob}, T={simulated_seconds:.2f}s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/optimized_timing_scalability_pdr_mod.png')
        print("Optimized timing scalability PDR_mod plot saved.")

        # Plot PDR vs PDR_mod comparison
        plt.figure()
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            pdr_means = [r['mean'] for r in results_full[algo]]
            pdr_cis = [r['ci'] for r in results_full[algo]]
            pdr_mod_means = [r['mean'] for r in results_full_mod[algo]]
            pdr_mod_cis = [r['ci'] for r in results_full_mod[algo]]
            plt.errorbar(node_counts, pdr_means, yerr=pdr_cis, label=f'{algo} PDR', marker=marker, capsize=5)
            plt.errorbar(node_counts, pdr_mod_means, yerr=pdr_mod_cis, label=f'{algo} PDR_mod', marker=marker, linestyle='--', capsize=5)

        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Packet Delivery Ratio')
        plt.title(f'Scalability (Optimized Timing): PDR vs PDR_mod\n(p={packet_prob}, T={simulated_seconds:.2f}s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/optimized_timing_scalability_pdr_compare.png')
        print("Optimized timing scalability PDR comparison plot saved.")

        # 4-panel summary (Throughput, PDR, Collision Rate, Efficiency)
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results[algo]]
            ax1.plot(node_counts, means, marker=marker, label=algo, linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Nodes', fontsize=12)
        ax1.set_ylabel('Throughput (packets/sec)', fontsize=12)
        ax1.set_title(f'Scalability (Optimized Timing): Throughput vs Network Density\n(p={packet_prob}, T={simulated_seconds:.2f}s)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results_full[algo]]
            ax2.plot(node_counts, means, marker=marker, label=algo, linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Nodes', fontsize=12)
        ax2.set_ylabel('Packet Delivery Ratio (PDR)', fontsize=12)
        ax2.set_title(f'Scalability (Optimized Timing): PDR vs Network Density\n(p={packet_prob}, T={simulated_seconds:.2f}s)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results_collision[algo]]
            ax3.plot(node_counts, means, marker=marker, label=algo, linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Nodes', fontsize=12)
        ax3.set_ylabel('Collision Rate', fontsize=12)
        ax3.set_title(f'Scalability (Optimized Timing): Collision Rate vs Network Density\n(p={packet_prob}, T={simulated_seconds:.2f}s)', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        for algo, marker in [('BEB', 'o'), ('RL', 's')]:
            means = [r['mean'] for r in results_efficiency[algo]]
            ax4.plot(node_counts, means, marker=marker, label=algo, linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Nodes', fontsize=12)
        ax4.set_ylabel('Efficiency (throughput × (1 - collision))', fontsize=12)
        ax4.set_title(f'Scalability (Optimized Timing): Efficiency vs Network Density\n(p={packet_prob}, T={simulated_seconds:.2f}s)', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Optimized Timing Scalability: 4-Metric Summary', fontsize=15, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/optimized_timing_scalability_4plots.png', dpi=150, bbox_inches='tight')
        print("Optimized timing scalability 4-plot figure saved.")

    def run_load_response(self, packet_probs, node_counts=[10], duration=1000, seeds=10):
        print("Running Load Response Experiment...")
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        for N in node_counts:
            print(f"  Running Load Response for Nodes: {N}")
            results = {'BEB': [], 'RL': []}
            
            for p in packet_probs:
                print(f"    Load (p): {p}")
                for algo in ['BEB', 'RL']:
                    collisions = []
                    for seed in seed_list:
                        sim = SimulationEngine(num_nodes=N, packet_prob=p, node_type=algo, duration=duration, seed=seed)
                        metrics = sim.run()
                        collisions.append(metrics['collision_rate'])
                    
                    avg_collision = np.mean(collisions)
                    std_collision = np.std(collisions, ddof=1)
                    ci_collision = 1.96 * std_collision / np.sqrt(len(seed_list))
                    
                    results[algo].append({'mean': avg_collision, 'ci': ci_collision})
            
            # Plot for this N
            plt.figure()
            for algo, marker in [('BEB', 'o'), ('RL', 's')]:
                means = [r['mean'] for r in results[algo]]
                cis = [r['ci'] for r in results[algo]]
                plt.errorbar(packet_probs, means, yerr=cis, label=f'{algo} (95% CI)', marker=marker, capsize=5)
                
            plt.xlabel('Packet Generation Probability (p)')
            plt.ylabel('Collision Rate')
            plt.title(f'Load Response: Collision Rate vs Traffic Load (N={N})')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.output_dir}/load_response_{N}.png')
            print(f"Load Response plot for N={N} saved.")
            plt.close()

    def run_learning_stability(self, duration=5000, num_nodes=10, packet_prob=0.8, seeds=10):
        print("Running Learning Stability Experiment...")
        results = {'BEB': [], 'RL': []}
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        # Run multiple simulations and average history
        for algo in ['BEB', 'RL']:
            all_histories = []
            for seed in seed_list:
                sim = SimulationEngine(num_nodes=num_nodes, packet_prob=packet_prob, node_type=algo, duration=duration, seed=seed)
                metrics = sim.run()
                all_histories.append(metrics['history'])
            
            # Average the histories
            avg_history = []
            if all_histories:
                num_points = len(all_histories[0])
                for i in range(num_points):
                    time_point = all_histories[0][i]['time']
                    avg_throughput = np.mean([h[i]['throughput'] for h in all_histories])
                    avg_collision = np.mean([h[i]['collision_rate'] for h in all_histories])
                    avg_history.append({
                        'time': time_point,
                        'throughput': avg_throughput,
                        'collision_rate': avg_collision
                    })
            results[algo] = avg_history
            
        # Plot Throughput over Time
        plt.figure()
        times = [h['time'] for h in results['BEB']]
        beb_throughput = [h['throughput'] for h in results['BEB']]
        rl_throughput = [h['throughput'] for h in results['RL']]
        
        plt.plot(times, beb_throughput, label='Static CSMA (BEB)', linestyle='--')
        plt.plot(times, rl_throughput, label='RL-CSMA')
        plt.xlabel('Time (Slots)')
        plt.ylabel('Throughput (Moving Avg)')
        plt.title('Learning Stability: Throughput Convergence')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/stability_throughput.png')
        print("Stability plot saved.")

    def run_reward_comparison(self, duration=2000, num_nodes=50, packet_prob=0.7, seeds=10):
        print("Running Reward Comparison Experiment...")
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        # Define configurations
        configs = [
            {'name': 'BEB', 'type': 'BEB', 'kwargs': {}},
            {'name': 'RL (Std)', 'type': 'RL', 'kwargs': {'reward_success': 10, 'reward_collision': -10}},
            {'name': 'RL (Aggressive)', 'type': 'RL', 'kwargs': {'reward_success': 10, 'reward_collision': -50}},
            {'name': 'RL (Throughput)', 'type': 'RL', 'kwargs': {'reward_success': 50, 'reward_collision': -10}},
            {'name': 'RL (Balanced)', 'type': 'RL', 'kwargs': {'reward_success': 1, 'reward_collision': -1}},
            {'name': 'RL (Low Alpha)', 'type': 'RL', 'kwargs': {'alpha': 0.01}},
            {'name': 'RL (High Alpha)', 'type': 'RL', 'kwargs': {'alpha': 0.5}},
            {'name': 'RL (Low Gamma)', 'type': 'RL', 'kwargs': {'gamma': 0.1}},
            {'name': 'RL (High Gamma)', 'type': 'RL', 'kwargs': {'gamma': 0.99}},
        ]
        
        results = {cfg['name']: {'throughput': [], 'collision': []} for cfg in configs}
        
        raw_data = []
        averaged_data = []

        for cfg in configs:
            print(f"  Testing {cfg['name']}...")
            throughputs = []
            collisions = []
            
            for seed in seed_list:
                sim = SimulationEngine(num_nodes=num_nodes, packet_prob=packet_prob, node_type=cfg['type'], duration=duration, seed=seed, **cfg['kwargs'])
                metrics = sim.run()
                throughputs.append(metrics['throughput'])
                collisions.append(metrics['collision_rate'])
                
                raw_data.append({
                    'Experiment': 'RewardComparison',
                    'Config': cfg['name'],
                    'Seed': seed,
                    'Throughput': metrics['throughput'],
                    'CollisionRate': metrics['collision_rate'],
                    'Efficiency': metrics['throughput'] * (1 - metrics['collision_rate'])
                })
                print(f"    Seed {seed}: Throughput={metrics['throughput']:.4f}, Collision={metrics['collision_rate']:.4f}")
            
            results[cfg['name']]['throughput'] = np.mean(throughputs)
            results[cfg['name']]['collision'] = np.mean(collisions)
            results[cfg['name']]['throughput_std'] = np.std(throughputs)
            results[cfg['name']]['collision_std'] = np.std(collisions)
            
            averaged_data.append({
                'Experiment': 'RewardComparison',
                'Config': cfg['name'],
                'Avg_Throughput': np.mean(throughputs),
                'Avg_CollisionRate': np.mean(collisions),
                'Avg_Efficiency': np.mean(throughputs) * (1 - np.mean(collisions))
            })
                


        self.save_results_to_csv('reward_comparison_raw.csv', raw_data)
        self.save_results_to_csv('reward_comparison_averaged.csv', averaged_data)

        # Plotting - 3 Subplots
        labels = [cfg['name'] for cfg in configs]
        throughput_means = [results[name]['throughput'] for name in labels]
        collision_means = [results[name]['collision'] for name in labels]
        
        # Calculate Efficiency Score: Throughput * (1 - Collision Rate)
        efficiency_scores = []
        for name in labels:
            eff = results[name]['throughput'] * (1 - results[name]['collision'])
            efficiency_scores.append(eff)
            print(f"  {name} Efficiency Score: {eff:.4f}")

        x = np.arange(len(labels))
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Plot 1: Throughput
        bars1 = ax1.bar(x, throughput_means, color='tab:blue', alpha=0.8)
        ax1.set_ylabel('Throughput')
        ax1.set_title('Throughput Comparison')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 1.0)
        
        # Plot 2: Collision Rate
        bars2 = ax2.bar(x, collision_means, color='tab:red', alpha=0.8)
        ax2.set_ylabel('Collision Rate')
        ax2.set_title('Collision Rate Comparison')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 1.0)
        
        # Plot 3: Efficiency Score
        bars3 = ax3.bar(x, efficiency_scores, color='tab:green', alpha=0.8)
        ax3.set_ylabel('Efficiency Score')
        ax3.set_title('Efficiency Score Comparison (Throughput * (1 - Collision))')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.set_ylim(0, 0.5) # Efficiency is usually lower

        # Common X-axis settings
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels function
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(bars1, ax1)
        autolabel(bars2, ax2)
        autolabel(bars3, ax3)
        
        plt.savefig(f'{self.output_dir}/reward_comparison.png')
        print("Reward comparison plot saved.")

    def run_retry_comparison(self, duration=2000, num_nodes=50, packet_prob=0.7, seeds=10):
        print("Running Retry Limit Comparison Experiment...")
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        # Define configurations
        configs = [
            {'name': 'BEB', 'type': 'BEB', 'kwargs': {}},
            {'name': 'BEB + Retry Limit', 'type': 'BEB_RETRY', 'kwargs': {}},
            {'name': 'RL (Std)', 'type': 'RL', 'kwargs': {'reward_success': 10, 'reward_collision': -10}},
            {'name': 'RL + Retry Limit', 'type': 'RL_RETRY', 'kwargs': {'reward_success': 10, 'reward_collision': -10}},
        ]
        
        results = {cfg['name']: {'throughput': [], 'collision': [], 'pdr': []} for cfg in configs}
        
        for cfg in configs:
            print(f"  Testing {cfg['name']}...")
            throughputs = []
            collisions = []
            pdrs = []
            
            for seed in seed_list:
                sim = SimulationEngine(num_nodes=num_nodes, packet_prob=packet_prob, node_type=cfg['type'], duration=duration, seed=seed, **cfg['kwargs'])
                metrics = sim.run()
                throughputs.append(metrics['throughput'])
                collisions.append(metrics['collision_rate'])
                pdrs.append(metrics['pdr'])
                print(f"    Seed {seed}: Throughput={metrics['throughput']:.4f}, PDR={metrics['pdr']:.4f}")
            
            results[cfg['name']]['throughput'] = np.mean(throughputs)
            results[cfg['name']]['collision'] = np.mean(collisions)
            results[cfg['name']]['pdr'] = np.mean(pdrs)
            results[cfg['name']]['throughput_std'] = np.std(throughputs)
            results[cfg['name']]['collision_std'] = np.std(collisions)
            results[cfg['name']]['pdr_std'] = np.std(pdrs)

        # Plotting Bar Chart for Comparison
        labels = [cfg['name'] for cfg in configs]
        throughput_means = [results[name]['throughput'] for name in labels]
        collision_means = [results[name]['collision'] for name in labels]
        pdr_means = [results[name]['pdr'] for name in labels]

        # Calculate Efficiency Score
        efficiency_scores = []
        for name in labels:
            eff = results[name]['throughput'] * (1 - results[name]['collision'])
            efficiency_scores.append(eff)
            print(f"  {name} - Throughput: {results[name]['throughput']:.4f}, PDR: {results[name]['pdr']:.4f}, Efficiency: {eff:.4f}")

        # Plotting
        x = np.arange(len(labels))
        width = 0.2
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Bar 1: Throughput
        rects1 = ax1.bar(x - 1.5*width, throughput_means, width, label='Throughput', color='tab:blue', alpha=0.8)
        ax1.set_ylabel('Metric Value', fontsize=12)
        ax1.set_title('Retry Limit Impact (N=50, p=0.7)', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha='right')
        ax1.set_ylim(0, 1.0)
        
        # Bar 2: Collision Rate
        rects2 = ax1.bar(x - 0.5*width, collision_means, width, label='Collision Rate', color='tab:red', alpha=0.8)
        
        # Bar 3: PDR
        rects3 = ax1.bar(x + 0.5*width, pdr_means, width, label='PDR', color='tab:orange', alpha=0.8)
        
        # Bar 4: Efficiency Score
        rects4 = ax1.bar(x + 1.5*width, efficiency_scores, width, label='Efficiency Score', color='tab:green', alpha=0.8)
        
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax1.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/retry_comparison.png')
        print("Retry comparison plot saved.")

    def run_retry_scalability(self, node_counts=[10, 50, 100, 200, 500], duration=5000, seeds=10):
        print("Running Retry Scalability Comparison...")
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        configs = [
            {'name': 'BEB', 'type': 'BEB'},
            {'name': 'BEB + Retry', 'type': 'BEB_RETRY'},
            {'name': 'RL', 'type': 'RL'},
            {'name': 'RL + Retry', 'type': 'RL_RETRY'},
        ]
        
        results = {cfg['name']: {'throughput': [], 'pdr': [], 'dropped': []} for cfg in configs}
        
        for N in node_counts:
            print(f"  Nodes: {N}")
            for cfg in configs:
                throughputs = []
                pdrs = []
                dropped_rates = []
                
                for seed in seed_list:
                    sim = SimulationEngine(num_nodes=N, packet_prob=0.5, node_type=cfg['type'], duration=duration, seed=seed)
                    metrics = sim.run()
                    throughputs.append(metrics['throughput'])
                    pdrs.append(metrics['pdr'])
                    dropped_rate = metrics['total_dropped'] / metrics['total_generated'] if metrics['total_generated'] > 0 else 0
                    dropped_rates.append(dropped_rate)
                
                results[cfg['name']]['throughput'].append(np.mean(throughputs))
                results[cfg['name']]['pdr'].append(np.mean(pdrs))
                results[cfg['name']]['dropped'].append(np.mean(dropped_rates))
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Throughput comparison
        for cfg in configs:
            ax1.plot(node_counts, results[cfg['name']]['throughput'], marker='o', label=cfg['name'], linewidth=2)
        ax1.set_xlabel('Number of Nodes', fontsize=12)
        ax1.set_ylabel('Throughput', fontsize=12)
        ax1.set_title('Scalability: Throughput with/without Retry Limits', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: PDR comparison
        for cfg in configs:
            ax2.plot(node_counts, results[cfg['name']]['pdr'], marker='s', label=cfg['name'], linewidth=2)
        ax2.set_xlabel('Number of Nodes', fontsize=12)
        ax2.set_ylabel('Packet Delivery Ratio (PDR)', fontsize=12)
        ax2.set_title('Scalability: PDR with/without Retry Limits', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/retry_scalability.png')
        print("Retry scalability plot saved.")

    def run_retry_convergence(self, duration=5000, num_nodes=50, packet_prob=0.7, seeds=10):
        print("Running Retry Convergence Analysis...")
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        configs = [
            {'name': 'BEB', 'type': 'BEB'},
            {'name': 'BEB + Retry', 'type': 'BEB_RETRY'},
            {'name': 'RL', 'type': 'RL'},
            {'name': 'RL + Retry', 'type': 'RL_RETRY'},
        ]
        
        results = {}
        
        for cfg in configs:
            print(f"  Running {cfg['name']}...")
            all_histories = []
            for seed in seed_list:
                sim = SimulationEngine(num_nodes=num_nodes, packet_prob=packet_prob, node_type=cfg['type'], duration=duration, seed=seed)
                metrics = sim.run()
                all_histories.append(metrics['history'])
            
            # Average histories
            avg_history = []
            if all_histories:
                num_points = len(all_histories[0])
                for i in range(num_points):
                    time_point = all_histories[0][i]['time']
                    avg_throughput = np.mean([h[i]['throughput'] for h in all_histories])
                    avg_collision = np.mean([h[i]['collision_rate'] for h in all_histories])
                    avg_history.append({
                        'time': time_point,
                        'throughput': avg_throughput,
                        'collision_rate': avg_collision
                    })
            results[cfg['name']] = avg_history
        
        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Throughput over time
        for cfg in configs:
            times = [h['time'] for h in results[cfg['name']]]
            throughputs = [h['throughput'] for h in results[cfg['name']]]
            linestyle = '--' if 'Retry' not in cfg['name'] else '-'
            ax1.plot(times, throughputs, label=cfg['name'], linestyle=linestyle, linewidth=2)
        
        ax1.set_xlabel('Time (Slots)', fontsize=12)
        ax1.set_ylabel('Throughput (Moving Avg)', fontsize=12)
        ax1.set_title('Convergence: Throughput Over Time', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Collision rate over time
        for cfg in configs:
            times = [h['time'] for h in results[cfg['name']]]
            collisions = [h['collision_rate'] for h in results[cfg['name']]]
            linestyle = '--' if 'Retry' not in cfg['name'] else '-'
            ax2.plot(times, collisions, label=cfg['name'], linestyle=linestyle, linewidth=2)
        
        ax2.set_xlabel('Time (Slots)', fontsize=12)
        ax2.set_ylabel('Collision Rate (Moving Avg)', fontsize=12)
        ax2.set_title('Convergence: Collision Rate Over Time', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/retry_convergence.png')
        print("Retry convergence plot saved.")

    def run_retry_heatmap(self, node_counts=[10, 50, 100, 200, 300, 500], packet_probs=[0.3, 0.5, 0.7, 0.9], duration=3000, seeds=10):
        print("Running Retry Heatmap Analysis...")
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        # We'll create heatmaps for BEB_RETRY and RL_RETRY showing dropped packet rate
        configs = [
            {'name': 'BEB + Retry', 'type': 'BEB_RETRY'},
            {'name': 'RL + Retry', 'type': 'RL_RETRY'},
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, cfg in enumerate(configs):
            print(f"  Computing heatmap for {cfg['name']}...")
            heatmap_data = np.zeros((len(packet_probs), len(node_counts)))
            
            for i, p in enumerate(packet_probs):
                for j, N in enumerate(node_counts):
                    dropped_rates = []
                    for seed in seed_list:
                        sim = SimulationEngine(num_nodes=N, packet_prob=p, node_type=cfg['type'], duration=duration, seed=seed)
                        metrics = sim.run()
                        dropped_rate = metrics['total_dropped'] / metrics['total_generated'] if metrics['total_generated'] > 0 else 0
                        dropped_rates.append(dropped_rate)
                    heatmap_data[i, j] = np.mean(dropped_rates)
            
            # Plot heatmap
            im = axes[idx].imshow(heatmap_data, cmap='YlOrRd', aspect='auto', origin='lower')
            axes[idx].set_xticks(range(len(node_counts)))
            axes[idx].set_yticks(range(len(packet_probs)))
            axes[idx].set_xticklabels(node_counts)
            axes[idx].set_yticklabels(packet_probs)
            axes[idx].set_xlabel('Number of Nodes', fontsize=12)
            axes[idx].set_ylabel('Packet Generation Probability', fontsize=12)
            axes[idx].set_title(f'{cfg["name"]}: Dropped Packet Rate', fontsize=13)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Dropped Packet Rate', fontsize=10)
            
            # Add text annotations
            for i in range(len(packet_probs)):
                for j in range(len(node_counts)):
                    axes[idx].text(j, i, f'{heatmap_data[i, j]:.5f}',
                                        ha="center", va="center", color="black", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/retry_heatmap.png')
        print("Retry heatmap plot saved.")

    def run_epsilon_comparison(self, duration=5000, num_nodes=50, packet_prob=0.7, seeds=10):
        print("Running Epsilon Comparison Experiment...")
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        # Define configurations with different epsilon strategies
        configs = [
            {'name': 'BEB', 'type': 'BEB', 'kwargs': {}},
            {'name': 'RL (eps=0.05)', 'type': 'RL', 'kwargs': {'epsilon': 0.05}},
            {'name': 'RL (eps=0.1)', 'type': 'RL', 'kwargs': {'epsilon': 0.1}},
            {'name': 'RL (eps=0.3)', 'type': 'RL', 'kwargs': {'epsilon': 0.3}},
            {'name': 'RL (eps=0.5)', 'type': 'RL', 'kwargs': {'epsilon': 0.5}},
            {'name': 'RL (Decay 0.001)', 'type': 'RL', 'kwargs': {'epsilon': 1.0, 'epsilon_decay': 0.001, 'epsilon_min': 0.01}},
            {'name': 'RL (Decay 0.005)', 'type': 'RL', 'kwargs': {'epsilon': 1.0, 'epsilon_decay': 0.005, 'epsilon_min': 0.01}},
        ]
        
        # Part 1: Performance comparison
        results = {cfg['name']: {'throughput': [], 'collision': [], 'pdr': []} for cfg in configs}
        
        for cfg in configs:
            print(f"  Testing {cfg['name']}...")
            throughputs = []
            collisions = []
            pdrs = []
            
            for seed in seed_list:
                sim = SimulationEngine(num_nodes=num_nodes, packet_prob=packet_prob, node_type=cfg['type'], duration=duration, seed=seed, **cfg['kwargs'])
                metrics = sim.run()
                throughputs.append(metrics['throughput'])
                collisions.append(metrics['collision_rate'])
                pdrs.append(metrics['pdr'])
                print(f"    Seed {seed}: Throughput={metrics['throughput']:.4f}, PDR={metrics['pdr']:.4f}")
            
            results[cfg['name']]['throughput'] = np.mean(throughputs)
            results[cfg['name']]['collision'] = np.mean(collisions)
            results[cfg['name']]['pdr'] = np.mean(pdrs)

        # Calculate Efficiency Score
        labels = [cfg['name'] for cfg in configs]
        throughput_means = [results[name]['throughput'] for name in labels]
        collision_means = [results[name]['collision'] for name in labels]
        pdr_means = [results[name]['pdr'] for name in labels]
        efficiency_scores = [results[name]['throughput'] * (1 - results[name]['collision']) for name in labels]
        
        for name, eff in zip(labels, efficiency_scores):
            print(f"  {name} - Throughput: {results[name]['throughput']:.4f}, Efficiency: {eff:.4f}")

        # Plot 1: Bar chart comparison
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        x = np.arange(len(labels))
        width = 0.2
        
        rects1 = ax1.bar(x - 1.5*width, throughput_means, width, label='Throughput', color='tab:blue', alpha=0.8)
        rects2 = ax1.bar(x - 0.5*width, collision_means, width, label='Collision Rate', color='tab:red', alpha=0.8)
        rects3 = ax1.bar(x + 0.5*width, pdr_means, width, label='PDR', color='tab:orange', alpha=0.8)
        rects4 = ax1.bar(x + 1.5*width, efficiency_scores, width, label='Efficiency', color='tab:green', alpha=0.8)
        
        ax1.set_ylabel('Metric Value', fontsize=12)
        ax1.set_title('Epsilon Strategy Comparison (N=50, p=0.7)', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20, ha='right')
        ax1.set_ylim(0, 1.0)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax1.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/epsilon_comparison.png')
        print("Epsilon comparison plot saved.")
        
        # Part 2: Convergence analysis for decay strategies
        print("  Running convergence analysis...")
        convergence_configs = [
            {'name': 'BEB', 'type': 'BEB', 'kwargs': {}},
            {'name': 'RL (Fixed eps=0.1)', 'type': 'RL', 'kwargs': {'epsilon': 0.1}},
            {'name': 'RL (Decay 0.001)', 'type': 'RL', 'kwargs': {'epsilon': 1.0, 'epsilon_decay': 0.001, 'epsilon_min': 0.01}},
            {'name': 'RL (Decay 0.005)', 'type': 'RL', 'kwargs': {'epsilon': 1.0, 'epsilon_decay': 0.005, 'epsilon_min': 0.01}},
        ]
        
        convergence_results = {}
        for cfg in convergence_configs:
            all_histories = []
            for seed in seed_list:
                sim = SimulationEngine(num_nodes=num_nodes, packet_prob=packet_prob, node_type=cfg['type'], duration=duration, seed=seed, **cfg['kwargs'])
                metrics = sim.run()
                all_histories.append(metrics['history'])
            
            # Average histories
            avg_history = []
            if all_histories:
                num_points = len(all_histories[0])
                for i in range(num_points):
                    time_point = all_histories[0][i]['time']
                    avg_throughput = np.mean([h[i]['throughput'] for h in all_histories])
                    avg_collision = np.mean([h[i]['collision_rate'] for h in all_histories])
                    avg_history.append({
                        'time': time_point,
                        'throughput': avg_throughput,
                        'collision_rate': avg_collision
                    })
            convergence_results[cfg['name']] = avg_history
        
        # Plot 2: Convergence over time
        fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(12, 10))
        
        for cfg in convergence_configs:
            times = [h['time'] for h in convergence_results[cfg['name']]]
            throughputs = [h['throughput'] for h in convergence_results[cfg['name']]]
            linestyle = '--' if cfg['type'] == 'BEB' else '-'
            ax2_1.plot(times, throughputs, label=cfg['name'], linestyle=linestyle, linewidth=2)
        
        ax2_1.set_xlabel('Time (Slots)', fontsize=12)
        ax2_1.set_ylabel('Throughput (Moving Avg)', fontsize=12)
        ax2_1.set_title('Epsilon Strategies: Throughput Convergence', fontsize=13)
        ax2_1.legend()
        ax2_1.grid(True, alpha=0.3)
        
        for cfg in convergence_configs:
            times = [h['time'] for h in convergence_results[cfg['name']]]
            collisions = [h['collision_rate'] for h in convergence_results[cfg['name']]]
            linestyle = '--' if cfg['type'] == 'BEB' else '-'
            ax2_2.plot(times, collisions, label=cfg['name'], linestyle=linestyle, linewidth=2)
        
        ax2_2.set_xlabel('Time (Slots)', fontsize=12)
        ax2_2.set_ylabel('Collision Rate (Moving Avg)', fontsize=12)
        ax2_2.set_title('Epsilon Strategies: Collision Rate Over Time', fontsize=13)
        ax2_2.legend()
        ax2_2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/epsilon_convergence.png')
        print("Epsilon convergence plot saved.")

    def run_optimized_scalability(self, node_counts=[10, 50, 100, 200, 500], duration=5000, seeds=10):
        print("Running Optimized Scalability Comparison...")
        print("Comparing: BEB vs Standard RL vs Best RL Configuration")
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        # Based on epsilon experiments, we'll test the best configurations
        configs = [
            {'name': 'BEB', 'type': 'BEB', 'kwargs': {}},
            {'name': 'RL (Standard eps=0.1)', 'type': 'RL', 'kwargs': {'epsilon': 0.1}},
            {'name': 'RL (Optimized)', 'type': 'RL', 'kwargs': {'epsilon': 1.0, 'epsilon_decay': 0.001, 'epsilon_min': 0.01}},
        ]
        
        results = {cfg['name']: {'throughput': [], 'pdr': [], 'collision': [], 'efficiency': []} for cfg in configs}
        
        raw_data = []
        averaged_data = []

        for N in node_counts:
            print(f"  Testing with {N} nodes...")
            for cfg in configs:
                throughputs = []
                pdrs = []
                collisions = []
                
                for seed in seed_list:
                    sim = SimulationEngine(num_nodes=N, packet_prob=0.5, node_type=cfg['type'], duration=duration, seed=seed, **cfg['kwargs'])
                    metrics = sim.run()
                    throughputs.append(metrics['throughput'])
                    pdrs.append(metrics['pdr'])
                    collisions.append(metrics['collision_rate'])
                    
                    raw_data.append({
                        'Experiment': 'OptimizedScalability',
                        'Nodes': N,
                        'Config': cfg['name'],
                        'Seed': seed,
                        'Throughput': metrics['throughput'],
                        'PDR': metrics['pdr'],
                        'CollisionRate': metrics['collision_rate'],
                        'Efficiency': metrics['throughput'] * (1 - metrics['collision_rate'])
                    })
                    print(f"      Seed {seed}: Throughput={metrics['throughput']:.4f}, Efficiency={metrics['throughput'] * (1 - metrics['collision_rate']):.4f}")
                
                avg_throughput = np.mean(throughputs)
                avg_pdr = np.mean(pdrs)
                avg_collision = np.mean(collisions)
                efficiency = avg_throughput * (1 - avg_collision)
                
                results[cfg['name']]['throughput'].append(avg_throughput)
                results[cfg['name']]['pdr'].append(avg_pdr)
                results[cfg['name']]['collision'].append(avg_collision)
                results[cfg['name']]['efficiency'].append(efficiency)
                
                averaged_data.append({
                    'Experiment': 'OptimizedScalability',
                    'Nodes': N,
                    'Config': cfg['name'],
                    'Avg_Throughput': avg_throughput,
                    'Avg_PDR': avg_pdr,
                    'Avg_CollisionRate': avg_collision,
                    'Avg_Efficiency': efficiency
                })
                
                print(f"    {cfg['name']}: Throughput={avg_throughput:.4f}, PDR={avg_pdr:.4f}, Efficiency={efficiency:.4f}")
        
        self.save_results_to_csv('optimized_scalability_raw.csv', raw_data)
        self.save_results_to_csv('optimized_scalability_averaged.csv', averaged_data)

        
    
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Throughput
        ax1 = fig.add_subplot(gs[0, 0])
        for cfg in configs:
            marker = 'o' if 'BEB' in cfg['name'] else ('s' if 'Standard' in cfg['name'] else '^')
            ax1.plot(node_counts, results[cfg['name']]['throughput'], marker=marker, label=cfg['name'], linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Nodes', fontsize=12)
        ax1.set_ylabel('Throughput', fontsize=12)
        ax1.set_title('Scalability: Throughput vs Network Density', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: PDR
        ax2 = fig.add_subplot(gs[0, 1])
        for cfg in configs:
            marker = 'o' if 'BEB' in cfg['name'] else ('s' if 'Standard' in cfg['name'] else '^')
            ax2.plot(node_counts, results[cfg['name']]['pdr'], marker=marker, label=cfg['name'], linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Nodes', fontsize=12)
        ax2.set_ylabel('Packet Delivery Ratio (PDR)', fontsize=12)
        ax2.set_title('Scalability: PDR vs Network Density', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Collision Rate
        ax3 = fig.add_subplot(gs[1, 0])
        for cfg in configs:
            marker = 'o' if 'BEB' in cfg['name'] else ('s' if 'Standard' in cfg['name'] else '^')
            ax3.plot(node_counts, results[cfg['name']]['collision'], marker=marker, label=cfg['name'], linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Nodes', fontsize=12)
        ax3.set_ylabel('Collision Rate', fontsize=12)
        ax3.set_title('Scalability: Collision Rate vs Network Density', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency Score
        ax4 = fig.add_subplot(gs[1, 1])
        for cfg in configs:
            marker = 'o' if 'BEB' in cfg['name'] else ('s' if 'Standard' in cfg['name'] else '^')
            ax4.plot(node_counts, results[cfg['name']]['efficiency'], marker=marker, label=cfg['name'], linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Nodes', fontsize=12)
        ax4.set_ylabel('Efficiency Score', fontsize=12)
        ax4.set_title('Scalability: Efficiency vs Network Density', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Optimized RL Performance: Comprehensive Scalability Analysis', fontsize=15, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/optimized_scalability.png', dpi=150, bbox_inches='tight')
        print("Optimized scalability plot saved.")
        
        # Print summary table
        print("\n=== Performance Summary ===")
        for N in node_counts:
            idx = node_counts.index(N)
            print(f"\nNodes: {N}")
            for cfg in configs:
                print(f"  {cfg['name']:25s}: Throughput={results[cfg['name']]['throughput'][idx]:.4f}, "
                      f"PDR={results[cfg['name']]['pdr'][idx]:.4f}, "
                      f"Efficiency={results[cfg['name']]['efficiency'][idx]:.4f}")

    def run_backoff_distribution(self, num_nodes=200, packet_prob=0.9, duration=2000, seeds=10):
        print("Running Backoff Distribution Experiment...")
        
        seed_list = seeds if isinstance(seeds, list) else range(42, 42 + seeds)
        
        beb_cws = []
        rl_cws = []
            
        for seed in seed_list:
            # BEB
            sim_beb = SimulationEngine(num_nodes=num_nodes, packet_prob=packet_prob, node_type='BEB', duration=duration, seed=seed)
            sim_beb.run()
            beb_cws.extend([node.cw for node in sim_beb.nodes])
            
            # RL
            sim_rl = SimulationEngine(num_nodes=num_nodes, packet_prob=packet_prob, node_type='RL', duration=duration, seed=seed)
            sim_rl.run()
            rl_cws.extend([node.cw for node in sim_rl.nodes])
        
        # Calculate statistics
        beb_mean = np.mean(beb_cws)
        rl_mean = np.mean(rl_cws)
        print(f"  BEB Mean CW: {beb_mean:.2f}")
        print(f"  RL Mean CW: {rl_mean:.2f}")
        
        # Plot Histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        
        bins = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
        ax1.hist(beb_cws, bins=bins, color='tab:blue', alpha=0.7, edgecolor='black')
        ax1.set_xscale('log', base=2)
        ax1.set_xticks(bins[:-1])
        ax1.set_xticklabels(bins[:-1])
        ax1.set_title(f'BEB Backoff Distribution (Mean: {beb_mean:.0f})')
        ax1.set_xlabel('Contention Window Size')
        ax1.set_ylabel('Number of Nodes')
        
        ax2.hist(rl_cws, bins=bins, color='tab:orange', alpha=0.7, edgecolor='black')
        ax2.set_xscale('log', base=2)
        ax2.set_xticks(bins[:-1])
        ax2.set_xticklabels(bins[:-1])
        ax2.set_title(f'RL Backoff Distribution (Mean: {rl_mean:.0f})')
        ax2.set_xlabel('Contention Window Size')
        
        plt.suptitle(f'Backoff Window Distribution at Saturation (N={num_nodes})', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/backoff_distribution.png')
        print("Backoff distribution plot saved.")

if __name__ == '__main__':
    # Simple test
    runner = ExperimentRunner()
    runner.run_backoff_distribution()
