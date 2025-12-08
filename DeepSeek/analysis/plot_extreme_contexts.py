#!/usr/bin/env python3
"""
plot_extreme_contexts.py - Generate side-by-side cycle distribution plots 
comparing extreme context sizes (128 vs 2048) to visualize side-channel signal

This plot demonstrates the strongest proof of a side channel:
- Larger context → larger working set → more cache/TLB pressure → higher cycles
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_run_data(run_dir: Path):
    """Load probe CSV and metadata for a single run."""
    meta_path = run_dir / "meta.json"
    probe_path = run_dir / "attacker_stdout.txt"
    
    if not meta_path.exists() or not probe_path.exists():
        return None, None
    
    with meta_path.open("r") as f:
        meta = json.load(f)
    
    try:
        probe_df = pd.read_csv(probe_path)
    except Exception as e:
        print(f"Warning: Failed to load {probe_path}: {e}")
        return None, None
    
    return probe_df, meta

def collect_extreme_context_data(logs_dir: Path, extreme_contexts=[128, 2048]):
    """Collect cycle data for extreme context sizes."""
    runs_dir = logs_dir / "runs"
    
    context_data = {ctx: [] for ctx in extreme_contexts}
    
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        probe_df, meta = load_run_data(run_dir)
        
        if probe_df is None or meta is None:
            continue
        
        # Extract context size from metadata
        ctx = meta.get('ctx')
        
        if ctx in extreme_contexts:
            cycles = probe_df['cycles'].values
            # Filter out extreme outliers (>10M cycles) for better visualization
            cycles = cycles[cycles < 10_000_000]
            context_data[ctx].extend(cycles)
            print(f"Loaded {len(cycles)} cycles for ctx={ctx} from {run_dir.name}")
    
    return context_data

def plot_extreme_contexts(context_data, output_path, model_name="Model"):
    """Create side-by-side histogram/KDE plot comparing extreme contexts."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72']  # Blue for small, Red/Purple for large
    extreme_contexts = sorted(context_data.keys())
    
    for i, ctx in enumerate(extreme_contexts):
        cycles = context_data[ctx]
        
        if len(cycles) == 0:
            print(f"Warning: No data for ctx={ctx}")
            continue
        
        # Plot histogram with transparency
        ax.hist(cycles, bins=50, alpha=0.5, color=colors[i], 
                label=f'Context = {ctx} tokens', density=True, edgecolor='black', linewidth=0.5)
        
        # Add KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(cycles)
        x_range = np.linspace(min(cycles), max(cycles), 200)
        ax.plot(x_range, kde(x_range), color=colors[i], linewidth=2.5, alpha=0.8)
        
        # Print statistics
        print(f"\nContext = {ctx} tokens:")
        print(f"  Mean: {np.mean(cycles):,.0f} cycles")
        print(f"  Median: {np.median(cycles):,.0f} cycles")
        print(f"  Std Dev: {np.std(cycles):,.0f} cycles")
        print(f"  Min: {np.min(cycles):,.0f} cycles")
        print(f"  Max: {np.max(cycles):,.0f} cycles")
        print(f"  P99: {np.percentile(cycles, 99):,.0f} cycles")
    
    # Calculate separation metrics
    if len(extreme_contexts) == 2:
        ctx_small, ctx_large = extreme_contexts
        cycles_small = context_data[ctx_small]
        cycles_large = context_data[ctx_large]
        
        if len(cycles_small) > 0 and len(cycles_large) > 0:
            mean_diff = np.mean(cycles_large) - np.mean(cycles_small)
            median_diff = np.median(cycles_large) - np.median(cycles_small)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(cycles_small)**2 + np.std(cycles_large)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"SEPARATION METRICS:")
            print(f"  Mean difference: {mean_diff:+,.0f} cycles")
            print(f"  Median difference: {median_diff:+,.0f} cycles")
            print(f"  Cohen's d (effect size): {cohens_d:.3f}")
            print(f"  Relative increase: {(mean_diff/np.mean(cycles_small)*100):.1f}%")
            print(f"{'='*60}")
    
    ax.set_xlabel('Attacker Probe Cycles', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title(f'Side-Channel Signal: Extreme Context Sizes ({model_name})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis with thousands separator
    ax.ticklabel_format(style='plain', axis='x')
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate extreme context comparison plots')
    parser.add_argument('--logs-dir', type=Path, required=True,
                        help='Path to logs directory')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for figures')
    parser.add_argument('--model-name', type=str, default='Model',
                        help='Model name for plot title')
    parser.add_argument('--contexts', type=int, nargs='+', default=[128, 2048],
                        help='Extreme context sizes to compare (default: 128 2048)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.logs_dir.parent / "analysis" / "figs"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {args.logs_dir}")
    print(f"Comparing extreme contexts: {args.contexts}")
    
    context_data = collect_extreme_context_data(args.logs_dir, args.contexts)
    
    # Check if we have data
    if all(len(cycles) == 0 for cycles in context_data.values()):
        print("Error: No data found for any of the specified contexts")
        return
    
    output_path = args.output_dir / f"extreme_contexts_comparison_{args.model_name.lower().replace(' ', '_')}.png"
    plot_extreme_contexts(context_data, output_path, args.model_name)

if __name__ == "__main__":
    main()
