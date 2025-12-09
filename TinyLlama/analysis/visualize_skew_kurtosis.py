#!/usr/bin/env python3
"""
Visualize skew and kurtosis distributions to understand their impact on classification.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

# Load the statistics
stats_file = Path('analysis/figs/run_statistics.csv')
stats = pd.read_csv(stats_file)

# Create output directory
output_dir = Path('analysis/figs')
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create a 2x3 subplot layout
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Skewness and Kurtosis Distributions by Attack Target', fontsize=16, fontweight='bold')

# 1. Skew by Context Size
if 'ctx' in stats.columns and 'skew' in stats.columns:
    ax = axes[0, 0]
    stats_ctx = stats[stats['ctx'].notna()]
    sns.boxplot(data=stats_ctx, x='ctx', y='skew', ax=ax, palette='Set2')
    ax.set_title('Skewness by Context Size', fontweight='bold')
    ax.set_xlabel('Context Size (tokens)')
    ax.set_ylabel('Skewness')
    ax.grid(True, alpha=0.3)

# 2. Kurtosis by Context Size
if 'ctx' in stats.columns and 'kurtosis' in stats.columns:
    ax = axes[0, 1]
    stats_ctx = stats[stats['ctx'].notna()]
    sns.boxplot(data=stats_ctx, x='ctx', y='kurtosis', ax=ax, palette='Set2')
    ax.set_title('Kurtosis by Context Size', fontweight='bold')
    ax.set_xlabel('Context Size (tokens)')
    ax.set_ylabel('Kurtosis')
    ax.grid(True, alpha=0.3)

# 3. Skew vs Kurtosis scatter for Context
if 'ctx' in stats.columns:
    ax = axes[0, 2]
    stats_ctx = stats[stats['ctx'].notna()]
    for ctx_val in stats_ctx['ctx'].unique():
        subset = stats_ctx[stats_ctx['ctx'] == ctx_val]
        ax.scatter(subset['skew'], subset['kurtosis'], label=f'{int(ctx_val)}', alpha=0.6, s=50)
    ax.set_title('Skew vs Kurtosis (Context)', fontweight='bold')
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Kurtosis')
    ax.legend(title='Context')
    ax.grid(True, alpha=0.3)

# 4. Skew by Decoding Strategy
if 'decoding' in stats.columns and 'skew' in stats.columns:
    ax = axes[1, 0]
    stats_dec = stats[stats['decoding'].notna()]
    sns.boxplot(data=stats_dec, x='decoding', y='skew', ax=ax, palette='Set3')
    ax.set_title('Skewness by Decoding Strategy', fontweight='bold')
    ax.set_xlabel('Decoding Strategy')
    ax.set_ylabel('Skewness')
    ax.grid(True, alpha=0.3)

# 5. Kurtosis by Decoding Strategy
if 'decoding' in stats.columns and 'kurtosis' in stats.columns:
    ax = axes[1, 1]
    stats_dec = stats[stats['decoding'].notna()]
    sns.boxplot(data=stats_dec, x='decoding', y='kurtosis', ax=ax, palette='Set3')
    ax.set_title('Kurtosis by Decoding Strategy', fontweight='bold')
    ax.set_xlabel('Decoding Strategy')
    ax.set_ylabel('Kurtosis')
    ax.grid(True, alpha=0.3)

# 6. Skew vs Kurtosis scatter for Decoding
if 'decoding' in stats.columns:
    ax = axes[1, 2]
    stats_dec = stats[stats['decoding'].notna()]
    for dec_val in stats_dec['decoding'].unique():
        subset = stats_dec[stats_dec['decoding'] == dec_val]
        ax.scatter(subset['skew'], subset['kurtosis'], label=dec_val, alpha=0.6, s=50)
    ax.set_title('Skew vs Kurtosis (Decoding)', fontweight='bold')
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Kurtosis')
    ax.legend(title='Decoding')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'skew_kurtosis_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'skew_kurtosis_analysis.png'}")
plt.close()

# Create separate detailed plot for prompt semantics
if 'prompt_label' in stats.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Skewness and Kurtosis for Prompt Semantics Classification', fontsize=16, fontweight='bold')
    
    stats_prompt = stats[stats['prompt_label'].notna()]
    
    # Filter out 'custom' to focus on semantic categories
    stats_semantic = stats_prompt[stats_prompt['prompt_label'].isin(['math', 'code', 'nl'])]
    
    # 1. Skew distribution
    ax = axes[0]
    sns.violinplot(data=stats_semantic, x='prompt_label', y='skew', ax=ax, palette='viridis')
    ax.set_title('Skewness Distribution', fontweight='bold')
    ax.set_xlabel('Prompt Type')
    ax.set_ylabel('Skewness')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Kurtosis distribution
    ax = axes[1]
    sns.violinplot(data=stats_semantic, x='prompt_label', y='kurtosis', ax=ax, palette='viridis')
    ax.set_title('Kurtosis Distribution', fontweight='bold')
    ax.set_xlabel('Prompt Type')
    ax.set_ylabel('Kurtosis')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Skew vs Kurtosis scatter
    ax = axes[2]
    for prompt_type in stats_semantic['prompt_label'].unique():
        subset = stats_semantic[stats_semantic['prompt_label'] == prompt_type]
        ax.scatter(subset['skew'], subset['kurtosis'], label=prompt_type, alpha=0.7, s=100, edgecolors='black', linewidths=0.5)
    ax.set_title('Skew vs Kurtosis Scatter', fontweight='bold')
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Kurtosis')
    ax.legend(title='Prompt Type')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prompt_semantics_skew_kurtosis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'prompt_semantics_skew_kurtosis.png'}")
    plt.close()

# Print summary statistics
print("\n=== Summary Statistics ===\n")
print("Skewness by Context Size:")
if 'ctx' in stats.columns:
    print(stats.groupby('ctx')['skew'].describe())

print("\nKurtosis by Context Size:")
if 'ctx' in stats.columns:
    print(stats.groupby('ctx')['kurtosis'].describe())

print("\nSkewness by Decoding Strategy:")
if 'decoding' in stats.columns:
    print(stats.groupby('decoding')['skew'].describe())

print("\nKurtosis by Decoding Strategy:")
if 'decoding' in stats.columns:
    print(stats.groupby('decoding')['kurtosis'].describe())

print("\nSkewness by Prompt Type:")
if 'prompt_label' in stats.columns:
    print(stats.groupby('prompt_label')['skew'].describe())

print("\nKurtosis by Prompt Type:")
if 'prompt_label' in stats.columns:
    print(stats.groupby('prompt_label')['kurtosis'].describe())

print("\nVisualization complete!")
