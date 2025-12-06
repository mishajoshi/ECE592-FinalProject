#!/usr/bin/env python3
"""
analysis.py - Analysis script for SMT contention experiment data

Loads probe CSV files and metadata, computes statistics, generates visualizations,
and trains classifiers for membership detection and model fingerprinting.

Usage:
    python3 analysis.py --logs-dir logs/runs [--output-dir analysis/figs]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_run_data(run_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load probe CSV and metadata for a single run."""
    meta_path = run_dir / "meta.json"
    probe_path = run_dir / "attacker_stdout.txt"
    
    if not meta_path.exists() or not probe_path.exists():
        return None, None
    
    with meta_path.open("r") as f:
        meta = json.load(f)
    
    # Load probe CSV
    try:
        probe_df = pd.read_csv(probe_path)
    except Exception as e:
        print(f"Warning: Failed to load {probe_path}: {e}")
        return None, None
    
    return probe_df, meta


def load_all_runs(logs_dir: Path, warmup_discard: int = 500) -> pd.DataFrame:
    """Load all runs from logs directory and combine into single DataFrame."""
    runs_dir = logs_dir / "runs"
    
    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist")
        return pd.DataFrame()
    
    all_data = []
    
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        
        probe_df, meta = load_run_data(run_path)
        if probe_df is None or meta is None:
            continue
        
        # Discard warmup iterations
        probe_df = probe_df[probe_df['iter'] >= warmup_discard].copy()
        
        # Add metadata columns
        for key in ['run_id', 'probe', 'model_tag', 'quant', 'ctx', 'npredict', 
                    'decoding', 'temp', 'top_k', 'top_p', 'seed', 'repeat_idx', 'victim_cpu', 'attacker_cpu', 'prompt_label', 'prompt']:
            if key in meta:
                probe_df[key] = meta[key]
        
        all_data.append(probe_df)
    
    if not all_data:
        print("Warning: No valid run data found")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(all_data)} runs with {len(combined)} total measurements")
    return combined


def compute_run_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-run statistics (mean, median, std, percentiles)."""
    stats = df.groupby('run_id')['cycles'].agg([
        'mean', 'median', 'std', 'min', 'max',
        ('p10', lambda x: np.percentile(x, 10)),
        ('p90', lambda x: np.percentile(x, 90)),
        ('p99', lambda x: np.percentile(x, 99)),
        ('skew', lambda x: x.skew()),
        ('kurtosis', lambda x: x.kurt()),
        'count'
    ]).reset_index()
    
    # Add metadata back
    meta_cols = [
        'probe', 'model_tag', 'quant', 'ctx', 'npredict', 'decoding', 'temp',
        'repeat_idx', 'prompt_label', 'prompt', 'top_k', 'top_p', 'victim_cpu', 'attacker_cpu'
    ]
    for col in meta_cols:
        if col in df.columns:
            stats = stats.merge(
                df[['run_id', col]].drop_duplicates(),
                on='run_id',
                how='left'
            )
    
    return stats


def plot_cycle_distributions(df: pd.DataFrame, output_dir: Path, probe: str = None):
    """Plot histograms and boxplots of cycle distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if probe:
        df = df[df['probe'] == probe]
        title_suffix = f" - {probe}"
        file_suffix = f"_{probe}"
    else:
        title_suffix = ""
        file_suffix = ""
    
    # Histogram
    plt.figure(figsize=(10, 6))
    if 'quant' in df.columns:
        for quant in df['quant'].unique():
            subset = df[df['quant'] == quant]
            plt.hist(subset['cycles'], bins=50, alpha=0.5, label=f'{quant}')
        plt.legend()
    else:
        plt.hist(df['cycles'], bins=50, alpha=0.7)
    
    plt.xlabel('Cycles')
    plt.ylabel('Frequency')
    plt.title(f'Cycle Distribution{title_suffix}')
    plt.tight_layout()
    plt.savefig(output_dir / f'hist_cycles{file_suffix}.png', dpi=150)
    plt.close()
    
    # Boxplot by quantization
    if 'quant' in df.columns and len(df['quant'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        df.boxplot(column='cycles', by='quant', figsize=(10, 6))
        plt.suptitle('')
        plt.title(f'Cycles by Quantization{title_suffix}')
        plt.xlabel('Quantization')
        plt.ylabel('Cycles')
        plt.tight_layout()
        plt.savefig(output_dir / f'boxplot_quant{file_suffix}.png', dpi=150)
        plt.close()
    
    # Boxplot by context size
    if 'ctx' in df.columns and len(df['ctx'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        df.boxplot(column='cycles', by='ctx', figsize=(10, 6))
        plt.suptitle('')
        plt.title(f'Cycles by Context Size{title_suffix}')
        plt.xlabel('Context Size')
        plt.ylabel('Cycles')
        plt.tight_layout()
        plt.savefig(output_dir / f'boxplot_ctx{file_suffix}.png', dpi=150)
        plt.close()


def plot_pca(stats: pd.DataFrame, output_dir: Path):
    """Plot PCA scatter of run statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select numeric features
    feature_cols = ['mean', 'median', 'std', 'p10', 'p90', 'p99']
    X = stats[feature_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if 'quant' in stats.columns:
        for quant in stats['quant'].unique():
            mask = stats['quant'] == quant
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.7, label=quant, s=50)
        plt.legend()
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA of Run Statistics')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_runs.png', dpi=150)
    plt.close()


def train_classifier(stats: pd.DataFrame, target_col: str, output_dir: Path):
    """Train classifier for membership detection or fingerprinting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if target_col not in stats.columns:
        print(f"Warning: Target column '{target_col}' not found, skipping classifier")
        return
    
    # Select features and target
    feature_cols = ['mean', 'median', 'std', 'p10', 'p90', 'p99', 'skew', 'kurtosis']
    X = stats[feature_cols].values
    #y = stats[target_col].values
    # Handle continuous floats like temperature by converting to categorical strings
    if target_col == "temp":
        y = stats[target_col].astype(str).values
    else:
        y = stats[target_col].values

    
    # Skip if only one class
    if len(np.unique(y)) < 2:
        print(f"Warning: Only one class in '{target_col}', skipping classifier")
        return
    
    # Convert continuous values (0.0, 0.5, 0.8) to categorical strings
    if target_col == "temp":
        y = stats[target_col].astype(str).values
    else:
        y = stats[target_col].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== Classifier for '{target_col}' ===")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {target_col}')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_{target_col}.png', dpi=150)
    plt.close()
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance)
    
    plt.figure(figsize=(8, 5))
    plt.barh(importance['feature'], importance['importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {target_col}')
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_importance_{target_col}.png', dpi=150)
    plt.close()


def build_pht_btb_prompt_dataset(stats: pd.DataFrame) -> pd.DataFrame:
    """Create a fused feature table using both PHT and BTB runs for the same prompt/repeat."""
    required_cols = {'probe', 'prompt_label', 'repeat_idx'}
    if not required_cols.issubset(stats.columns):
        return pd.DataFrame()

    feature_cols = ['mean', 'median', 'std', 'p10', 'p90', 'p99', 'skew', 'kurtosis']
    merge_cols = [c for c in ['prompt_label', 'repeat_idx', 'ctx', 'npredict', 'decoding', 'temp', 'victim_cpu', 'attacker_cpu'] if c in stats.columns]

    pht = stats[stats['probe'] == 'pht'].copy()
    btb = stats[stats['probe'] == 'btb'].copy()
    if pht.empty or btb.empty:
        return pd.DataFrame()

    pht = pht[merge_cols + feature_cols].drop_duplicates(subset=merge_cols)
    btb = btb[merge_cols + feature_cols].drop_duplicates(subset=merge_cols)

    pht = pht.rename(columns={c: f"{c}_pht" for c in feature_cols})
    btb = btb.rename(columns={c: f"{c}_btb" for c in feature_cols})

    fused = pht.merge(btb, on=merge_cols, how='inner')
    return fused


def train_prompt_classifier_pht_btb(fused: pd.DataFrame, output_dir: Path):
    """Train prompt-label classifier using fused PHT+BTB features."""
    if fused.empty:
        print("Skipping fused PHT+BTB classifier (no paired runs found).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    base_features = ['mean', 'median', 'std', 'p10', 'p90', 'p99', 'skew', 'kurtosis']
    feature_cols = [f"{f}_pht" for f in base_features] + [f"{f}_btb" for f in base_features]

    # Add delta features to capture differential behaviour between probes
    for f in base_features:
        delta_col = f"{f}_delta"
        fused[delta_col] = fused[f"{f}_pht"] - fused[f"{f}_btb"]
        feature_cols.append(delta_col)

    X = fused[feature_cols].values
    y = fused['prompt_label'].values

    if len(np.unique(y)) < 2:
        print("Skipping fused PHT+BTB classifier (only one prompt class).")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, multi_class='auto')
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== Fused PHT+BTB Prompt Classifier ===")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - prompt_label (PHT+BTB)')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_prompt_label_pht_btb.png', dpi=150)
    plt.close()

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.coef_.mean(axis=0)
    })
    importance['abs_importance'] = importance['importance'].abs()
    importance = importance.sort_values('abs_importance', ascending=False)

    print("\nTop features (absolute weight):")
    print(importance[['feature', 'abs_importance']].head(12))

    plt.figure(figsize=(9, 6))
    plt.barh(importance['feature'].head(12), importance['abs_importance'].head(12))
    plt.xlabel('Absolute Weight')
    plt.title('Top Features - prompt_label (PHT+BTB)')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_prompt_label_pht_btb.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze SMT contention experiment data')
    parser.add_argument('--logs-dir', type=str, default='logs',
                        help='Path to logs directory')
    parser.add_argument('--output-dir', type=str, default='analysis/figs',
                        help='Output directory for figures')
    parser.add_argument('--warmup-discard', type=int, default=500,
                        help='Number of warmup iterations to discard')
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    
    print("Loading data...")
    df = load_all_runs(logs_dir, warmup_discard=args.warmup_discard)
    
    if df.empty:
        print("No data loaded. Exiting.")
        return
    
    print("\nComputing statistics...")
    stats = compute_run_statistics(df)
    
    # Save summary statistics
    output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_dir / 'run_statistics.csv', index=False)
    print(f"Saved statistics to {output_dir / 'run_statistics.csv'}")
    
    print("\nGenerating visualizations...")
    
    # Overall distributions
    plot_cycle_distributions(df, output_dir)
    
    # Per-probe distributions
    if 'probe' in df.columns:
        for probe in df['probe'].unique():
            plot_cycle_distributions(df, output_dir, probe=probe)
    
    # PCA
    plot_pca(stats, output_dir)
    
    # Train classifiers
    print("\nTraining classifiers...")
    
    if 'quant' in stats.columns:
        train_classifier(stats, 'quant', output_dir)
    
    if 'ctx' in stats.columns:
        train_classifier(stats, 'ctx', output_dir)
    
    if 'decoding' in stats.columns:
        train_classifier(stats, 'decoding', output_dir)

    if 'temp' in stats.columns:
        train_classifier(stats, 'temp', output_dir)

    if 'prompt_label' in stats.columns:
        train_classifier(stats, 'prompt_label', output_dir)

    fused_prompt = build_pht_btb_prompt_dataset(stats)
    train_prompt_classifier_pht_btb(fused_prompt, output_dir)



    # if 'npredict' in stats.columns:
    #     train_classifier(stats, 'npredict', output_dir)

    # if 'top_k' in stats.columns:
    #     train_classifier(stats, 'top_k', output_dir)


    
    print(f"\nAnalysis complete! Figures saved to {output_dir}")


if __name__ == '__main__':
    main()
