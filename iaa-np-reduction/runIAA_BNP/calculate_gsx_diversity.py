#!/usr/bin/env python3
"""
Calculate GSx Diversity Metric (Eq. 6) on z-score standardised features.

Definition:
    div_GSx = (1 / N) * sum_{i=1}^N min_{j != i} d(R_i, R_j)

Where:
    - R: reduced dataset
    - N: number of instances in R
    - d(x, y): Euclidean distance between standardised feature vectors x and y

The features are always z-score standardised (StandardScaler). When a
domain-filtered dataset is supplied, the scaler is fitted on it (the domain is
the reference distribution) and the IAA-reduced dataset is transformed with the
same scaler. Otherwise the scaler is fitted on the IAA-reduced dataset itself.

It supports:
    - Evaluating the IAA-reduced dataset (1,800 samples)
    - Evaluating the domain-driven reduced dataset (300,889 samples) in memory-efficient chunks
"""

import argparse
import os
import pickle
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def compute_diversity(X, n_neighbors=2, batch_size=10000, n_jobs=-1):
    """
    Compute mean minimum distance to nearest neighbor (excluding self).
    Uses batch querying for large datasets to keep memory consumption low.
    """
    n_samples = X.shape[0]
    print(f"  Fitting NearestNeighbors on {n_samples:,} samples with {X.shape[1]} features...")
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=n_jobs)
    nn.fit(X)

    min_dists = []
    if n_samples <= batch_size:
        dists, _ = nn.kneighbors(X)
        min_dists = dists[:, 1]
    else:
        print(f"  Querying in batches of {batch_size:,}...")
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = X[start_idx:end_idx]
            dists, _ = nn.kneighbors(batch)
            min_dists.append(dists[:, 1])
            sys.stdout.write(f"\r  Processed {end_idx:,} / {n_samples:,} samples")
            sys.stdout.flush()
        print()
        min_dists = np.concatenate(min_dists)

    return float(np.mean(min_dists)), float(np.std(min_dists))


def extract_numeric_features(df):
    """Extract numeric feature array and feature names."""
    num_cols = [c for c in df.columns if df[c].dtype != "object"]
    return df[num_cols].values.astype(np.float64), num_cols


def main():
    parser = argparse.ArgumentParser(description="Compute GSx diversity metric on z-score standardised features.")
    parser.add_argument(
        "--iaa_path",
        type=str,
        default="iteraa/docs/reducedNPdata/reducedBNP_nanoparticle_data.pkl",
        help="Path to IAA-reduced dataframe pickle.",
    )
    parser.add_argument(
        "--domain_path",
        type=str,
        default=None,
        help="Path to domain-filtered dataframe pickle (e.g. BNPFeatsDF.pkl).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for nearest neighbor search on large datasets.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel CPU jobs (-1 for all cores).",
    )
    args = parser.parse_args()

    # 1. Load IAA dataset
    if not os.path.exists(args.iaa_path):
        print(f"Error: IAA file not found at {args.iaa_path}")
        sys.exit(1)

    print(f"Loading IAA-reduced dataset from {args.iaa_path}...")
    with open(args.iaa_path, "rb") as f:
        df_iaa = pickle.load(f)
    X_iaa_raw, cols_iaa = extract_numeric_features(df_iaa)
    print(f"IAA dataset: {X_iaa_raw.shape[0]:,} samples, {X_iaa_raw.shape[1]} features")

    # 2. Check Domain dataset
    X_dom_raw = None
    if args.domain_path and os.path.exists(args.domain_path):
        print(f"\nLoading domain-filtered dataset from {args.domain_path}...")
        with open(args.domain_path, "rb") as f:
            df_dom = pickle.load(f)
        X_dom_raw, cols_dom = extract_numeric_features(df_dom)
        print(f"Domain dataset: {X_dom_raw.shape[0]:,} samples, {X_dom_raw.shape[1]} features")
    else:
        print("\nNote: Domain-filtered dataset not provided or not found.")
        print("Computing standardised diversity metric for IAA-reduced dataset only.")

    results = {}

    # --- Z-score standardised features ---
    print("\n" + "=" * 50)
    print("Z-SCORE STANDARDISED FEATURES")
    print("=" * 50)
    if X_dom_raw is not None:
        # Fit scaler on domain (reference) data, then transform both
        print("  Using shared scaler fitted on domain dataset...")
        scaler = StandardScaler()
        X_dom_std = scaler.fit_transform(X_dom_raw)
        X_iaa_std = scaler.transform(X_iaa_raw)
        dom_std_mean, dom_std_std = compute_diversity(X_dom_std, batch_size=args.batch_size, n_jobs=args.n_jobs)
        iaa_std_mean, iaa_std_std = compute_diversity(X_iaa_std, batch_size=args.batch_size, n_jobs=args.n_jobs)
        results["dom_std"] = dom_std_mean
        results["iaa_std"] = iaa_std_mean
        print(f"IAA div_GSx (standardised):    {iaa_std_mean:.4f} +/- {iaa_std_std:.4f}")
        print(f"Domain div_GSx (standardised): {dom_std_mean:.4f} +/- {dom_std_std:.4f}")
        print(f"Ratio (IAA / Domain): {iaa_std_mean / dom_std_mean:.2f}x")
    else:
        scaler = StandardScaler()
        X_iaa_std = scaler.fit_transform(X_iaa_raw)
        iaa_std_mean, iaa_std_std = compute_diversity(X_iaa_std, batch_size=args.batch_size, n_jobs=args.n_jobs)
        results["iaa_std"] = iaa_std_mean
        print(f"IAA div_GSx (standardised): {iaa_std_mean:.4f} +/- {iaa_std_std:.4f}")

    print("\n" + "=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:10s}: {v:.4f}")


if __name__ == "__main__":
    main()
