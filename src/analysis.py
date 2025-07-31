import os
import re
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

###############################################################################
# CONSTANTS & CONFIGURATION
###############################################################################

DEFAULT_RESULTS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                   "outputs", "results")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "outputs", "analysis")

# Target densities to surface in tables
DENSITY_POINTS = [10, 50, 100, 250, 500]

###############################################################################
# TEXT-PROCESSING HELPERS
###############################################################################

def _tokenise_constraints(cell: str) -> List[str]:
    """Split the constraint column (comma-separated) into a clean list."""
    return [c.strip().lower() for c in cell.split(',') if c.strip()]


def _analyse_constraint(term: str, text: str) -> str:
    """Classify a constraint term relative to text.

    Returns one of {"exact", "modification", "omission"}.
    """
    # Exact match
    if re.search(rf"\b{re.escape(term)}\b", text):
        return "exact"

    # Modified (≥80 % prefix present)
    min_len = max(3, int(len(term) * 0.8))
    prefix = re.escape(term[:min_len])
    if re.search(rf"\b{prefix}[a-z]*\b", text):
        return "modification"

    return "omission"


def _evaluate_row(row) -> Tuple[int, int, int]:
    """Return counts of (exact, modification, omission) for a dataframe row."""
    text = str(row['generated_text']).lower()
    constraints = _tokenise_constraints(row['constraints'])

    exact = modification = omission = 0
    for term in constraints:
        outcome = _analyse_constraint(term, text)
        if outcome == "exact":
            exact += 1
        elif outcome == "modification":
            modification += 1
        else:
            omission += 1
    return exact, modification, omission

###############################################################################
# PRIMARY ANALYSES
###############################################################################

def compute_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute all required metrics and return them as DataFrames."""

    # Precompute per-sample results ------------------------------------------------
    per_sample = []
    for _, row in df.iterrows():
        exact, modif, omit = _evaluate_row(row)
        total = exact + modif + omit or 1  # guard against divide by zero for accuracy

        accuracy_pct = exact / total * 100.0
        # Compute error percentages: relative to total errors (omit + modif)
        error_count = modif + omit or 1  # guard against divide by zero for error rates
        omission_pct = omit / error_count * 100.0
        modification_pct = modif / error_count * 100.0

        # Recency effect (early vs late thirds)
        constraints = _tokenise_constraints(row['constraints'])
        recency_ratio = np.nan
        if len(constraints) >= 3:
            bucket = max(1, len(constraints) // 3)
            early = constraints[:bucket]
            late = constraints[-bucket:]

            early_errors = sum(1 for t in early if not re.search(rf"\b{re.escape(t)}\b", row['generated_text'].lower()))
            late_errors = sum(1 for t in late if not re.search(rf"\b{re.escape(t)}\b", row['generated_text'].lower()))
            early_rate = early_errors / len(early) if early else 0
            late_rate = late_errors / len(late) if late else 0
            if early_rate > 0:
                recency_ratio = late_rate / early_rate

        sample_dict = {
            'model': row['model'],
            'num_rules': int(row['num_rules']),
            'seed': row['seed'],
            'accuracy': accuracy_pct,
            'omission_pct': omission_pct,
            'modification_pct': modification_pct,
            'recency_ratio': recency_ratio,
        }
        # Latency field (optional)
        if 'latency_seconds' in row:
            sample_dict['latency'] = float(row['latency_seconds']) if not pd.isna(row['latency_seconds']) else np.nan
        per_sample.append(sample_dict)

    per_sample_df = pd.DataFrame(per_sample)

    # Accuracy per num_rules ------------------------------------------------------
    acc_df = per_sample_df.groupby(['model', 'num_rules'])['accuracy'].mean().reset_index()

    # Accuracy standard deviation per num_rules --------------------------------
    var_df = per_sample_df.groupby(['model', 'num_rules'])['accuracy'].std().reset_index()
    # Rename column to indicate std deviation in percentage points
    var_df = var_df.rename(columns={'accuracy': 'accuracy_std'})

    # Failure modes per num_rules -------------------------------------------------
    failure_df = per_sample_df.groupby(['model', 'num_rules']).agg({
        'omission_pct': 'mean',
        'modification_pct': 'mean'
    }).reset_index()

    # Recency effect per num_rules ------------------------------------------------
    recency_df = per_sample_df.groupby(['model', 'num_rules'])['recency_ratio'].mean().reset_index()

    return {
        'accuracy': acc_df,
        'variance': var_df,
        'failure_modes': failure_df,
        'recency': recency_df,
        'per_sample': per_sample_df  # keep for threshold analysis
    }

###############################################################################
# MAIN ENTRYPOINT
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Full analysis of Density-Bench generations.")
    parser.add_argument('--input', type=str, help='Path to generations.csv file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save analysis outputs')
    args = parser.parse_args()

    # Resolve paths -----------------------------------------------------------
    generations_file = args.input or os.path.join(DEFAULT_RESULTS_DIR, 'generations.csv')
    if not os.path.exists(generations_file):
        raise FileNotFoundError(f"Generations file not found: {generations_file}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data ---------------------------------------------------------------
    df = pd.read_csv(generations_file)

    # Compute metrics ---------------------------------------------------------
    metrics = compute_metrics(df)
    acc_df = metrics['accuracy']
    var_df = metrics['variance']
    fail_df = metrics['failure_modes']
    rec_df = metrics['recency']

    # Save CSVs ---------------------------------------------------------------
    acc_df.to_csv(os.path.join(args.output_dir, 'accuracy_by_rules.csv'), index=False)
    var_df.to_csv(os.path.join(args.output_dir, 'accuracy_std_by_rules.csv'), index=False)
    fail_df.to_csv(os.path.join(args.output_dir, 'failure_modes_by_rules.csv'), index=False)
    rec_df.to_csv(os.path.join(args.output_dir, 'recency_by_rules.csv'), index=False)

    # extract per-sample to compute additional time-series metrics
    per_sample_df = metrics['per_sample']
    # Save per-sample breakdown for plotting
    per_sample_df.to_csv(os.path.join(args.output_dir, 'per_sample.csv'), index=False)

    # Additional time-series CSVs ------------------------------------------------
    # Omission:Modification ratio by density
    omod = fail_df.copy()
    omod['ratio'] = omod.apply(lambda r: np.nan if r['modification_pct'] == 0 else r['omission_pct'] / r['modification_pct'], axis=1)
    omod[['model','num_rules','ratio']].to_csv(os.path.join(args.output_dir, 'omod_ratio_by_rules.csv'), index=False)

    # Accuracy/Latency efficiency by density
    if 'latency' in per_sample_df.columns:
        lat = per_sample_df.groupby(['model','num_rules']).agg({'accuracy':'mean','latency':'mean'}).reset_index()
        lat['acc_per_latency'] = lat['accuracy'] / lat['latency']
        lat[['model','num_rules','accuracy','latency','acc_per_latency']].to_csv(
            os.path.join(args.output_dir, 'accuracy_latency_by_rules.csv'), index=False)
    else:
        # write empty template if no latency
        pd.DataFrame(columns=['model','num_rules','accuracy','latency','acc_per_latency']).to_csv(
            os.path.join(args.output_dir, 'accuracy_latency_by_rules.csv'), index=False)

    print(f"Full analysis complete. Outputs saved to {args.output_dir}")

if __name__ == '__main__':
    main() 