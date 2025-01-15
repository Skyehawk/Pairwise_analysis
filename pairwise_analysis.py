#!/usr/bin/env python3
"""
Pairwise Analysis Tool
---------------------
Analyzes pairwise comparison data using SVD method to generate rankings.

This script takes a CSV file containing pairwise comparisons between teams and
generates a detailed analysis including rankings, statistical measures, and 
validation information.

The CSV should have three columns:
- teamA: First team's ID/number
- teamB: Second team's ID/number
- diff: Difference/comparison score between teams (positive means teamA is better)

For example (note the lack of spaces): 

teamA,teamB,diff
123,456,1
789,456,2
789,123,1
...

An example function call for a CSV in the same directory as the .py script would be: 

python pairwise_analysis.py comparisons.csv

"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy.sparse.csgraph import connected_components
from typing import List, Tuple, Dict, Any, Union
import json

def safe_ratio(num: float, denom: float, default: str = "∞") -> Union[float, str]:
    """Calculate ratio safely handling zeros and infinities."""
    try:
        if denom == 0:
            return default
        result = float(num) / float(denom)
        if np.isinf(result):
            return default
        return result
    except:
        return default

def get_validation_info(M: np.ndarray, teams: List[int], comparison_count: int, 
                       condition_number: Union[float, str]) -> Dict[str, List[str]]:
    """Generate validation messages, warnings, and suggestions."""
    messages = []
    warnings = []
    suggestions = []

    # Analyze comparison distribution
    comparison_counts = np.sum(np.abs(M) > 0, axis=1)
    min_comparisons = int(np.min(comparison_counts))
    max_comparisons = int(np.max(comparison_counts))
    avg_comparisons = np.mean(comparison_counts)

    # Find isolated teams
    isolated_teams = [
        teams[i] for i in range(len(teams))
        if comparison_counts[i] < avg_comparisons/2
    ]

    # Analyze connectivity
    adj_matrix = np.abs(M) > 0
    n_components, labels = connected_components(adj_matrix)

    if n_components > 1:
        components = [[] for _ in range(n_components)]
        for i, label in enumerate(labels):
            components[label].append(teams[i])
            
        warnings.append(
            f"WARNING: Teams are in {n_components} separate groups:"
        )
        for i, comp in enumerate(components):
            warnings.append(f"   Group {i+1}: {comp}")
        
        suggestions.append(
            "* Need comparisons between groups for accurate rankings"
        )

    if isolated_teams:
        warnings.append(
            f"WARNING: Teams with few comparisons: {isolated_teams}"
        )
        suggestions.append(
            "* Collect more comparisons for these teams"
        )

    # Check ranking stability
    if condition_number == "∞":
        warnings.append(
            "WARNING: Comparison network is numerically unstable"
        )
    elif isinstance(condition_number, (int, float)) and condition_number > 1000:
        warnings.append(
            f"WARNING: Rankings may be sensitive (condition number: {condition_number})"
        )

    # Add context
    messages.extend([
        f"Comparison stats:",
        f"   - Teams have {min_comparisons}-{max_comparisons} comparisons each",
        f"   - Average: {avg_comparisons:.1f} comparisons per team",
        f"   - Total: {comparison_count} comparisons",
        f"   - Stability metric: {condition_number}"
    ])

    return {
        "messages": messages,
        "warnings": warnings,
        "suggestions": suggestions
    }

def analyze_comparisons(csv_path: Path) -> Dict[str, Any]:
    """Analyze pairwise comparison data from CSV file."""
    # Read and validate CSV
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['teamA', 'teamB', 'diff']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must have columns: {required_cols}")
    except Exception as e:
        raise ValueError(f"Error reading CSV: {str(e)}")

    # Extract unique teams and create comparison matrix
    teams = sorted(list(set(df['teamA'].unique()) | set(df['teamB'].unique())))
    n = len(teams)
    if n < 2:
        raise ValueError("Need at least 2 teams")

    team_idx = {team: i for i, team in enumerate(teams)}
    M = np.zeros((n, n))
    
    for _, row in df.iterrows():
        i = team_idx[row['teamA']]
        j = team_idx[row['teamB']]
        diff = float(row['diff'])
        M[i][j] = diff
        M[j][i] = -diff

    # Compute SVD
    U, s, Vh = np.linalg.svd(M)
    rankings = U[:,0]
    
    # Scale rankings
    max_abs = np.max(np.abs(rankings))
    if max_abs > 0:
        scaled_rankings = (rankings / max_abs) * 100
    else:
        scaled_rankings = rankings * 100

    # Create ranking list
    ranked_teams = []
    for team, idx in team_idx.items():
        raw_score = float(scaled_rankings[idx])
        ranked_teams.append({
            'team': team,
            'rank': None,
            'score': raw_score
        })

    # Sort and assign ranks
    ranked_teams.sort(key=lambda x: x['score'], reverse=False)
    for i, team in enumerate(ranked_teams):
        team['rank'] = i + 1

    # Calculate statistics
    condition_number = safe_ratio(s[0], s[-1])
    
    # Get validation information
    validation = get_validation_info(
        M, teams, len(df),
        condition_number
    )

    return {
        'rankings': ranked_teams,
        'stats': {
            'condition_number': condition_number,
            'matrix_rank': int(np.linalg.matrix_rank(M)),
            'num_comparisons': len(df),
            'num_teams': n,
            'consistency_ratio': safe_ratio(s[-1], s[0], default=0.0)
        },
        'validation': validation
    }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze pairwise comparison data from CSV file"
    )
    parser.add_argument(
        'csv_path',
        type=Path,
        help="Path to CSV file containing comparisons"
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help="Output full results in JSON format"
    )
    args = parser.parse_args()

    try:
        results = analyze_comparisons(args.csv_path)
        
        if args.json:
            print(json.dumps(results, indent=2))
            return

        # Print rankings
        print("\nRANKINGS")
        print("-" * 40)
        for team in results['rankings']:
            print(f"Rank {team['rank']:2d}: Team {team['team']:4d} (Score: {team['score']:6.2f})")

        # Print stats
        print("\nSTATISTICS")
        print("-" * 40)
        stats = results['stats']
        print(f"Teams: {stats['num_teams']}")
        print(f"Comparisons: {stats['num_comparisons']}")
        print(f"Condition number: {stats['condition_number']}")
        print(f"Consistency ratio: {stats['consistency_ratio']}")

        # Print validation info
        validation = results['validation']
        
        if validation['messages']:
            print("\nINFORMATION")
            print("-" * 40)
            for msg in validation['messages']:
                print(msg)

        if validation['warnings']:
            print("\nWARNINGS")
            print("-" * 40)
            for warning in validation['warnings']:
                print(warning)

        if validation['suggestions']:
            print("\nSUGGESTIONS")
            print("-" * 40)
            for suggestion in validation['suggestions']:
                print(suggestion)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
