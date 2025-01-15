# Pairwise Analysis Tool

A Python script for analyzing pairwise comparison data using Singular Value Decomposition (SVD) to generate rankings and provide statistical analysis.

## Overview

This tool takes a CSV file containing pairwise comparisons between teams and generates:
- Rankings with scores
- Statistical measures of ranking reliability
- Validation information and suggestions
- Network connectivity analysis

## Requirements

- Python 3.7+
- Required packages:
  - numpy
  - pandas
  - scipy

Install dependencies using:
```bash
pip install numpy pandas scipy
```

## Input Format

The tool expects a CSV file with three columns:
- `teamA`: First team's ID/number
- `teamB`: Second team's ID/number
- `diff`: Difference/comparison score between teams
  - Positive value means teamA is better than teamB
  - Negative value means teamB is better than teamA
  - Magnitude indicates the strength of the difference

Example CSV (`comparisons.csv`):
```csv
teamA,teamB,diff
1,2,1
3,2,2
1,3,3
```

## Usage

Basic usage:
```bash
python pairwise_analysis.py comparisons.csv
```

Output full results in JSON format (may be broken at the moment):
```bash
python pairwise_analysis.py comparisons.csv --json
```

## Output

The tool provides:
1. Rankings
   - Ordered list of teams
   - Each team's score and rank

2. Statistics
   - Number of teams and comparisons
   - Condition number (ranking stability)
   - Consistency ratio

3. Validation Information
   - Distribution of comparisons
   - Warnings about isolated teams
   - Suggestions for improvement

## Example Output

```
RANKINGS
----------------------------------------
Rank  1: Team  254 (Score: -89.45)
Rank  2: Team 1678 (Score: -45.32)
Rank  3: Team 1114 (Score:  78.91)

STATISTICS
----------------------------------------
Teams: 3
Comparisons: 6
Condition number: 487.23
Consistency ratio: 0.15

INFORMATION
----------------------------------------
Teams have 2-4 comparisons each
Average: 3.0 comparisons per team

WARNINGS
----------------------------------------
Teams with few comparisons: [1114]

SUGGESTIONS
----------------------------------------
* Collect more comparisons for team 1114
```

## Notes

- The condition number indicates ranking stability
  - Lower numbers (closer to 1) indicate more stable rankings
  - Higher numbers suggest rankings may be sensitive to small changes
  - "∞" indicates numerical instability

- The consistency ratio measures how well the comparisons agree
  - Values closer to 1 indicate better agreement
  - Low values suggest contradictory comparisons

## License

MIT License
