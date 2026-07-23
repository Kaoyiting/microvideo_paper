#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_weight_ablation_two_stage.py
Systematically test different weight combinations for Two-Stage Yellow Config.
Targets: Hit@20 performance.
"""

import subprocess
import os
import re
import json
import pandas as pd

# Path Configuration
SCRIPT_DIR = r"c:\Users\kao\microvideo\content\shortvideo_experiment"
BASE_SCRIPT = os.path.join(SCRIPT_DIR, "test_yellow_config_two_stage.py")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "weight_ablation_two_stage_results.csv")

# Define Weight Combinations (Ensuring W_I, W_S, W_C > 0)
# S and C are the weights for the semantic and content normalization
WEIGHT_SETS = [
    (0.8, 0.1, 0.1),  # Yellow Default
    (0.8, 0.05, 0.15), # Favors Content
    (0.8, 0.15, 0.05), # Favors Semantic
    (0.7, 0.1, 0.2),
    (0.7, 0.2, 0.1),
    (0.7, 0.15, 0.15),
    (0.6, 0.1, 0.3),
    (0.6, 0.3, 0.1),
    (0.6, 0.2, 0.2),
]

def update_weights(w_i, w_s, w_c):
    with open(BASE_SCRIPT, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update weights in the script
    content = re.sub(r"W_I = \d+\.\d+", f"W_I = {w_i}", content)
    content = re.sub(r"W_S = \d+\.\d+", f"W_S = {w_s}", content)
    content = re.sub(r"W_C = \d+\.\d+", f"W_C = {w_c}", content)
    
    with open(BASE_SCRIPT, 'w', encoding='utf-8') as f:
        f.write(content)

def run_eval():
    result = subprocess.run(["python", BASE_SCRIPT], capture_output=True, text=True)
    output = result.stdout
    
    # Extract Hit@20 for "Yellow Two-Stage (Z-Score)"
    # Pattern to match: Yellow Two-Stage (Z-Score)          | 20    | 0.2784
    try:
        match = re.search(r"Yellow Two-Stage \(Z-Score\)\s+\|\s+20\s+\|\s+([\d\.]+)", output)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error parsing output: {e}")
    return 0.0

def main():
    results = []
    print(f"{'W_I':<5} | {'W_S':<5} | {'W_C':<5} | {'Hit@20':<10}")
    print("-" * 35)
    
    for w_i, w_s, w_c in WEIGHT_SETS:
        update_weights(w_i, w_s, w_c)
        hit20 = run_eval()
        print(f"{w_i:<5.2f} | {w_s:<5.2f} | {w_c:<5.2f} | {hit20:<10.4f}")
        results.append({
            'W_I': w_i, 'W_S': w_s, 'W_C': w_c, 'Hit@20': hit20
        })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    
    # Find Best
    best = df.iloc[df['Hit@20'].idxmax()]
    print("-" * 35)
    print(f"Best Weights: W_I={best['W_I']}, W_S={best['W_S']}, W_C={best['W_C']}")
    print(f"Best Hit@20: {best['Hit@20']:.4f}")

if __name__ == "__main__":
    main()
