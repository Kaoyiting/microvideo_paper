import os
import sys
import subprocess
import json

# Define the absolute path to the fast evaluation script
eval_script = r"C:\Users\kao\microvideo\content\shortvideo_experiment\eval_sensitivity_fast.py"
sasrec_pth = r"C:\Users\kao\microvideo\content\shortvideo_experiment\golden_model_results\golden_sasrec.pth"
align_npz = r"C:\Users\kao\microvideo\content\shortvideo_experiment\golden_model_results\golden_align.npz"

def run_evaluation(w_i, w_s, w_c, layers, heads):
    cmd = [
        "python", eval_script,
        "--sasrec_pth", sasrec_pth,
        "--align_npz", align_npz,
        "--w_i", str(w_i),
        "--w_s", str(w_s),
        "--w_c", str(w_c),
        "--layers", str(layers),
        "--heads", str(heads)
    ]
    
    # Run process with KMP bug fix environment variable
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Parse output for the json metric block
    for line in result.stdout.split('\n'):
        if "EVAL_RESULT_JSON:" in line:
            json_str = line.split("EVAL_RESULT_JSON:")[1].strip()
            return json.loads(json_str)
            
    print("Error parsing output for weights:", w_i, w_s, w_c)
    print(result.stdout)
    return None

def main():
    # Model config
    layers = 1
    heads = 1
    
    print("=== ✨ GOLDEN CONFIGURATION SETTINGS ===")
    print("• Architecture: Dim=128, Batch=128, Layers=1, Heads=1")
    print("• Learning: LR=0.001, Temp=0.05, Align=0.33/0.33/0.33")
    print("=========================================\n")
    
    # Define Ablation Matrix configs (w_i, w_s, w_c)
    ablation_matrix = [
        {"name": "1. I (Baseline)", "w_i": 1.0, "w_s": 0.0, "w_c": 0.0},
        {"name": "2. I + S (Visual)", "w_i": 0.999, "w_s": 0.01, "w_c": 0.0},
        {"name": "3. I + C (Text)", "w_i": 0.999, "w_s": 0.0, "w_c": 0.01},
        {"name": "4. I + S + C (All)", "w_i": 0.999, "w_s": 0.001, "w_c": 0.001}
    ]

    print("Running Ablation Matrix...")
    print(f"{'Variant':<25} | {'MRR@20':<8} | {'NDCG@20':<8} | {'Hit@20':<8} | {'Hit@10':<8} | {'Hit@5':<8}")
    print("-" * 75)
    
    for cfg in ablation_matrix:
        res = run_evaluation(cfg["w_i"], cfg["w_s"], cfg["w_c"], layers, heads)
        if res:
            # Emphasize best values visually (e.g. bolded MRR/NDCG in markdown format)
            name = cfg["name"]
            hit20 = f"{res['Hit@20']:.5f}"
            hit10 = f"{res['Hit@10']:.5f}"
            hit5 = f"{res['Hit@5']:.5f}"
            ndcg20 = f"{res['NDCG@20']:.5f}"
            mrr20 = f"{res['MRR@20']:.5f}"
            
            print(f"{name:<25} | {mrr20:<8} | {ndcg20:<8} | {hit20:<8} | {hit10:<8} | {hit5:<8}")

if __name__ == "__main__":
    main()
