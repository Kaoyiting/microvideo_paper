import os
import subprocess
import json

eval_script = r"C:\Users\kao\microvideo\content\shortvideo_experiment\eval_sensitivity_fast.py"
sasrec_pth = r"C:\Users\kao\microvideo\content\shortvideo_experiment\golden_model_results\golden_sasrec.pth"
align_npz = r"C:\Users\kao\microvideo\content\shortvideo_experiment\golden_model_results\golden_align.npz"

def run_evaluation(w_i, w_s, w_c):
    cmd = [
        "python", eval_script,
        "--sasrec_pth", sasrec_pth,
        "--align_npz", align_npz,
        "--w_i", str(w_i),
        "--w_s", str(w_s),
        "--w_c", str(w_c),
        "--layers", "1",
        "--heads", "1"
    ]
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in result.stdout.split('\n'):
        if "EVAL_RESULT_JSON:" in line:
            return json.loads(line.split("EVAL_RESULT_JSON:")[1].strip())
    return None

# Definition of the 7 ablation setups
# For mixed ID+Modalities, we use the optimal micro-weights found earlier so they don't break.
# For pure modalities (S, C, S+C), we use standard weighting.
configs = [
    {"name": "S",       "w_i": 0.0,   "w_s": 1.0,   "w_c": 0.0},
    {"name": "I",       "w_i": 1.0,   "w_s": 0.0,   "w_c": 0.0},
    {"name": "C",       "w_i": 0.0,   "w_s": 0.0,   "w_c": 1.0},
    {"name": "S+I",     "w_i": 0.5,   "w_s": 0.5,   "w_c": 0.0},
    {"name": "I+C",     "w_i": 0.5,   "w_s": 0.0,   "w_c": 0.5},
    {"name": "S+C",     "w_i": 0.0,   "w_s": 0.5,   "w_c": 0.5},
    {"name": "S+I+C",   "w_i": 0.999, "w_s": 0.001, "w_c": 0.001}
]

print(f"{'Config':<10} | {'w_i , w_s , w_c':<22} | {'Hit@5':<7} | {'Hit@10':<7} | {'Hit@20':<7} | {'NDCG@5':<7} | {'NDCG@10':<7} | {'NDCG@20':<7} | {'MRR@5':<7} | {'MRR@10':<7} | {'MRR@20':<7}")
print("-" * 135)

for cfg in configs:
    res = run_evaluation(cfg["w_i"], cfg["w_s"], cfg["w_c"])
    w_str = f"{cfg['w_i']:.3f}, {cfg['w_s']:.3f}, {cfg['w_c']:.3f}"
    if res:
        print(f"{cfg['name']:<10} | {w_str:<22} | {res['Hit@5']:.5f} | {res['Hit@10']:.5f} | {res['Hit@20']:.5f} | {res['NDCG@5']:.5f} | {res['NDCG@10']:.5f} | {res['NDCG@20']:.5f} | {res['MRR@5']:.5f} | {res['MRR@10']:.5f} | {res['MRR@20']:.5f}")
    else:
        print(f"{cfg['name']:<10} | {w_str:<22} | FAILED")