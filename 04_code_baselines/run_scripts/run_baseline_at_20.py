import os
import subprocess
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SASREC_PTH = os.path.join(SCRIPT_DIR, "architectures/sasrec_h4_l3.pth")
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_alignment_real.py")
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_sensitivity_fast.py")

BASE_CONFIG = {
    "dim": 256,
    "tau": 0.1,
    "bs": 256,
    "lr": 0.0001,
    "w_si": 0.6,
    "w_sc": 0.2,
    "w_ic": 0.2,
    "tag": "baseline_v4"
}

def run_experiment(tag, dim, tau, bs, lr, w_si, w_sc, w_ic):
    file_tag = "baseline" if tag == "baseline" else tag

    align_npz = os.path.join(SCRIPT_DIR, "checkpoints_h4l3", f"h4l3_align_{file_tag}.npz")
    
    if not os.path.exists(align_npz):
        print(f"\n>> Training: {tag} ...")
        train_cmd = f"python {TRAIN_SCRIPT} --sasrec_pth {SASREC_PTH} --dim {dim} --tau {tau} --batch_size {bs} --lr {lr} --w_si {w_si} --w_sc {w_sc} --w_ic {w_ic} --tag {file_tag} --seed 42"
        subprocess.run(train_cmd, shell=True)
        
    eval_cmd = f"python {EVAL_SCRIPT} --sasrec_pth {SASREC_PTH} --align_npz {align_npz} --w_i 0.985 --w_s 0.005 --w_c 0.01 --layers 3 --heads 4 --dim 64"
    result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
    
    res_json = None
    for line in result.stdout.split("\n"):
        if "EVAL_RESULT_JSON:" in line:
            try:
                res_json = json.loads(line.split("EVAL_RESULT_JSON:")[-1])
            except: pass
            break
    
    if res_json is None:
        print(f"DEBUG: Eval failed for {tag}. Stdout:\n{result.stdout}\nStderr:\n{result.stderr}")
        
    return res_json

if __name__ == "__main__":
    print("====== Running Exact Baseline parameters for @20 metrics ======")
    baseline_res = run_experiment(
        BASE_CONFIG["tag"], 
        BASE_CONFIG["dim"], 
        BASE_CONFIG["tau"], 
        BASE_CONFIG["bs"], 
        BASE_CONFIG["lr"], 
        BASE_CONFIG["w_si"], 
        BASE_CONFIG["w_sc"], 
        BASE_CONFIG["w_ic"]
    )
    
    if baseline_res:
        print("\n=== BASELINE METRICS ===")
        for k in ["Hit@5", "Hit@10", "Hit@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR@5", "MRR@10", "MRR@20"]:
            print(f"{k}: {baseline_res.get(k, str(None))}")
        print("\nAll stats:", json.dumps(baseline_res, indent=4))

