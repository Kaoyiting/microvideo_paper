import os
import subprocess
import json
import pandas as pd
from datetime import datetime

# ================= Configuration =================
SCRIPT_DIR = r"c:\Users\kao\microvideo\content\shortvideo_experiment"
SASREC_SCRIPT = os.path.join(SCRIPT_DIR, "sensitivity_sasrec_train_shortvideo.py")
ALIGN_SCRIPT = os.path.join(SCRIPT_DIR, "sensitivity_alignment_train_shortvideo.py")
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_sensitivity_fast.py")
OUT_DIR = os.path.join(SCRIPT_DIR, "optimized_test_results")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_OUTPUT = os.path.join(SCRIPT_DIR, "sensitivity_results_2024.csv")
LOG_OUTPUT = os.path.join(SCRIPT_DIR, "sensitivity_full_log.txt")

# THE ANCHOR (Optimized Yellow)
ANCHOR = {
    "w_i": 0.96, "w_s": 0.02, "w_c": 0.02,
    "w_si": 0.2, "w_sc": 0.6, "w_ic": 0.2,
    "temp": 0.1,
    "lr": 1e-3,
    "dim": 1024,
    "batch_size": 128,
    "layers": 1,
    "heads": 4,
    "epochs": 50
}

# THE PARAMETER GRIDS
EXPERIMENTS = {
    "Fusion Weights": [
        {"w_i": 1.0, "w_s": 0.0, "w_c": 0.0, "note": "Only I"},
        {"w_i": 0.94, "w_s": 0.06, "w_c": 0.0, "note": "Semantic Focus"},
        {"w_i": 0.90, "w_s": 0.10, "w_c": 0.0, "note": "Semantic Focus ++"},
        {"w_i": 0.94, "w_s": 0.0, "w_c": 0.06, "note": "Content Focus"},
        {"w_i": 0.90, "w_s": 0.0, "w_c": 0.10, "note": "Content Focus ++"},
        {"w_i": 0.90, "w_s": 0.05, "w_c": 0.05, "note": "Averaged"},
        {"w_i": 0.80, "w_s": 0.10, "w_c": 0.10, "note": "Averaged ++"},
        {"w_i": 0.96, "w_s": 0.02, "w_c": 0.02, "note": "Anchor Optimal"}
    ],
    "Alignment Weights": [
        {"w_si": 0.33, "w_sc": 0.33, "w_ic": 0.33, "note": "Averaged"},
        {"w_si": 0.1, "w_sc": 0.8, "w_ic": 0.1, "note": "Focus S-C"},
        {"w_si": 0.4, "w_sc": 0.2, "w_ic": 0.4, "note": "Focus Interaction"},
        {"w_si": 0.0, "w_sc": 1.0, "w_ic": 0.0, "note": "Only S-C"},
        {"w_si": 0.2, "w_sc": 0.6, "w_ic": 0.2, "note": "Anchor"}
    ],
    "Temperature": [
        {"temp": 0.05}, {"temp": 0.08}, {"temp": 0.1}, {"temp": 0.2}, {"temp": 0.5}, {"temp": 1.0}
    ],
    "Learning Rate": [
        {"lr": 5e-5}, {"lr": 1e-4}, {"lr": 5e-4}, {"lr": 1e-3}, {"lr": 5e-3}
    ],
    "Dimension": [
        {"dim": 128}, {"dim": 256}, {"dim": 512}, {"dim": 1024}
    ],
    "Batch Size": [
        {"batch_size": 128}, {"batch_size": 256}, {"batch_size": 512}, {"batch_size": 1024}
    ],
    "Layers": [
        {"layers": 1}, {"layers": 2}, {"layers": 3}, {"layers": 4}
    ],
    "Attention Heads": [
        {"heads": 1}, {"heads": 2}, {"heads": 4} # Note 8, 16 is excluded if bugs happen, kept simple
    ]
}

# ================= Setup Logging =================
def log_msg(msg):
    print(msg)
    with open(LOG_OUTPUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

if not os.path.exists(CSV_OUTPUT):
    with open(CSV_OUTPUT, "w", encoding="utf-8") as f:
        f.write("ExperimentGroup,ParamValue,Hit@5,Hit@10,Hit@20,NDCG@5,NDCG@10,NDCG@20,MRR@5,MRR@10,MRR@20,Time\n")

def append_csv(group, label, res):
    with open(CSV_OUTPUT, "a", encoding="utf-8") as f:
        f.write(f"{group},{label},{res['Hit@5']},{res['Hit@10']},{res['Hit@20']},{res['NDCG@5']},{res['NDCG@10']},{res['NDCG@20']},{res['MRR@5']},{res['MRR@10']},{res['MRR@20']},{datetime.now().strftime('%H:%M:%S')}\n")

# ================= Core Execution =================
def run_cmd(cmd, desc):
    log_msg(f"    [RUNNING] {desc}")
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Prevent OpenMP crashing
    res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        log_msg(f"    [ERROR] Command Failed:\n{res.stderr}")
        return None
    return res.stdout

def evaluate_metrics(cfg, tag):
    # Determine which cache tag to use for SASRec and Alignment
    sasrec_tag = f"sens_sasrec_b{cfg['batch_size']}_lr{cfg['lr']}_l{cfg['layers']}_h{cfg['heads']}"
    align_tag = f"sens_align_b{cfg['batch_size']}_d{cfg['dim']}_t{cfg['temp']}_w{cfg['w_si']}_{cfg['w_sc']}_{cfg['w_ic']}"
    
    pth_sasrec = os.path.join(OUT_DIR, f"{sasrec_tag}.pth")
    npz_path = os.path.join(OUT_DIR, f"{align_tag}.npz")
    pth_align = os.path.join(OUT_DIR, f"{align_tag}.pth")

    # PHASE 1: SASREC
    if not os.path.exists(pth_sasrec):
        log_msg(f"      -> Trainging Base SASRec ({sasrec_tag})")
        cmd_2 = (f"python \"{SASREC_SCRIPT}\" --lr {cfg['lr']} --batch_size {cfg['batch_size']} "
                 f"--layers {cfg['layers']} --heads {cfg['heads']} --epochs {cfg['epochs']} "
                 f"--npz_path \"dummy.npz\" --out_pth \"{pth_sasrec}\" --loss_type BCE")
        run_cmd(cmd_2, "SASRec Training")
    else:
        log_msg(f"      -> Base SASRec Cached ({sasrec_tag})")

    # PHASE 2: ALIGNMENT
    if not os.path.exists(npz_path):
        log_msg(f"      -> Trainging Alignment ({align_tag})")
        cmd_1 = (f"python \"{ALIGN_SCRIPT}\" --dim {cfg['dim']} --temp {cfg['temp']} "
                 f"--w_si {cfg['w_si']} --w_sc {cfg['w_sc']} --w_ic {cfg['w_ic']} "
                 f"--epochs {cfg['epochs']} --batch_size {cfg['batch_size']} "
                 f"--out_npz \"{npz_path}\" --out_pth \"{pth_align}\" --sasrec_pth \"{pth_sasrec}\"")
        run_cmd(cmd_1, "Alignment Training")
    else:
        log_msg(f"      -> Alignment Cached ({align_tag})")

    # PHASE 3: EVALUATION (FAST)
    cmd_eval = (f"python \"{EVAL_SCRIPT}\" --sasrec_pth \"{pth_sasrec}\" --align_npz \"{npz_path}\" "
                f"--w_i {cfg['w_i']} --w_s {cfg['w_s']} --w_c {cfg['w_c']} "
                f"--layers {cfg['layers']} --heads {cfg['heads']}")
    
    out = run_cmd(cmd_eval, f"Evaluation ({tag})")
    
    for line in out.split('\n'):
        if line.startswith("EVAL_RESULT_JSON:"):
            res = json.loads(line.split("EVAL_RESULT_JSON:")[1])
            return res
    log_msg(f"      -> Failed to parse output!")
    return None

# ================= Master Loop =================
def main():
    log_msg("==================================================")
    log_msg(f"🚀 SENSITIVITY ANALYSIS STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_msg("==================================================")

    for group_name, variants in EXPERIMENTS.items():
        log_msg(f"\n📌 Testing Group: {group_name}")
        log_msg(f"--------------------------------------------------")
        
        for variation in variants:
            # Reconstruct configs based on anchor
            cfg = ANCHOR.copy()
            label_parts = []
            
            for k, v in variation.items():
                if k == "note": continue
                cfg[k] = v
                label_parts.append(f"{k}={v}")
                
            label = " | ".join(label_parts)
            if "note" in variation:
                label += f" ({variation['note']})"
                
            log_msg(f"🟢 Running: {label}")
            
            res = evaluate_metrics(cfg, label)
            if res:
                log_msg(f"    ✔️ Result: Hit@20={res['Hit@20']:.4f} | NDCG@20={res['NDCG@20']:.4f} | MRR@20={res['MRR@20']:.4f}")
                append_csv(group_name, label, res)

    log_msg("\n✅ ALL SENSITIVITY TASKS COMPLETED!")

if __name__ == "__main__":
    main()