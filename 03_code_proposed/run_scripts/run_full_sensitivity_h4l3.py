import os
import subprocess
import json

# 配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SASREC_PTH = os.path.join(SCRIPT_DIR, "architectures/sasrec_h4_l3.pth")
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_alignment_h4l3_fast.py")
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_sensitivity_fast.py")

# 敏感度範圍
tau_list = [0.05, 0.1, 0.2]
dim_list = [128, 256, 512, 1024]
bs_list = [128, 256, 512]
lr_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

def run_experiment(name, dim=1024, tau=0.1, bs=128, lr=0.0001):
    align_npz = os.path.join(SCRIPT_DIR, f"checkpoints_h4l3/h4l3_align_dim{dim}_tau{tau}_bs{bs}_lr{lr}.npz")
    
    # 1. Train Alignment (Fast Mode)
    train_cmd = f"python {TRAIN_SCRIPT} --sasrec_pth {SASREC_PTH} --dim {dim} --tau {tau} --batch_size {bs} --lr {lr}"
    subprocess.run(train_cmd, shell=True)
    
    # 2. Evaluate
    eval_cmd = f"python {EVAL_SCRIPT} --sasrec_pth {SASREC_PTH} --align_npz {align_npz} --w_i 0.985 --w_s 0.005 --w_c 0.01 --layers 3 --heads 4 --dim 64"
    result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
    
    for line in result.stdout.split('\n'):
        if "EVAL_RESULT_JSON:" in line:
            return json.loads(line.split("EVAL_RESULT_JSON:")[-1])
    return None

results = {"Tau": {}, "Dim": {}, "BS": {}, "LR": {}}

print("Starting Sensitivity Analysis for H4L3...")

# 1. Tau
for t in tau_list:
    print(f"Testing Tau={t}...")
    results["Tau"][t] = run_experiment(f"tau_{t}", tau=t)

# 2. Dim
for d in dim_list:
    print(f"Testing Dim={d}...")
    results["Dim"][d] = run_experiment(f"dim_{d}", dim=d)

# 3. BS
for b in bs_list:
    print(f"Testing BS={b}...")
    results["BS"][b] = run_experiment(f"bs_{b}", bs=b)

# 4. LR
for l in lr_list:
    print(f"Testing LR={l}...")
    results["LR"][l] = run_experiment(f"lr_{l}", lr=l)

# 打印表格
def print_table(title, data):
    print(f"\n[Table] {title}")
    print("-" * 60)
    print("{:<10} | {:<8} | {:<8} | {:<8} | {:<8}".format("Setting", "Hit@5", "Hit@10", "NDCG@10", "MRR@10"))
    print("-" * 60)
    for k, v in data.items():
        if v:
            print("{:<10} | {:.4f}   | {:.4f}   | {:.4f}    | {:.4f}".format(
                str(k), v.get("Hit@5",0), v.get("Hit@10",0), v.get("NDCG@10",0), v.get("MRR@10",0)
            ))

print_table("Temperature (Tau)", results["Tau"])
print_table("Hidden Dimension (Dim)", results["Dim"])
print_table("Batch Size (BS)", results["BS"])
print_table("Learning Rate (LR)", results["LR"])
