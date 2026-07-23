import subprocess
import os
import json

# ============ 路徑與基準設定 ============
TRAIN_SCRIPT = r"c:\Users\kao\microvideo\content\shortvideo_experiment\sensitivity_alignment_train_shortvideo.py"
SASREC_MODEL = r"c:\Users\kao\microvideo\content\shortvideo_experiment\best_h4_l3_sasrec.pth" # 使用最強 H4L3 模型
CHECKPOINT_DIR = r"c:\Users\kao\microvideo\checkpoints_h4l3"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# ============ 敏感度測試範圍 ============
# 1. 溫度係數 (Tau)
TAU_RANGE = [0.05, 0.1, 0.2]
# 2. 對比損失權重 (si, sc, ic)
LOSS_WEIGHTS = [
    (0.2, 0.6, 0.2),   # 基準
    (0.33, 0.33, 0.33),# 平均
    (0.4, 0.4, 0.2),   # 重互動
    (0.2, 0.4, 0.4)    # 重內容
]
# 3. 學習率 (LR)
LR_RANGE = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
# 4. 批次大小 (Batch Size)
BS_RANGE = [128, 256, 512]

def run_alignment_training(lr=1e-4, temp=0.1, bs=128, w_si=0.2, w_sc=0.6, w_ic=0.2, tag=""):
    print(f"\n>>> Running Alignment Training: {tag}")
    
    filename_base = f"aligned_H4L3_dim1024_temp{temp}_bs{bs}_lr{lr}_w{w_si}-{w_sc}-{w_ic}"
    out_npz = os.path.join(CHECKPOINT_DIR, f"{filename_base}.npz")
    out_pth = os.path.join(CHECKPOINT_DIR, f"{filename_base}.pth")
    
    if os.path.exists(out_npz):
        print(f"Skipping {tag}, file already exists.")
        return out_npz

    cmd = [
        "python", TRAIN_SCRIPT,
        "--sasrec_pth", SASREC_MODEL,
        "--lr", str(lr),
        "--temp", str(temp),
        "--batch_size", str(bs),
        "--w_si", str(w_si),
        "--w_sc", str(w_sc),
        "--w_ic", str(w_ic),
        "--out_npz", out_npz,
        "--out_pth", out_pth,
        "--epochs", "30",
        "--dim", "1024"
    ]
    
    # 設置環境變數避免 OMP 錯誤
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    subprocess.run(cmd, env=env)
    return out_npz

def main():
    # 測試集 1: 溫度 (Tau)
    for tau in TAU_RANGE:
        run_alignment_training(temp=tau, tag=f"Tau={tau}")

    # 測試集 2: 損失權重
    for (si, sc, ic) in LOSS_WEIGHTS:
        run_alignment_training(w_si=si, w_sc=sc, w_ic=ic, tag=f"LossW={si}/{sc}/{ic}")

    # 測試集 3: 學習率
    for lr in LR_RANGE:
        run_alignment_training(lr=lr, tag=f"LR={lr}")

    # 測試集 4: 批次大小
    for bs in BS_RANGE:
        run_alignment_training(batch_size=bs, tag=f"BS={bs}")

if __name__ == "__main__":
    main()
