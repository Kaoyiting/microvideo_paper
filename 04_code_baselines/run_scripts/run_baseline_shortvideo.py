import os
import sys
import subprocess
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALIGNMENT_TRAIN = os.path.join(SCRIPT_DIR, "sensitivity_alignment_train_shortvideo.py")
SASREC_TRAIN = os.path.join(SCRIPT_DIR, "sensitivity_sasrec_train_shortvideo.py")

# ================= 新 Baseline 最佳參數設定 =================
DIM = 1024
TAU = 0.1
W_SI, W_SC, W_IC = 0.2, 0.6, 0.2
EPOCHS = 50
BATCH_SIZE = 128
LR = 0.001
LAYERS = 1
HEADS = 4
LOSS_TYPE = "BCE"
W_S, W_C, W_I = 0.3, 0.3, 0.4  # (保留給後續 Fusion 階段使用)

def run_cmd(cmd):
    print(f"執行指令: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc

def main():
    print(f"\n{'='*60}")
    print("🚀 [Step 1] 開始訓練 Phase 1: 多模態特徵空間對齊 (Alignment)")
    print(f"{'='*60}")
    
    npz_dir = os.path.join(SCRIPT_DIR, '..', '..', 'checkpoints')
    os.makedirs(npz_dir, exist_ok=True)
    suffix = f"{DIM}dim_{TAU}tau_{W_SI:.1f}si_{W_SC:.1f}sc_{W_IC:.1f}ic_bs{BATCH_SIZE}_lr{LR}_baseline"
    npz_path = os.path.abspath(os.path.join(npz_dir, f"shortvideo_aligned_embeddings_{suffix}.npz"))
    pth_path = os.path.abspath(os.path.join(npz_dir, f"shortvideo_alignment_{suffix}.pth"))

    cmd_align = [
        sys.executable, ALIGNMENT_TRAIN,
        "--dim", str(DIM),
        "--temp", str(TAU),
        "--w_si", str(W_SI),
        "--w_sc", str(W_SC),
        "--w_ic", str(W_IC),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--epochs", str(EPOCHS),
        "--out_npz", npz_path,
        "--out_pth", pth_path
    ]
    
    try:
        run_cmd(cmd_align)
        print(f"✅ Phase 1 完成！特徵檔已儲存至: {npz_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Phase 1 失敗:\n{e.stderr}")
        return

    print(f"\n{'='*60}")
    print("🚀 [Step 2] 開始訓練 Phase 2: 序列推薦主幹模型 (LightSASRec)")
    print(f"{'='*60}")
    
    sasrec_pth_path = os.path.abspath(os.path.join(npz_dir, f"shortvideo_sasrec_{suffix}.pth"))

    cmd_sasrec = [
        sys.executable, SASREC_TRAIN,
        "--dim", str(DIM),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--layers", str(LAYERS),
        "--heads", str(HEADS),
        "--loss_type", LOSS_TYPE,
        "--npz_path", npz_path,
        "--out_pth", sasrec_pth_path
    ]
    
    try:
        proc = run_cmd(cmd_sasrec)
        output_lines = [line.strip() for line in proc.stdout.split('\n') if line.strip()]
        if output_lines:
            try:
                metrics = json.loads(output_lines[-1])
                print(f"✅ Phase 2 完成！模型指標 (Metrics):")
                print(json.dumps(metrics, indent=4))
                print(f"模型權重已儲存至: {sasrec_pth_path}")
            except json.JSONDecodeError:
                print(f"✅ Phase 2 完成！最後輸出為:\n{output_lines[-1]}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Phase 2 失敗:\n{e.stderr}")
        return
        
    print(f"\n{'='*60}")
    print("🚀 [Step 3] 開始驗證 Phase 3: 晚期融合權重 (Late Fusion)")
    print(f"{'='*60}")

    FUSION_RUNNER = os.path.join(SCRIPT_DIR, "sensitivity_fusion_runner_shortvideo.py")
    cmd_fusion = [
        sys.executable, FUSION_RUNNER,
        "--npz_path", npz_path,
        "--pth_path", pth_path,
        "--sasrec_pth", sasrec_pth_path, # Pass the new SASRec model
        "--w_s", str(W_S), 
        "--w_i", str(W_I), 
        "--w_c", str(W_C) 
    ]
    
    try:
        proc = run_cmd(cmd_fusion)
        print(proc.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Phase 3 失敗:\n{e.stderr}")
        return

    print(f"\n{'='*60}")
    print("🎉 新基準線 (Baseline) 最佳綜合結果生成完畢！")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
