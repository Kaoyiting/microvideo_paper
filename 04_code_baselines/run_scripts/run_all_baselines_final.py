import os
import subprocess
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 根據清單找到的腳本路徑
BASELINES = {
    "Random": "baseline_random_shortvideo.py",
    "POP": "baseline_pop_shortvideo.py",
    "BPR-MF": "baseline_bprmf_shortvideo.py",
    "LightGCN": "baseline_lightgcn_shortvideo.py",
    "ImageBind": "baseline_imagebind_shortvideo.py",
    "SASRec": "baseline_sasrec_shortvideo.py",
    "GRU4Rec": "baseline_gru4rec_shortvideo.py"
}

# 這是我們之前的 Ours (Ultimate) 數據 (H4L3, Dim 256, SI_Focus, 5-run avg)
OURS_RESULTS = {
    "Method": "Ours (Ultimate)",
    "Hit@5": 0.1623, "Hit@10": 0.2468, "Hit@20": 0.3414,
    "NDCG@5": 0.0427, "NDCG@10": 0.0448, "NDCG@20": 0.0536,
    "MRR@5": 0.0900, "MRR@10": 0.1010, "MRR@20": 0.1074
}

def run_baseline(name, script_name):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"Running baseline: {name} ({script_name})...")
    result = subprocess.run(f"python {script_path}", shell=True, capture_output=True, text=True)
    
    # 解析非 JSON 格式的輸出 (K | Hit | NDCG | MRR)
    data = {"Method": name}
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        # 尋找 5, 10, 20 的行
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 4:
            k_val = parts[0]
            if k_val in ["5", "10", "20"]:
                try:
                    data[f"Hit@{k_val}"] = float(parts[1])
                    data[f"NDCG@{k_val}"] = float(parts[2])
                    data[f"MRR@{k_val}"] = float(parts[3])
                except ValueError:
                    continue
    
    # 如果還是有 EVAL_RESULT_JSON，也保留備用
    for line in lines:
        if "EVAL_RESULT_JSON:" in line:
            try:
                json_data = json.loads(line.split("EVAL_RESULT_JSON:")[-1])
                data.update(json_data)
            except:
                continue

    # 檢查是否真的抓到數據
    if "Hit@10" in data:
        return data
    return None

def main():
    final_results = []
    
    # 跑所有的 Baseline
    for name, script_name in BASELINES.items():
        res = run_baseline(name, script_name)
        if res:
            final_results.append(res)
        else:
            print(f"Warning: Failed to get results for {name}")

    # 加入我們的模型數據
    final_results.append(OURS_RESULTS)

    # 輸出 Markdown 表格
    print("\n### Final Comparison: Baselines vs Ours (Ultimate)")
    header = "| Method | Hit@5 | Hit@10 | Hit@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR@5 | MRR@10 | MRR@20 |"
    sep = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    print(header)
    print(sep)
    
    metrics = ["Hit@5", "Hit@10", "Hit@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR@5", "MRR@10", "MRR@20"]
    for r in final_results:
        row = f"| {r['Method']} "
        for m in metrics:
            val = r.get(m, 0.0)
            row += f"| {val:.4f} "
        row += "|"
        print(row)

if __name__ == "__main__":
    main()
