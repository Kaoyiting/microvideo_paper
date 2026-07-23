# 安裝環境說明

本專案是 PyTorch 推薦系統實驗。

## 建議環境

- Python 3.9 以上
- PyTorch
- Windows 或 Linux
- 若要重新訓練模型，建議使用 NVIDIA GPU

## 基本套件

在專案根目錄執行：

```bash
pip install -r requirements.txt
pip install torch
```

`requirements.txt` 目前包含：

```text
numpy
pandas
scikit-learn
tqdm
openai
scipy
```

## ImageBind 相關套件

如果要重新產生 ImageBind/content embedding，請另外查看：

```text
external/ImageBind/requirements.txt
```

## 資料位置

前處理後資料：

```text
thesis_artifacts/01_data_processed/
```

embedding 與特徵：

```text
thesis_artifacts/02_features_embeddings/
```

原始資料位置請看：

```text
thesis_artifacts/00_manifest/raw_data_note.md
```

## 程式位置

Proposed model：

```text
thesis_artifacts/03_code_proposed/
```

Baseline：

```text
thesis_artifacts/04_code_baselines/
```
