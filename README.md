# 論文

整理論文實驗需要的資料、程式、結果與說明。
論文架構如下
<img width="1847" height="896" alt="論文架構" src="https://github.com/user-attachments/assets/bf84d19f-2d16-494f-b3db-3d3e6208acd6" />
本研究提出一個整合多模態內容與序列化互動的推薦框架，於單一體系下同時建模三條資訊路徑：
* 語意路徑（Semantic）：利用 Gemini 生成影片語意摘要與主題標籤，並透過 OpenAI text-embedding-3-small 模型轉換為語意向量，強化語意層級的檢索能力；
* 內容路徑（Content）：採用 ImageBind 將文本、視覺與聲音對齊至共享特徵空間，獲取多模態內容表示；
* 互動路徑（Interaction）：結合 LightGCN 捕捉長期結構性偏好，並整合序列模型 SASRec 吸收短期情境偏好，兼顧圖式擴散效應與近期序列動態。

## 主要用途

- 說明資料內容、程式位置、結果檔案與安裝環境。

## 資料夾內容

```text
00_manifest/                  檔案清單、資料說明、檢查報告
01_data_processed/            前處理後的實驗資料
02_features_embeddings/       語意、內容、互動、對齊後的 embedding
03_code_proposed/             Proposed model 程式
04_code_baselines/            各 baseline 程式
05_results_excel/             實驗中間與最後結果
06_checkpoints_selected/      可重現主要分數的小型 checkpoint
```

## 程式位置

Proposed model：

```text
03_code_proposed/
```

Baseline：

```text
04_code_baselines/
```

目前 baseline 包含：

```text
POP
BPR-MF
LightGCN
GRU4Rec
SASRec
ImageBind
MMGCN
DualGNN
VBPR
LATTICE
MMSSL
MetaMMF
```

## 結果檔案

結果 Excel 與中間 CSV 放在：

```text
05_results_excel/
```

其中：

- `baseline_comparison/`：baseline comparison
- `ablation/`：ablation study
- `sensitivity/`：sensitivity analysis
- `real_case_study/`：real case study，之後由使用者補 Excel
- `intermediate_results/`：實驗中間結果

## 安裝環境

請看：

```text
INSTALLATION.md
```

## 原始資料

原始影片、音訊、frame 檔案目前沒有直接放入 GitHub-ready 資料包。位置寫在：

```text
00_manifest/raw_data_note.md
```
