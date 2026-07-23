# 論文交付資料包

這個資料夾整理論文實驗交付需要的資料、程式、結果與說明。

## 主要用途

- 複製到隨身碟，一式三份。
- 上傳 GitHub 可公開的資料與程式。
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
07_docs/                      論文與投影片檔案
08_editable_figures_tables/   Word 不能直接修改時才放圖表原始檔
09_ieee_format/               IEEE 格式論文檔案
10_elsevier_format/           Elsevier 格式論文檔案
11_github_links/              GitHub 連結
12_personal_photo/            個人照片
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
