# Baseline 程式說明

這個資料夾放所有 baseline 的程式。每個 baseline 都是獨立資料夾，方便老師或審查者直接找到對應程式。

## Baseline 清單

```text
pop/          POP baseline
bprmf/        BPR-MF baseline
lightgcn/     LightGCN baseline
gru4rec/      GRU4Rec baseline
sasrec/       SASRec baseline
imagebind/    ImageBind/content-only baseline
mmgcn/        MMGCN baseline
dualgnn/      DualGNN baseline
vbpr/         VBPR baseline
lattice/      LATTICE baseline
mmssl/        MMSSL baseline
metammf/      MetaMMF baseline
```

## 共用程式

```text
shared_src/
```

這裡放 dataloader、model、evaluator 與部分 baseline 需要的 helper。

## 注意

部分 baseline 需要重新訓練，執行時間會比較長。

大型 checkpoint 不一定放在這個資料夾，請查看：

```text
../06_checkpoints_selected/
```
