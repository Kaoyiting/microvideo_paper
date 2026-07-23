# 語意 Embedding 說明

這個資料夾放 Short-video 實驗使用的語意 embedding。

## 用途

語意 embedding 用來表示影片文字、字幕或描述內容，是 proposed model 三路徑中的語意路徑。

## 目前檔案

```text
semantic_shortvideo_ai_only.csv
semantic_shortvideo_cleaned.csv
```

## 重新產生方式

原始產生程式位於：

```text
semantic/
```

若要重新產生，需要設定 OpenAI API key，並安裝相關套件：

```bash
pip install openai pandas numpy tqdm tiktoken scikit-learn
```

注意：不要把 API key 上傳到 GitHub。
