# Phase 0 基建 — 完成總結

狀態：**完成**。所有基建就緒，可進入 Phase 1 baseline。

## 交付物

| 模組 | 檔案 | 驗證 |
|---|---|---|
| 路徑修正資料載入器 | `src/data/paths.py` | 解析 69,352 train / 7,821 test on-disk，標記 134,997 缺檔 |
| FREUID metric | `src/metric.py` | 6 項自我測試全過；AuDET=1−ROC_AUC（線性軸假設） |
| CV split | `src/data/cv.py` → `artifacts/folds.csv` | stratified 5-fold 詐欺率穩定 0.423；type-LOO 就緒 |
| EDA / dedup | `scripts/eda.py` | stats + pHash + 近重複偵測 |
| 外部資料清單 | `reports/external_data.md` | 候選 + license 紀律 |
| 環境 | `env/requirements.txt` | 含 5060 Ti 需 cu128 nightly 註記 |

## 關鍵發現（影響建模）

### 1. 指標極嚴苛 → 操作點為王
本地實驗：AUC≈0.92 的模型 FREUID 仍只有 **0.47**（1% BPCER 門檻下 63% 攻擊漏網）。
→ 勝負在 **APCER@1%BPCER 的尾部校準**，集成/TTA/pAUC 損失/校準是重點，非整體 AUC。

### 2. 重拍訊號稀少且集中
全 train 僅 **20 張**實拍(is_digital=False)，**15 張為 MAURITIUS/ID**，標籤 14 假/6 真。
→ 幾無真實重拍訓練訊號；**重拍模擬增強**為唯一橋樑；這 20 張保留作重拍域驗證金標準。

### 3. EGYPT/DL 偏斜
樣本最多(15,867)、詐欺率最高(0.496 vs 其他 ~0.40)。→ type-LOO 的 EGYPT 折會偏樂觀。

### 4. 近重複 / 洩漏檢查結論：**無 1:1 洩漏，標準 stratified CV 安全**
- ⚠️ **pHash(8×8 DCT) 報出的「99.86% train↔test 近重複」是假象**：該雜湊只解析到**模板**層級。
  驗證：d=0 配對的像素 MSE≈1000（完全不同文件），4,693 個不同 train id 命中 → 是同模板群聚，非重複。
- **無基底文件 1:1 配對**：最緊密的 within-train 配對（如 BENIN d1905f22 假 ~ 894eeb46 真）
  其實是**共用同一張庫存肖像照**但姓名/日期/證號全不同、標籤相反。
- **肖像照來自共用照片池**，真/假文件都會重用 → **臉部身分不是詐欺訊號**；不可建立臉部捷徑；
  也因此隨機分折不會把標籤洩漏到 val。
→ **結論：沿用 `StratifiedKFold`（已建），不需 group-aware CV。** 詐欺訊號為細節/語義層級，
  呼應高解析 + 取證高頻 + 語義一致性的架構。

### 5. 模板層級結構幾乎相同
同模板文件在低頻 DCT 上近乎一致 → 全域低頻結構無鑑別力，**鑑別訊號在高頻細節與局部區域**
（肖像邊界、欄位、字體）。強化「高解析 + 局部注意力 + 取證分支」的設計方向。

## 待官方確認（不阻塞 Phase 1）
- AuDET 確切定義（線性軸 vs probit）——目前採線性(=1−ROC_AUC)，與「[0,1] 有界」一致。
- 完整 `test/`（142,818）下載——目前僅 public_test 7,821 張。

## 產出的 artifacts
- `artifacts/folds.csv`、`artifacts/phash_{train,test}.npz`、`artifacts/train_test_neardups.csv`、`artifacts/dedup_report.txt`
- 註：`train_test_neardups.csv` / `dedup_report.txt` 為**模板層級**結果，非洩漏，勿誤用。
