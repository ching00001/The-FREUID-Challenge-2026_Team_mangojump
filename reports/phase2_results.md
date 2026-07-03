# Phase 2 結果與策略修正（含負面結果）

## 公開 LB 記分板（越低越好；只算 7,821 張公開測試，私有填 dummy）
| # | 模型 | 設定 | clean val | **公開 LB** |
|---|---|---|---|---|
| 1 | **baseline CNN** | ConvNeXt-V2-T, 320×512, 無重拍, fold0 | 0.0005 | **0.269 ← 目前最佳** |
| 2 | R0 重拍增強 CNN | + recapture_p=0.9, val_recapture=0.9 | 0.176 | 0.357（更差）|
| 3 | patch-DINOv2 | 凍結 DINOv2-S, 4×3 grid, attention-MIL | 0.0004 | 0.454（最差）|

**核心教訓：每加一層複雜度，公開 LB 都更糟。最簡單的 full-image CNN 最佳。**

## 為什麼複雜方法失敗（已診斷）
- **R0（重拍增強）**：(a) 重增強摧毀 clean 數位表現（公開測試其實獎勵它）；(b) 我的重拍 sim 不匹配真實——R0 在 rc0.9 sim 上 0.149 但真實公開 0.357。rc-sim 因此對「跨模型選型」無效。
- **patch-DINOv2**：切 4×3 patch + 半解析度**打碎了肖像合成所需的全域脈絡**；凍結 DINOv2 特徵照樣飽和 clean（沒避開來源 artifact）。

## 公開測試硬樣本 characterization（看圖）
公開硬樣本幾乎都是**數位合成 + 文字竄改，不是 print-and-capture**：
- **肖像合成/貼照/GenAI 臉**（public 080f63ae, 0f23ae28）→ full-image CNN 抓得到、patch 漏掉。
- **文字欄位竄改**（地址覆寫/刪除線，d4e18cc1, 31f3b413, 0c8b0c71）→ baseline 不確定(score~0.15)、patch 常抓到。
- 沒有眩光/透視/紙張紋理 → **公開測試不是重拍硬樣本**（重拍可能集中在私有集）。

→ baseline 唯一明顯弱點是**文字竄改**，疑似 320×512 下文字痕跡（~2px）看不清 → **提高解析度**是最有針對性的保守改進。

## 驗證策略（誠實現況）
- **clean val 對所有模型飽和(~0)** → 無法選型。
- **rc-sim proxy 無效**（被 R0 game 掉）。
- **20 張真實重拍 holdout**（is_digital=False, train.py 自動排除+每 epoch 報 ho_auc/ho_gap）→ 是**重拍/私有軸**的粗略 proxy，**不是公開（數位合成）軸**的 proxy。
- **公開與私有可能獎勵不同東西**（公開=數位合成；私有=未見類型+重拍）。公開 LB 省著用。

## 當前方向（保守、留在贏的範式）
1. **B-hires**：full-image ConvNeXt-V2-T @ 448×704（攻文字竄改）— 進行中。
2. **集成**：baseline(320) + B-hires(448)，rank-average，多尺度 TTA（`src/infer.py` 已支援）。
3. 之後：多 fold/seed 集成；視情況再考慮針對性的「合成詐欺資料生成」攻數位合成（但需小心 sim≠real）。

## 工程基建（本階段新增）
- `src/aug/recapture.py`（重拍模擬，校準 rc0.9≈公開 baseline）— 目前不用於訓練。
- `src/precompute_dino.py` + `src/train_patch_head.py`（凍結嵌入快取→秒級 head 訓練）。
- `src/eval_robustness.py`（robustness curve）；`src/infer.py`（集成+多尺度 TTA+rank）。
- train.py：20 樣本 real holdout 內建；CLI bool 修正；channels_last/cudnn 加速；Windows commit/worker 修正。
