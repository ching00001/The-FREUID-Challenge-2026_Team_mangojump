# Phase 1 Runbook — Baseline 與實驗紀錄

目標：建立可信的 FREUID 基準、首個校準提交、以及公開 LB 相關性；所有 run 全超參記錄。

## 實驗紀錄機制（每個 run 自動產生）
`experiments/<run_id>/`：
- `config.json` — **完整超參數**（單檔即可複現）
- `env.json` — python/torch/timm 版本、GPU、VRAM
- `metrics.jsonl` — 逐 epoch FREUID/AuDET/APCER@1%BPCER/ROC_AUC/loss
- `log.txt` — 完整日誌
- `oof.csv` — best epoch 的 OOF 預測（id,label,score）
- `best.pt` — best checkpoint
- `summary.json` — 最終摘要
全域：`experiments/registry.csv`（每 run 一列）。檢視：`python scripts/show_runs.py`

## 常用指令
```bash
# 單折訓練（任一 ExperimentConfig 欄位都是 CLI 旗標）
python -m src.train --name <n> --cv_scheme skf --val_fold 0 --img_h 320 --img_w 512 \
  --batch_size 32 --grad_accum 1 --grad_checkpointing false --epochs 5 --lr 2e-4

# 跨域泛化估計（leave-one-type-out）
python -m src.train --name loto_egypt --cv_scheme loto --loto_type EGYPT/DL ...

# 推論→提交（自動處理缺檔 test id）
python -m src.infer --runs <run_id> --tta hflip --out subs/<n>.csv

# 校準（主要供集成；單模型 FREUID 對單調轉換不變）
python -m src.calibrate --run <run_id>

# smoke
python -m src.train --name smoke --subset 800 --epochs 1 --img_h 128 --img_w 192
```

## 計畫中的實驗序列（依序，全部登錄 registry）
1. **B0 baseline**：cnv2-tiny @320×512, skf f0, 5ep, light aug, bce, ema ✅(進行中)
2. **B1 OOD 檢核**：同設定但 `--cv_scheme loto`（逐一 5 類）→ 估 in-dist vs OOD 落差。
3. **B2 5-fold**：B0 設定跑 f0–f4 → 完整 OOF + 集成提交（若 B0 數字合理才投入算力）。
4. **B3 解析度/速度**：384×608 vs 256×384；測 torch.compile 加速（Blackwell nightly）。
5. **B4 損失**：focal vs bce；pos_weight；label_smoothing（觀察對 APCER@1%BPCER 的影響）。
6. **B5 backbone**：eva02_small / swin_base 對照。

> 決策準則：以 **type-LOO 的 FREUID** 為主要選型依據（最貼私有 OOD），skf 數字僅供 LB 相關性參考；**不追公開 LB**。

## 已知環境要點
- 單卡 RTX 5060 Ti 16GB；torch 2.11 nightly cu128（Blackwell sm_120 必需）。
- 實測吞吐 ~55 img/s @320×512（GPU compute bound；cudnn 對 sm_120 可能未最佳化）。
- `expandable_segments` 在 Windows 不支援（警告可忽略）。
- 已修正 CLI bool 解析 bug（`from __future__ import annotations` 使型別字串化，改由 default 值推斷型別）。

## 待辦/風險
- 完整 `test/`（142,818）尚未下載 → 目前提交只能對 7,821 張公開測試評分，其餘填中性值。
- AuDET 軸定義待官方確認（現採線性=1−ROC_AUC）。
