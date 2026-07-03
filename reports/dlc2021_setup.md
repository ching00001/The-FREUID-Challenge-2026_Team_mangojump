# DLC-2021 接入流程（真實重拍仲裁集）

> **目的**:私有測試(95% 總分)以**重拍/翻拍**為主軸。我們手上的重拍訊號全不可信——
> `recapture.py` 模擬幫了本地代理卻**害了公開 LB**(模擬≠真實),in-train 真實重拍只有 **n=20**。
> DLC-2021 提供 ~1400 段**真實**拍攝 clip(原件 vs 螢幕/列印重拍),當**仲裁集**判斷模型是否
> 「genuine 保持乾淨 + 抓到重拍」,取代不可信的代理。**只用於評估/選型,不混入訓練**(避免重蹈
> SIDTD 域不匹配 5x 退步、recapture-aug 害公開的覆轍)。

## 0. 一句話總結
```
python -m src.data.fetch_dlc2021            # 下載 part1: or(genuine)+cg(copy) ~34GB
#  解壓 or.zip / cg.zip 到 external/dlc2021/
python -m src.data.index_dlc2021            # → artifacts/dlc2021_index.csv
python -m src.eval_robust_fusion --arbiter artifacts/dlc2021_index.csv   # 仲裁冠軍融合
```

## 1. 來源 / License（引用必填）
- Zenodo(CC BY-SA 2.5,允許競賽+衍生,需署名):
  - part1 `7467028`:**or.zip 18.8GB(原件=genuine)** + cg.zip 15GB(灰階copy) + dlc-2021.csv  ← 單一 zip,最省事
  - part2 `6792396`:re.zip + re.z01..z08 ~38GB(**螢幕重拍**)  ← 多卷分割壓縮
  - part3 `7467000`:cc.zip + cc.z01..z07 ~33GB(彩色copy)     ← 多卷分割壓縮
- 引用:Polevoy et al., "Document Liveness Challenge dataset (DLC-2021)", *J. Imaging* 8(7):181, 2022.
- 取得日期:2026-06-24(record ID 經 Zenodo API 核實)。

## 2. 下載(`src/data/fetch_dlc2021.py`)
串流下載 + 斷點續傳(HTTP Range)+ 進度條,存到 `external/dlc2021/`。
```
python -m src.data.fetch_dlc2021 --list           # 只列檔案大小,不下載
python -m src.data.fetch_dlc2021                  # 預設 or+cg(~34GB,推薦起步)
python -m src.data.fetch_dlc2021 --parts or       # 只要 genuine(18.8GB)
python -m src.data.fetch_dlc2021 --parts re       # 加螢幕重拍(38GB,最貼近「拍螢幕」)
```
**建議**:先 `or + cg`(part1 全是單一 zip,無分割壓縮麻煩)就足以做 genuine-vs-copy 仲裁。
要最貼近「手機拍螢幕」的重拍才追加 `--parts re`(注意是 9 卷分割檔)。

## 3. 解壓
- **單一 zip**(or/cg):直接解壓到 `external/dlc2021/`(任何工具)。
- **分割壓縮**(re/cc:`.zip` + `.z01..`):需先合卷再解。
  - 7-Zip(Windows 最簡單):直接對 `re.zip` 按解壓,它會自動讀 `.z01..` 各卷。
  - 或:`zip -s 0 re.zip --out re_full.zip && unzip re_full.zip`

## 4. 建索引(`src/data/index_dlc2021.py`)
掃 `external/dlc2021/`,依 DLC 命名 `<template>/<NN>.<cat><NNNN>`(cat∈{or,cg,cc,re})判類,
輸出與 `sidtd_index.csv` 同 schema 的 `artifacts/dlc2021_index.csv`:
`id, abspath, label, is_digital, type`。
- **標籤**:`or`→**0(genuine)**,`cg/cc/re`→**1(重拍/翻拍 spoof)**;全部 `is_digital=False`;`type=DLC/<cat>`。
- **影格 vs 影片皆相容**:有抽好的 jpg/png 就直接用;若是 mp4 影片,用 OpenCV 等距抽 N 幀(存到各 clip 旁 `_frames/`)。
```
python -m src.data.index_dlc2021                              # 每段影片抽 1 幀
python -m src.data.index_dlc2021 --frames_per_clip 3 --max_per_cat 1500  # 更多幀 / 平衡上限
```
> 解壓後實際目錄若與假設不同,腳本用正則 `\.(or|cg|cc|re)\d+` 從路徑判類,通常不需改;
> 真有出入只要確認檔名仍含 `.cc0001` 這種片段即可。

## 5. 仲裁冠軍融合(`src/eval_robust_fusion.py --arbiter`)
用快取的乾淨 train 特徵重建**與冠軍完全相同的 head**,在 DLC 真實重拍上評分,
回報 AUC / gap / **genuine_p(誤報指標)** / **APCER@1%BPCER**(賽事指標)。
```
python -m src.eval_robust_fusion --arbiter artifacts/dlc2021_index.csv
```
**判讀**(對照目前 n=20 結果:genuine_p=0.002 零誤報、AUC 0.881):
- `genuine_p` 仍接近 0 → 冠軍在真實重拍上**不誤報 genuine** = 安全,可放心當最終提交。
- `APCER@1%` 低 → 在 1% genuine 誤報下漏掉的重拍詐欺少 = 私有軸穩。
- 若 genuine_p 明顯上升 / APCER@1% 變差 → DINOv3 重拍脆弱確實滲進融合,
  才考慮 **SigLIP 加權的融合** 或 **用 DLC 做真實重拍硬化訓練**(用 DLC 仲裁、**不是**公開 LB 選型)。

## 6. 紀律(沿用 external_data.md checklist)
- [x] license 核實:CC BY-SA 2.5(2026-06-24,Zenodo API)。
- [ ] 技術報告列出來源(名稱/版本/URL/license/用途)。
- [x] **只做仲裁,不混訓練**(除非仲裁先證明硬化有效)。
- [ ] 若日後混入訓練:小比例 + 用 DLC 留出仲裁,監控公開 LB 操作點不退化。
