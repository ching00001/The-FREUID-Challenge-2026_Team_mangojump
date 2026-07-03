# FREUID Challenge 2026 — 架構設計與規劃 (PLAN)

> 目標：偵測偽造身分證件。輸出每張測試影像的「詐欺機率」分數（越高越可能是假）。
> 評分 FREUID Score 越低越好，由 AuDET 與 APCER@1%BPCER 調和平均組成。

---

## 0. 資料現況（已盤點 + 官方確認）

- 訓練：69,352 張；label 0=真 57.7% / 1=假 42.3%。
- 類型：僅 5 種 `EGYPT/DL, GUINEA/DL, BENIN/DL, MOZAMBIQUE/DL, MAURITIUS/ID`，各約 13k，分布均勻。
- **`is_digital`（官方語義：1=純數位 / 0=重拍 printed+captured）嚴重失衡**：純數位 69,332，重拍僅 **20**。
- 影像：已裁切矯正成 ID-1 卡片比例 (~1584×1000, aspect≈1.58)，RGB。
- 測試：sample_submission 142,818 筆；目前硬碟僅 7,821 張(public_test)，**完整 `test/` 未下載**（總包 77,189 檔=train+public_test+sample，不含完整測試）。
- **路徑陷阱**：CSV 的 `image_path=train/xxx.jpeg`，實檔在 `train/train/xxx.jpeg`；測試在 `public_test/public_test/`。
- 詐欺型態（看圖確認）：主肖像 vs 鬼影肖像換人、Gender 與照片性別矛盾、肖像拼接光暈、字體/欄位異常 → **語義/結構不一致為主**。

### 官方描述確認的關鍵約束（鎖定設計）
- **私有測試集含 2 種 train/public 都沒有的證件類型** → 跨域泛化是明文最終評分項；type-LOO CV + 外部證件資料為正解。
- 測試集**側重非合成、實拍(recaptured)樣本** → domain shift 為中心；重拍模擬增強為最高槓桿。
- **提交需為 [0,1] 校準機率** → 校準是硬性要求（非僅為指標）。
- **外部公開資料/預訓練模型允許**（license 相容 + 報告引用所有來源）；專有非公開資料禁止。
- 算力：**單卡 RTX 5060 Ti 16GB** → backbone 規模/解析度/集成數受 VRAM 約束（見 §3 Tier A）。

## 1. 三大核心挑戰

1. **Domain shift（digital→physical）**：訓練幾乎全 digital，但賽題考 print-and-capture。最高槓桿在「**重拍模擬增強**」。
2. **跨域泛化（seen→unseen 類型）**：私有集可能含未見證件 → 不可過擬合 5 個模板。
3. **嚴格操作點（APCER@1%BPCER）**：分數尾部要乾淨且校準 → 集成、TTA、pAUC 損失。

## 2. 評分指標（本地需精準重現）

- `g_audet = 1 - AuDET`；`g_apcer = 1 - APCER@1%BPCER`
- `FREUID = 1 - 2*g_audet*g_apcer/(g_audet+g_apcer)`（調和平均，越低越好）
- 需自行實作 `metric.py`：DET 曲線面積 + 在 BPCER=1% 門檻下的 APCER。**待確認 AuDET 積分定義**（線性 vs normal-deviate 軸），若官方有 metric code 以其為準。

## 3. 整體架構（分層多訊號 + 後段融合）

```
                 ┌─────────────────────────────────────────────┐
   影像 ──┬────► │ Tier A: 全域分類器 (主力)                     │──┐
          │      │  ConvNeXt-V2 / EVA-02 / Swin, 512–768px      │  │
          │      │  + 重拍模擬增強 (核心)                         │  │
          │      └─────────────────────────────────────────────┘  │
          │      ┌─────────────────────────────────────────────┐  │
          ├────► │ Tier B: 取證/偽影分支 (輔助)                  │──┤──► 融合
          │      │  SRM/Bayar 高通殘差, ELA, FFT/DCT 高頻        │  │   (meta-learner
          │      │  ※同時用重拍增強訓練，避免只抓 digital 偽影    │  │    / 加權平均)
          │      └─────────────────────────────────────────────┘  │       │
          │      ┌─────────────────────────────────────────────┐  │       ▼
          └────► │ Tier C: 語義一致性特徵 (輔助, 後期)           │──┘   校準 (isotonic)
                 │  臉↔鬼影 embedding 距離 (ArcFace)             │       │
                 │  OCR 欄位一致性 (性別/日期/MRZ checksum)       │       ▼
                 └─────────────────────────────────────────────┘    最終分數
```

### ⭐ 文獻驅動更新（2026-06-09，詳見 `reports/literature_review.md`）
- **新增 patch-DINOv2 分支為核心**：證件切 patch(64–128px)→凍結 DINOv2→patch 分數平均。文獻(FakeIDet)在未見資料庫 DLC-2021 達**文件級 0% EER**（整張圖 33%）。直攻 2 個未見私有類型，且呼應我們 EDA「訊號在局部」。
- **Tier A 納入 DINOv2 backbone**（foundation model 跨域 > ImageNet CNN），與 ConvNeXt 融合（局部+全域）。
- **Phase 2 重拍增強採 FHAG/BOIL + print-scan 通道**配方（頻域定向，非樸素模糊/JPEG）。
- **外部資料優先**：DLC-2021(重拍驗證) → IDNet(跨模板,六類 taxonomy) → SIDTD；小比例試 + type-LOO 驗證；注意合成 utility gap。

### Tier A — 全域分類器（主力，先做）｜鎖定 16GB 單卡
- Backbone：ConvNeXt-V2 **Tiny/Base** 或 EVA-02 **Small/Base** 或 Swin-Base（ImageNet 預訓練）。**避免 Large @ 高解析**（VRAM 不足）。
- 解析度：起步 **384px**，最終推到 **512px**；768 僅在記憶體允許時用切塊/局部裁切補充。
- VRAM 技巧：AMP(bf16) + gradient checkpointing + 梯度累積（有效 batch 32–64，實體 batch 8–16）。
- 集成 = **序列訓練**（一次一個模型存 ckpt），上限約 3–5 模型；受時間預算約束。
- 損失：BCE/Focal；二階段加 pAUC@low-BPCER 代理損失微調尾部。
- 推論：142,818 張 @512 fp16，需規劃吞吐（高效 dataloader，預估數小時）。

### Tier B — 取證分支（輔助）
- 噪聲殘差 (SRM/Bayar 卷積)、ELA、FFT 高頻 → 抓數位拼接。
- **關鍵**：這些對重拍脆弱，因此**也在重拍增強資料上訓練**使其優雅退化，作為訊號之一而非單一依賴。

### Tier C — 語義一致性（輔助，後期，高槓桿但工程量大）
- 臉區 vs 鬼影區 face embedding 距離（換臉→距離大）。
- OCR 關鍵欄位 → 性別 token vs 臉性別分類器、日期合理性、MRZ checksum。
- 5 個已知模板可用固定 ROI 抽取；但**保留通用 backbone**以應付未見模板。

### 融合
- Tier A 多模型 + B + C 的 OOF 預測 → 輕量 meta-learner (LR/GBM) 或加權平均。
- 最終以 OOF 做 isotonic/temperature 校準（**在重拍模擬驗證集上校準，非僅 digital**），專門優化 1%BPCER 操作點。

### 外部資料/模型模組（官方允許，須引用）
- **預訓練權重**：timm ImageNet（Tier A）、insightface/ArcFace（Tier C 臉比對）、PaddleOCR/Tesseract（Tier C 欄位）。
- **外部證件資料集**（補未見模板 + 重拍真實性，直攻 2 個未見私有類型）：候選 MIDV-2020、SIDTD / IDNet、DocXPand-25k、Fantasy-ID 等。
  - 用途：(a) 擴增模板/語言多樣性提升 OOD；(b) 真實重拍/翻拍樣本校準 domain。
  - 紀律：逐一**確認 license 相容 + 報告引用**；非公開專有資料禁止。
- 風險控管：外部資料分布不同，先以小比例混入並用 type-LOO 驗證是否真的提升 OOD 再加碼。

## 4. 重拍模擬增強（最高槓桿，獨立模組）

模擬「print → photo/scan」的 analog hole：
- 重採樣 + JPEG 雙重壓縮、半色調/moiré、模糊/失焦
- 光照梯度、鏡面反光/glare、白平衡/色偏、色差 (chromatic aberration)
- 透視微扭曲、感測器噪聲、列印網點
- 與標準增強（翻轉、輕度旋轉、cutout）併用；對真/假**同等**套用，避免增強洩漏標籤。

## 5. 驗證策略（防過擬合 digital 是重點）

1. **Leave-One-Type-Out CV**：訓 4 類驗第 5 類 → 估跨域泛化（**官方確認私有集含 2 未見類型，這是最貼合的代理**）。
2. **重拍模擬驗證**：對 hold-out fold 套重型重拍增強後量 FREUID → 估 digital→physical 遷移。
3. 標準 type-stratified k-fold → 估與公開 LB 的相關性。
4. 本地以 (1)(2) 為主信任私有表現；公開 LB(7,821, 可能全 digital) 僅參考、**不追榜**。
5. 先查 train↔test **近重複**，避免洩漏灌水。

## 6. 分階段路線圖

- **Phase 0 基建**：補齊測試下載；修路徑；實作 `metric.py`；建 CV split；深度 EDA（每類詐欺型態盤點、近重複偵測、品質檢查）。
- **Phase 1 Baseline**：單一 ConvNeXt @512 + 標準增強 + type-stratified 5-fold → 出第一個校準提交，建立 LB 相關性。
- **Phase 2 強健化**：重拍增強套件 + 多解析度 + 取證分支。預期最大增益。
- **Phase 3 語義**：臉↔鬼影、OCR 欄位一致性。
- **Phase 4 集成**：多模型 + meta-learner + 校準 + TTA → 最終。
- **Phase 5 可複現**：原始碼 + 技術報告 + 環境/容器（領獎硬性要求）。

## 7. 風險與對策

| 風險 | 對策 |
|---|---|
| 公開 LB(digital) 好看、私有(physical) 崩 | 重拍增強 + type-LOO CV；不追榜 |
| 過擬合 5 個模板，私有有未見類型 | 保留通用 backbone；增強多樣化；type-LOO 監控 |
| train/test 近重複灌水假象 | Phase 0 做 hash/embedding 近重複偵測 |
| 取證偽影對重拍脆弱 | B 分支用重拍增強訓練，僅作多訊號之一 |
| 測試影像下載不全 | Phase 0 先補齊 142,818 張 |
| 算力（69k 高解析×多模型） | 規劃 GPU 預算；先小解析驗證再放大 |
| 1%BPCER 尾部不穩 | 集成、TTA、pAUC 損失、OOF 校準 |

## 8. 程式架構（待 Phase 1 才寫）

```
freuid/
  configs/            # 每個實驗一個 yaml
  src/
    data/             # CSV 載入、路徑修正、CV split (type-LOO/stratified)
    aug/              # 重拍模擬增強管線
    models/           # backbones、取證分支、head
    forensics/        # SRM/Bayar、ELA、FFT
    semantic/         # 臉↔鬼影比對、OCR 一致性
    metric.py         # FREUID (AuDET, APCER@1%BPCER)
    train.py infer.py cv.py calibrate.py ensemble.py
  notebooks/          # EDA
  reports/            # 技術報告
  env/                # requirements / Dockerfile
```

## 9. 待確認問題

1. AuDET 確切積分定義（官方是否提供 metric code？）。
2. 是否允許/提供外部資料與預訓練權重（規則的 licensing 條款）。
3. 提交是 CSV(全 142,818) 還是 Kaggle code-competition 線上推論？（目前看似 CSV）。
4. 計算資源上限（本地 GPU 規格 / Kaggle 限制）。
5. 私有測試是否含未見證件類型 / 更高比例 physical（影響 type-LOO 權重）。
