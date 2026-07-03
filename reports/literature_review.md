# 文獻回顧 — ID 證件詐欺/PAD 偵測（FREUID 2026 相關）

回顧日期 2026-06-09。重點：可直接落地到本賽的方法、增強、外部資料、指標。

## 最重要的可落地發現（依槓桿排序）

### 1. ⭐ Patch-based DINOv2 → 跨域泛化的關鍵（FakeIDet, arXiv 2504.07761 / 2508.11716）
- 把證件切成 patch（64–128px），用**凍結的 DINOv2** 抽每個 patch 特徵→sigmoid 分類→**patch 分數取平均**得文件級分數。
- **跨資料庫結果驚人**：在未見的 DLC-2021（不同裝置/國家/模板/實體攻擊）上，**文件級 EER = 0%**；而同一 backbone 餵整張圖是 33% EER。
- 為何泛化好：patch 去除模板專屬特徵（背景/版面/文字內容），逼模型學**局部取證線索**（列印網點、螢幕像素、護膜缺陷）。
- 細節：patch 縮到 224 餵 DINOv2、BCE、Adam lr 1.5e-4、150 epoch early-stop；128>64>32 表現（64 與 128 僅差 ~1% EER）；ViT 對 patch 尺寸遠比 ResNet 穩。
- **對本賽**：直攻「2 個未見私有類型」+ 呼應我們 EDA「訊號在局部、模板全域結構無鑑別力」。→ **列為核心方法（新增 patch-DINOv2 分支）**。

### 2. ⭐ Foundation model（DINOv2/CLIP）> ImageNet CNN 的跨域能力（系統綜述 arXiv 2511.06056）
- DINOv2 zero-shot border-attack EER 4.33%、fine-tuned print 1.28%。
- **DINOv2(局部) + 傳統 CNN(全域) 融合**把 IDNet 上 EER 從 27.86% 降到 8.25%。
- 結論：**預訓練多樣性比領域內資料更重要**。→ Tier A 應納入 DINOv2 backbone，並與 ConvNeXt 融合。

### 3. ⭐ 重拍增強：FHAG + BOIL（頻域定向）
- **Frequency-domain Halftoning Augmentation (FHAG)** + **Band-of-Interest Localisation (BOIL)**：只在「取證偽影所在的頻帶」做 halftone 混合 + 加噪，跨域 EER 顯著下降，優於樸素 FDA。
- **ForgeNet/print-scan channel**：pre-compensation 加入列印-掃描通道的色彩/雜訊失真；inverse-halftoning 去網點防 moiré。
- → Phase 2 重拍模擬增強的具體配方（非泛泛的模糊/JPEG）。

### 4. 跨類型崩潰是頭號風險（IDNet, arXiv 2408.01690）
- 模型嚴重依賴 type-specific 特徵（背景/透明度/尺寸）：face-morph within-type 98.65% → cross-type 50.55%；text-replace 100%→50-60%。
- → 我們的私有 2 未見類型風險被量化證實。對策：foundation model、patch-based、取證感知增強、外部多模板資料。

## 詐欺型態分類（與我們看圖一致，IDNet 六類可作 taxonomy）
1. Face morphing（兩臉對齊+混合 0.5）
2. Portrait substitution（換成不合格照）— 我們在 MAURITIUS 例子看到
3. Text-field replacement（字體/大小/對比改；難例做到內部一致）
4. Inpaint-and-rewrite（遮罩重繪欄位、保留背景）
5. Crop-and-replace（跨證件搬欄位）
6. Mixed（text + morph/substitution）
> FREUID 還加上 **print-and-capture 重拍** 與 **GenAI 多模態編輯**，把上述數位痕跡壓抑掉。

## GenAI 偽造的現況（arXiv 2601.00829, 2026-01）
- 測 Stable Diffusion / Qwen / Flux / Nano-Banana 等 t2i + i2i。
- 發現：GenAI 能模擬**表面美學**但無法達到**結構/取證真實性** → 純 GenAI 偽造仍可被偵測。
- → 印證我們走「語義/結構不一致」是對的；難點是 **GenAI 編輯 + 重拍** 疊加後數位痕跡被壓抑。

## 可用外部資料（公開、license 相容，須引用）
| 資料集 | 用途 | 備註 |
|---|---|---|
| **IDNet** (Zenodo, 837k, 20 類 US+EU) | 跨模板多樣性 + 六類偽造 taxonomy | 合成無 PII；但 bona-fide 也是合成（注意 utility gap）|
| **DLC-2021** (1424 影片, 列印/螢幕重拍) | 重拍真實性 + 重拍域驗證 | FakeIDet 用它證明 patch 泛化 |
| **SIDTD** (github Oriolrt) | Crop&Replace + inpaint 偽造 | |
| **MIDV-2020 / KID34K** | 模板/重拍多樣性 | KID34K 含實體塑膠卡重拍 |

⚠️ **Reality / Synthetic-utility gap**（綜述強調）：合成 bona-fide（像我們訓練集的 SPECIMEN 樣本）會混淆分類器；測試集偏真實實拍。→ 對訓練資料做**重拍增強**橋接，且小心模型把「合成感」當 bona-fide 捷徑。

## 指標對照（ISO/IEC 30107-3）
- APCER（攻擊被當真）、BPCER（真件被當攻擊）、EER、ACER。
- IJCB 競賽多用 EER / BPCER@APCER 與 AVRank(BPCER10×0.2+BPCER20×0.3+BPCER100×0.5)。
- **FREUID = APCER@1%BPCER（極嚴格 1% 真件誤拒下測漏網率）+ AuDET** → 比 EER 更嚴；操作點/尾部為勝負點（與我們本地實驗一致）。
- 難度標定：IJCB 2024 最佳 EER 21.87%、2025 Track1 11.34%、Track2 open-set 6.36% → 即便 SOTA 也難；1%BPCER 會更硬。

## 對本賽方案的具體調整（→ 併入 PLAN）
1. **新增 patch-DINOv2 分支**為核心（最高泛化槓桿，攻未見類型）。
2. **Tier A 納入 DINOv2 backbone**，與 ConvNeXt 融合（局部+全域）。
3. **Phase 2 重拍增強採 FHAG/BOIL + print-scan 通道**配方（非樸素）。
4. **外部資料優先序**：DLC-2021（重拍驗證）→ IDNet（跨模板）→ SIDTD；皆小比例試 + type-LOO 驗證。
5. 持續以 **type-LOO FREUID** 為選型主依據；警惕「合成感」捷徑與 utility gap。

## 來源
- 系統綜述：arxiv.org/abs/2511.06056
- IDNet：arxiv.org/abs/2408.01690
- FakeIDet：arxiv.org/abs/2504.07761、2508.11716
- 多模態取證解耦：arxiv.org/abs/2404.06663
- GenAI 偽造：arxiv.org/abs/2601.00829
- DocXPand-25k：arxiv.org/abs/2407.20662
