# FREUID Challenge 2026 — 進度報告

最後更新：CLIP-LoRA@224 LB 0.191；取得同學 SigLIP-2 配方（0.03296），已轉向移植+升級 @512（訓練中，ETA ~11h）。

---

## 1. 公開 LB 記分板（越低越好；只算 7,821 張公開測試，私有填 dummy）

| # | 模型 / 提交 | 公開 LB | 備註 |
|---|---|---|---|
| 1 | baseline ConvNeXt-V2-Tiny @320 | 0.269 | 第一個提交 |
| 2 | R0：+ 重拍增強 | 0.357 | ❌ 更差（增強摧毀數位表現）|
| 3 | patch-DINOv2（凍結+attention-MIL）| 0.454 | ❌ 最差（切 patch 打碎全域脈絡）|
| 4 | 集成 320+448（2 成員 rank）| 0.246 | ✅ 高解析抓文字竄改 |
| 5 | **集成 320+448+512+448f1（4 成員）** | **0.209** | ✅ **CNN 路線最佳** |
| 6 | 集成 6 成員（+320f2,512f1 冗餘）| 0.214 | ❌ 同解析度冗餘成員稀釋 |
| 7 | 集成 8 成員（+2 弱成員）| 0.237 | ❌ 弱/未收斂成員拖累 |
| 8 | 集成 + DINOv2（非飽和 transformer）| 0.237 | ❌ 非飽和成員稀釋公開（但私有避險）|
| 9 | CLIP ViT-L/14 + LoRA + hflip TTA @224 | 0.191 | ✅ 勝過 CNN 集成，但離同學 0.130 有距離 |
| 10 | CLIP@336 / @448 | （中止） | 解析度路線被 #11 碾壓，砍掉省 GPU |
| 11 | **SigLIP-2 SO400M @512 + DoRA（同學配方移植+升級）** | **0.02667**（rank 23）| 🎯 勝同學 0.03296 → 512/MLP/EMA 升級有效；ho_auc 0.911、ho_gap 0.77 |
| 12 | SigLIP-2 @378（patch14，同配方）| 待上傳 | 集成第二成員；ho_auc 0.899、ho_gap 0.81 |
| 13 | 集成 512+378（**rank-mean**）| **0.06608** | ❌ **比單模 512 差 2.5x**！rank-mean 把雙峰信心壓成均勻(std 0.48→0.27)→ APCER@1%BPCER 操作點爆掉 |
| 14 | 集成 512+378（**mean 機率平均**）| **0.03480** | ✅ 方法修對(0.066→0.0348) 但❌ 仍輸單模 → 378 是稀釋型弱成員，無互補性 |
| 15 | **SigLIP-378 單模（同 512 配方）** | **0.08104** | 🔑 **決定性對照**：同配方僅差解析度，378→512 = 0.081→0.027（3x）→ 解析度是主槓桿，集成是死路 |
| 17 | **SigLIP-512 + attn-only + 無 EMA（同學配方@512）** | **0.04408** | 🔑 配方 ablation：在 512 **輸給**我們 MLP+EMA(0.027) → 配方在 512 是**有益**非有害；疑雲解除 |
| 18 | SigLIP-512 內插 @768（純解析度測試）| **本地判死** | ❌ 內插傷預訓練：canary 拖到 ep2 才飽和、ho_auc 崩到 0.53、提交中間帶 965 → 內插≠原生 |
| 19 | 512 multi-crop TTA（半圖, mean）| **本地判死** | ❌ 半圖 OOD（右半 0.35 vs full 0.63）→ mean 抹糊雙峰（中間帶 280→6725），操作點爆 |
| 20 | 512 multi-crop TTA（半圖, max）| **本地判死** | ❌ max 保住詐欺端但良民乾淨低端全毀（<0.01: 2830→0）→ 操作點一樣崩 |

**🧱 第五次撞牆：post-hoc 技巧全敗（rank集成/弱成員集成/768內插/multi-crop mean/max）。規律＝任何事後混 predictions 都抹糊超自信雙峰，APCER@1%BPCER 零容忍。停止後處理，只投資更好的單一原生模型。**

**🧱 第六次撞牆：3 範式深融合(+forensic)=0.03298，8x 差於 2-way(0.00426)。即使是「深」特徵融合,加一個高相關(0.974)又帶猶豫的成員照樣抹糊操作點(中間帶 263→669)。規律＝成員必須「正交且銳利」,不只正交。forensic 太愛猶豫。pseudo-FREUID 閘在融合自身成員當偽標籤時會自我參照失效,不能當綠燈。冠軍仍為 2-way fusion 0.00426。**

**🪦 身分比對閘證冗餘：face_consistency 頭(sim<0.2 抓換臉)的 193 個高信心詐欺,冠軍融合已把 191 個打到 >0.9,只剩 2/7821 不同 → 融合早已吸收換臉訊號,閘無用(且 2 個分歧正是 OOD 上頭可能錯的地方)。3 連敗確認公開加成員已飽和。**

**🧱 第七次撞牆：「更強/更正交 VL」+「演算法」雙線皆敗。**
- **FGTS-on-VL(演算法)**:Fisher token 選擇套到 SigLIP 只有邊際變化(pseudoAUC 0.982→0.986)。原理:SigLIP 對比學習用 *pooled* 目標,global pool 本就聚合好線索;FGTS 是 DINOv3(密集 SSL,局部線索被稀釋)專屬槓桿,**非 VL 槓桿**。
- **DFN5B ViT-H@378(更強 VL backbone,訓練 9.75h)**:單模較 SigLIP 弱(pseudoAUC 0.950 vs 0.982)但**與 DINOv3 更正交**(corr 0.809 vs siglip 0.909)。然而 `fusion dino⊕dfn5b` 被真實重拍仲裁否決:genuine_p 0.002→**0.640**(打破零誤報安全性質!)、recap AUC 0.881→0.798、mid 0→11/20。DFN 單模重拍弱(ho 0.810<siglip 0.917)拖垮融合。**「更正交但較弱」不划算——成員仍須夠強夠穩。**

**🧱 第八次撞牆:全量微調(full-FT)證偽,DoRA 是適配甜蜜點。** DINOv3-L@512 解凍 303M(lr2e-5/head1e-3,2ep)公開 LB=**0.03948**——比 DoRA 單模(0.01134)差 3.5x、比冠軍融合差 9x,即使它**最銳利(mid163<冠軍263)+ 重拍 ho_auc 0.935 史上最佳**。原因:train 99.97%數位=特定 generator,full-FT 學死 generator 痕跡→公開(不同 generator)操作點崩,canary 飽和(roc1.0)藏不住。再證:ho_auc/銳利度不預測公開 LB。凍結(LB1.0)與 full-FT(0.039)兩極皆差,**DoRA 唯一甜蜜點**。`--full_ft` + `src/infer_fullft.py` 已實作但此路作廢。

**🔒 最終提交鎖定 = 2-way fusion 0.00426。** 私有重拍軸用「真實重拍 n=20(非模擬,刻意避開 recapture.py 因它害公開 LB)」驗證:`eval_robust_fusion` 顯示融合在真實重拍 genuine_p=0.002(**零誤報**,APCER@1%BPCER 最關鍵性質)、AUC 0.881(遠勝 DINOv3 單模 ho 0.738,近 SigLIP 0.917=融合 head 學會靠穩健特徵)、中間帶 0/20(不猶豫)。冠軍在所有可信軸(公開 LB / 數位 AUC 1.0 / 真實重拍零誤報)皆最佳。剩餘唯一真槓桿=DLC-2021 真實重拍當仲裁(需下載)。

| 23 | **DINOv3 ViT-L @512 + DoRA** | **0.01134** | 🏆 **新最佳(#3)！SigLIP 0.0267 的 2.4x**。DINO 細粒度 artifact 抓公開硬樣本 |
| 24 | AIMv2-large @448 + DoRA | **0.07436** ❌ | 比 SigLIP 還差！VL 家族非答案，是 DINOv3 的 SSL 細粒度獨強 |
| 25 | DINOv3 @640（RoPE 高解析）| **0.02499** ❌ | 比 @512(0.01134) 差一倍！高解析**有害**非到頂。@768 免跑。512 是甜蜜點 |
| 26 | DINOv3 @512 + SIDTD 外部硬詐欺 | **0.05967** ❌ | 差 5x！域不匹配(歐洲證件 17%)推離非洲分佈。外部資料路線止步(2-epoch confound 不足解釋 5x)|
| 27 | FGTS Fisher token 選擇 on DINOv3+DoRA | k64=0.00956 | 0.01134→0.00956。k 甜蜜點=k64。 |
| 28 | 集成 DINOv3-FGTS + SigLIP（0.7/0.3 mean）| 0.00595 | 不同架構(SSL+VL)真互補。 |
| 29 | **特徵融合 DINOv3-FGTS ⊕ SigLIP（2176d 聯合 head）** | **0.00426** 🏆 | 深融合 ≫ 淺權重混(0.00595)!2-way 追平同學 3-way。學出的融合勝手調權重 |
| 30 | Forensic Bayar-ConvNeXt（第三範式:噪聲鑑識）| canary AUC 1.0 | corr 0.792 vs fusion=正交(VL/SSL 對之間 0.92-0.96)。門控太危險(77 覆寫,OOD 誤判)→ 改特徵融合 |
| 31 | **3 範式深融合 dino⊕siglip⊕forensic** | **0.03298** ❌ | 慘敗！8x 差於 2-way(0.00426)，正好掉回 SigLIP 基線 0.03296。forensic 與 fusion corr 0.974=大多同意 dino+siglip，只在邊緣稀釋信心(中間帶 263→669)→ 操作點被 forensic 的猶豫拖爆。教訓:成員相關性高+加不確定性=毒藥(非互補)。pseudo-FREUID 閘給它略好(0.0334<0.047)=**閘自我參照失效**(兩融合都含 dino 偽標籤源),不可信。要再用 forensic 須先大幅提升其精準(它太愛猶豫) |

工具:`src/train_forensic.py`(Bayar+ConvNeXt)、`src/fusion.py`(多成員特徵融合+快取,支援 ViT token / ConvNeXt forensic 成員)。

### 🔄 策略重大更正（官方+用戶更正）：公開測試也含實拍
- **公開 ≠ 純數位**（舊假設推翻）：公開=數位+實拍混合；私有=實拍主軸+2未見類型+罰 generator 痕跡。
- **∴ DINOv3 拿下 0.01134（含實拍的測試）= 它數位+實拍都強**。我那些「DINOv3 實拍弱」全是爛代理（循環模擬+n=20）誤判。
- **重拍模擬增強應放棄**：它傷 public(0.041)=優化我的爛模擬、真實更差。
- **DLC-2021 真實重拍 = 必需仲裁**（取代爛代理）。
- **本地代理教訓**：canary 飽和藏數位差、模擬重拍≠真實、n=20 太偏 → **數位/實拍軸只能靠公開 LB 提交判斷**。

| 21 | **SigLIP-512 + 重拍增強（私有拍攝軸）** | **私有 proxy 勝 baseline** | 🎯 真實重拍 ho_auc 0.911→**0.976**、數位不退化。私有更佳賭注 |
| 22 | **type-LOO 診斷（未見類型軸）** | **未見類型 AUC 0.9995** | 🎯 只訓 3 類型，測 MAURITIUS/ID+GUINEA/DL=近乎完美泛化。**複合(未見類型+重拍)rc0.9 仍 AUC 0.99、真實重拍 0.93** → 私有兩軸都穩 |

### 🧬 換 backbone（最大槓桿）：DINOv3 ViT-L
搜尋結論：**DINOv3**（Meta 2025）對偽造偵測理論上勝 SigLIP——文獻指 DINO 捕捉細粒度局部 artifact > CLIP/SigLIP（語義但對 artifact 不敏感），且 FGTS 論文（arXiv 2511.22471）證明 DINOv3 跨生成器偽造 SOTA = 正是私有「不依賴 generator 痕跡」。RoPE 位置編碼 → 高解析不會像 SigLIP 內插崩。`vit_large_patch16_dinov3.lvd1689m` 24 blocks/1024-d，@512。**結果：DINOv3 public = 0.01134（SigLIP 0.02667 的 2.4x 進步！新最佳，#3）**。⚠️ 前述「DINOv3 更差」是錯的——那只看了重拍 ho_auc(0.738，私有軸);public/數位軸 DINOv3 大勝(canary 飽和本地測不出，被誤導)。文獻對:DINO 細粒度 artifact 捕捉 > SigLIP，抓到公開硬樣本(文字/composite)。**權衡：DINOv3 數位強(0.01134)但重拍弱(ho 0.738 vs SigLIP 0.917)。** 教訓:本地重拍 proxy ≠ public 軸,canary 飽和會藏住數位差異 → 數位軸只能靠提交。
**進行中：AIMv2-large @448**（`aimv2_large_patch14_448.apple_pt`，310M VL backbone，DINO 不同族、benchmark 贏 CLIP/SigLIP）。用 SwiGLU(fc1_g/fc1_x)→ build_model 已改成通用 DoRA 注入(依存在的 Linear 名,相容 SigLIP+AIMv2)，120 層/7.04M。native 448=1024 tokens 同 SigLIP 成本。對照 0.02667。

### ✅ 私有軸雙確認 + 外部資料決策更新
- **未見類型**：type-LOO 證明配方近乎完美泛化(clean AUC 0.9995，非循環)→ 私有 2 新類型在數位軸非問題。**→ SIDTD/IDNet(補類型多樣性)不需要,省 400GB 下載。**
- **拍攝**：重拍增強真實重拍 0.911→0.976/0.93(n=20)。模擬曲線部分循環。**→ 唯一值得下載=DLC-2021(真實重拍,破 n=20)。**
- 工具：`eval_robust_siglip.py --loto_types` 可測複合 OOD。

## 🔭 重大策略轉向（2026-06-15）：公開 5% ≠ 私有 95%
賽方明文：私有測試 = **2 個未見證件類型 + 重拍/實拍為主**，**懲罰依賴 generator-specific 痕跡**。我們把公開 LB 0.033→0.027 壓的正是私有刻意淡化的數位軸。
- **clean val 飽和 = 模型騎數位來源 artifact = 賽方警告的「generator 痕跡」**。
- **私有軸 proxy 建好了**（`src/eval_robust_siglip.py`，免提交）：0.02667 模型在重拍下退化——
  | clean | rc0.3 | rc0.5 | rc0.7 | rc0.9 | 真實重拍(20) |
  |---|---|---|---|---|---|
  | AUC 1.0 | 0.924 | 0.883 | 0.843 | 0.831 | **0.917** |
  → clean 完美但重拍掉到 0.83 = 私有 headroom。

對照：**同學 CLIP-L/14+LoRA = 0.130 → SigLIP-2 SO400M@378+DoRA = 0.03296**（`src/train_siglip.py`）→ backbone+配方是決定性槓桿。

---

## 2. 核心診斷（這趟學到的關鍵）

1. **資料會飽和但考 OOD**：clean val 對幾乎所有模型都飽和(AUC~1.0)，因為模型抓到「跨模板共通的生成來源 artifact」。type-LOO 也飽和（未見類型仍同來源）。→ clean/type-LOO 無法選型。
2. **公開測試硬樣本 = 數位合成 + 文字竄改**（看圖確認），**不是 print-and-capture**：
   - 肖像合成/貼照/GenAI 臉、主照≠鬼影、性別欄≠照片 → full-image CNN 抓得到。
   - 文字欄位竄改（刪除線/覆寫姓名/日期/地址）→ 需**高解析**（320 下文字痕跡~2px 看不見）。
3. **詐欺手法 ↔ 模板結構相關**（20 張盤點）：EGYPT/MAURITIUS（有鬼影照）→ 換臉/性別不一致；BENIN/GUINEA/MOZAMBIQUE → 貼照/文字/GenAI。~20-25% 是 subtle 無明顯破綻。
4. **換架構/加重拍都更糟**：R0 重拍增強、patch-DINOv2 都比 baseline 差 → 留在 full-image 範式。
5. **集成的變異降低對 FREUID 操作點極有效**，但**只有「加入新能力的成員」有用**（高解析→文字）；同解析度冗餘成員、弱/非飽和成員會稀釋拖累。
6. **🚨 backbone 才是最大槓桿**：ConvNeXt-Tiny 集成卡 0.209，CLIP ViT-L/14 視覺-語言特徵能把**語義理解**泛化到 OOD → ~0.13。

---

## 3. 目前主力：SigLIP-2 SO400M @512 + DoRA（`src/train_siglip512.py`）

同學 0.03296 配方（`src/train_siglip.py`，SigLIP-2@378 + attn-only DoRA）的本地移植，升級三處（皆已標註）：
- **[512]** `vit_so400m_patch16_siglip_512.v2_webli`：原生 512（無 pos-embed 內插），文字竄改像素 ~1.8x
- **[MLP]** DoRA r16 α32 注入 attn+MLP 共 **108 層**（7.98M 可訓；同學只打 attn 54 層）
- **[EMA]** ModelEmaV3(0.9995)，eval/推論用 EMA 權重
- 忠實保留的配方核心：**全量訓練**（68,985，只留 0.5% canary 347 張 + 20 重拍當 val——不再燒 20% 當 fold val）、type×class 加權採樣 + pos_weight、RandomResizedCrop(0.7–1.0)+ColorJitter+hflip、3 epochs、lr 2e-4 warmup 1000、hflip TTA
- 已捨棄：temperature calibration（單調變換，對排序型指標 AuDET/APCER@BPCER 無效）
- 資源實測：VRAM ~8GB（grad ckpt）、~2.85s/step、共 12,933 steps ≈ **10.5h**，訓完自動推論寫 `subs/siglip512_dora.csv`

前主力 CLIP-L/14+LoRA@224 = LB 0.191（`subs/clip_lora.csv`）；336/448 解析度路線已中止。

---

## 4. 驗證策略（誠實現況）

- **clean val 對所有模型飽和** → 無法選型。
- **rc-sim proxy 無效**、**type-LOO 飽和**。
- **20 張真實重拍 holdout**（is_digital=False，train.py 自動排除+每 epoch 報 `ho_auc`/`ho_gap`）= 唯一真實重拍訊號，但 n=20 太 noisy（不可用於選 epoch；改用「選最收斂 epoch」）。是**私有/重拍軸**的粗略 proxy，非公開（數位合成）軸。
- **公開 LB 是唯一可信的數位合成軸判準**，但省著用、別過擬合（私有 = 2 未見類型 + 重拍，才是最終排名）。

---

## 5. 程式基建

```
src/
  experiment.py        實驗追蹤（全超參→registry.csv）+ ho_auc holdout
  data/paths.py        路徑修正載入器（巢狀路徑/缺檔）
  data/dataset.py      Dataset + 輕度增強（aspect-preserving）
  data/cv.py           type-LOO + stratified folds
  models/factory.py    timm backbone + 單 logit（支援 ViT img_size + LoRA）
  models/lora.py       LoRA 實作（無 peft 依賴）
  models/patch_dino.py patch-DINOv2（已放棄）
  aug/recapture.py     重拍模擬增強（已放棄用於訓練）
  losses.py metric.py  Focal/BCE；FREUID(AuDET+APCER@1%BPCER)
  train.py             AMP/EMA/cosine/holdout；arch=single/patch
  infer.py             集成（多 run）+ 多尺度 TTA + hflip + rank/mean
  precompute_dino.py + train_patch_head.py  凍結嵌入快取（patch 實驗）
  eval_robustness.py   重拍 robustness curve
```

關鍵環境：單卡 RTX 5060 Ti 16GB；torch 2.11 nightly cu128（Blackwell 必需）；Windows commit/worker 限制 → num_workers=2、persistent_workers=False。

---

## 6. 下一步（轉攻私有軸 OOD；公開 LB 降級為 sanity）

**新判準 = `src/eval_robust_siglip.py` 的重拍退化曲線 + ho_auc + type-LOO，不是公開 LB。**
1. ✅ **重拍穩健訓練完成**（run siglip512_rcaug）：真實重拍 0.911→0.976、數位不退化。私有拍攝軸更佳賭注（n=20 待 DLC-2021 確認）。
2. **進行中：type-LOO 診斷**（run siglip512_loto_recap，留出 MAURITIUS/ID+GUINEA/DL，重拍 aug ON）→ canary_freuid 直接量「未見類型泛化」(5 類型中的 2 個當 proxy)。若 ~0=泛化好；若高=類型記憶問題→需外部多樣性。
3. **外部資料已確認可用**(CC BY-SA 2.5,引用)：DLC-2021 真實重拍→驗證；SIDTD(`github.com/Oriolrt/SIDTD_Dataset`)真實模板+竄改→未見類型訓練增量。
4. 重拍強度掃描（proxy 免提交調）。凍結 SigLIP+DoRA 本身利 OOD。
5. **最終提交候選**：私有用 recapture-aug 模型(+未來 type-robust 版)；公開保底 0.02667。

### （舊）解析度路線結論：撞實務天花板
512 是 patch16 原生頂；896 原生太慢（~62-77h，加速槓桿 no-ckpt OOM / compile 無 Triton 皆失敗）；內插已證實傷特徵。**解析度暫時封頂在 512。**

## 附：解析度探索歷程（SigLIP-512 = 0.02667 為公開最佳單模）

🚨 **集成教訓更新（rank-mean 對此 metric 是陷阱）**：FREUID = AuDET + APCER@1%BPCER，後者是 1% BPCER 操作點的指標，對「信心分佈」極敏感。單模 512 預測高度雙峰(0.0001 / 0.9999)＝操作點乾淨。**rank-mean 把每個成員映成均勻 rank 再平均 → 摧毀雙峰 → APCER 爆炸**（0.027→0.066）。改用 **mean（機率平均）或 logit-mean**：只有「成員不同意」處才往中間移，信心一致處保留。`src/combine_subs.py --method {mean,logit}`。

### 🔑 決定性發現：解析度是主槓桿，集成是死路
- **同配方對照**：378=0.08104、512=0.02667（只差解析度/patch，差 3x）。
- **集成全敗**：rank=0.066、mean=0.0348，都輸單模 512 → APCER@1%BPCER 是操作點指標，獎勵單一銳利模型；混入弱成員(378)只會稀釋。**集成路線封存**。
- **seed-1234 已砍**（為已死的集成燒 GPU，沒意義）。

### 解析度路線圖（記憶體都 3–5GB，瓶頸是時間）
| 目標 | tokens | s/iter | 3-ep 時間 | 狀態 |
|---|---|---|---|---|
| 512 原生 | 1024 | ~1.4 | 10.7h | ✅ 0.02667 |
| 640 內插 | 1600 | 1.92 | 18h | 備選 |
| **768 內插** | 2304 | 2.14 | **31h** | 🎯 訓練中 |
| 896 原生+OCR | 4096 | 5.47 | 78h | ❌ 太慢（除非 1-ep 賭）|

### 🔑 配方疑雲解除（但勿過度詮釋——矩陣有一格不乾淨）
| | attn-only 無EMA | MLP+EMA（我們）|
|---|---|---|
| **378 (patch14)** | 0.033 ⚠️**同學整套 pipeline** | 0.081（我們）|
| **512 (patch16)** | 0.044（我們）| **0.027**（我們）|
- **唯一受控對照 = 512**（同腳本只切 `--attn_only --ema_decay 0`）：我們配方 0.027 **乾淨贏**瘦配方 0.044 → 配方在 512 有益、非有害。「配方有害」假設否定。
- ⚠️ **不要詮釋成「配方×解析度交互」**：378-瘦那格(0.033)是同學在 Kaggle 的整套不同 pipeline(fp16+scaler、temp-cal、batch16、不同 timm/環境)，非我們腳本切旋鈕。跟我們 378(0.081)比是混了≥4 個變數，無效。
- **我們從未在自己 pipeline 跑「378+瘦配方」**，故「我們配方在 378 較差」未被證實。要釐清需補這一格（~8h+1 提交，純學術，不急）。
- **我們 512 + MLP+EMA = 0.02667 是受控確認的全域最佳**（非僥倖）。

### 解析度路線（配方已確認，可放心 scale）
1. **進行中：768 內插**（512-p16 backbone + 我們配方 + `--img_size 768`）= 真正的**同 backbone 純解析度**測試。看 1.5x 解析能否壓過 0.0267。
2. 若有效 → 896 原生+OCR 1-epoch 賭、非方形長寬比（修證件壓扁）。
3. 若無效（內插≠原生增益）→ 512 單模其他改進。
工具：`src/probe_speed.py`、`build_model --img_size`、`--attn_only`、`--ema_decay 0`。

教訓更新：**記憶體從來不是瓶頸**（grad ckpt 下 224~512 都 ~5-8GB，util 99%）——解析度/模型大小的代價是純時間。

注意：**私有測試（2 未見類型 + 重拍）才是最終排名**——CLIP 的視覺-語言特徵 + ho_auc 0.92 對私有應有利，但別過擬合公開 LB。

---

## 7. 待確認/風險

- AuDET 確切定義（現採線性=1−ROC_AUC）。
- 完整 `test/`（142,818）尚未下載，目前只能對 7,821 公開評分、私有填 dummy（host 證實先填 dummy、私有集稍後釋出再重測）。
- 領獎需：原始碼 + 技術報告 + 環境/容器（本報告 + registry + reports/ 為基礎）。
