# 外部資料 / 預訓練模型候選清單（FREUID 2026）

> 規則：公開可取得、license 相容、且**報告中引用所有來源**即可使用；專有非公開資料禁止。
> ⚠️ 以下 license 標註為「待逐一核實」——使用前務必到原始來源確認當前授權條款與是否允許競賽/衍生使用，並記錄取得日期與 commit/版本。

## A. 預訓練模型（風險低，幾乎必用）

| 資源 | 用途 | License（待核實） | 備註 |
|---|---|---|---|
| timm ImageNet 權重 (ConvNeXt-V2, EVA-02, Swin) | Tier A backbone | Apache-2.0 / model-specific | EVA-02 部分權重含 LAION 預訓練，需確認子授權 |
| insightface / ArcFace (buffalo_l 等) | Tier C 臉↔鬼影比對 | 非商業研究用為主 | 競賽為非商業研究，需確認條款 |
| PaddleOCR / Tesseract | Tier C 欄位 OCR | Apache-2.0 | 多語/阿拉伯文支援需額外語言包 |

## B. 跨域模板/版面多樣性（攻 2 個未見私有類型，價值高）

| 資料集 | 內容 | License（待核實） | 對本賽用途 |
|---|---|---|---|
| **MIDV-2020** | 1000 份 mock 證件、10 類、含拍攝/掃描/合成 | 研究用（基於 Wikipedia specimen） | 擴增模板/語言，提升 OOD |
| **MIDV-500 / MIDV-2019** | 50 類 mock 證件影片 | 研究用 | 更多模板多樣性 |
| **DocXPand-25k** | 25k 合成證件卡（多版面） | CC BY-NC-SA 4.0（待確認） | 版面/欄位多樣性；NC 符合本賽 |
| **IDNet (2024)** | ~60 萬合成證件、多類、多種偽造型態 | 研究用（待確認） | 直接含偽造正/負樣本，最貼近任務 |
| **SIDTD** | 基於 MIDV 模板的合成真/偽證件 | 研究用 | 偽造型態多樣 |

## C. 重拍 / print-and-capture 真實性（攻 analog hole，價值高）

| 資料集 | 內容 | License（待核實） | 對本賽用途 |
|---|---|---|---|
| **DLC-2021** (Document Liveness Challenge) | 原件 vs 螢幕/列印重拍 | 研究用 | **直接對應重拍偵測**，校準 domain |
| **KID34K** | 韓國證件 spoof/recapture | 研究用 | 重拍/翻拍真實樣本 |
| **MIDV-Holo** | 全像圖偽造偵測 | 研究用 | 物理防偽特徵（部分相關） |

## D. 不可用 / 高風險（先排除）

- 真實國民身分證資料（如部分 BID 巴西證件）：含個資、授權受限 → 排除。
- 任何「需簽 NDA 或非公開」資料 → 規則明文禁止。

## 使用紀律（Checklist，供 Phase 2 引入時逐項勾選）

- [ ] 確認原始來源 license 文字 + 截圖存證 + 記錄取得日期
- [ ] 確認允許「競賽用途 + 衍生模型」與「非商業研究」相容
- [ ] 在 `reports/` 技術報告列出每個來源（名稱、版本、URL、license、用途）
- [ ] 先以**小比例**混入 train，用 type-LOO CV 驗證確實提升 OOD 再加碼
- [ ] 外部資料分布偏移大 → 監控是否反而傷害校準/操作點

## 引入優先序（建議）

1. **Phase 2**：DLC-2021 / KID34K（重拍真實性）→ 最直接補 domain gap。
2. **Phase 2–3**：MIDV-2020 + IDNet（模板多樣性）→ 補未見類型泛化。
3. **Phase 3**：ArcFace + PaddleOCR（Tier C 語義特徵）。
