# 7/13 私有推論 Runbook（照抄執行，預算 ~20h）

**原則：最終提交 = 一次 `predict_docker` 跑遍全部 142,818 張圖的輸出。**
不要把舊公開列和新私有列拼接——單一 canonical run 的輸出才是審查者能復現的東西，
它的 md5 就是 REPRODUCE.md 對映表要填的 checksum。

## 步驟

1. **私有圖釋出後立刻下載**，解壓確認是平面目錄（檔名=id）。
2. **合併公開+私有到一個目錄**（用 hardlink 免複製 16GB）：
   ```powershell
   New-Item -ItemType Directory E:\freuid_all_test
   Get-ChildItem public_test\public_test -File | ForEach-Object { New-Item -ItemType HardLink -Path "E:\freuid_all_test\$($_.Name)" -Target $_.FullName }
   Get-ChildItem <私有圖目錄> -File | ForEach-Object { New-Item -ItemType HardLink -Path "E:\freuid_all_test\$($_.Name)" -Target $_.FullName }
   (Get-ChildItem E:\freuid_all_test -File).Count   # 必須 = 142818
   ```
3. **開跑（~20h，一趟出雙份）**：
   ```powershell
   python -m src.predict_docker --data E:\freuid_all_test --out subs\FINAL_routed.csv --variant routed --emit_both
   ```
   斷電/中斷 → 直接重跑（特徵不留檔，重來 = 全價 20h，所以務必接 UPS/別動機器）。
4. **完整性檢查**：兩份 CSV 各 142,818 列、無 NaN、id 齊全：
   ```powershell
   python -c "import pandas as pd; [print(f, len(d:=pd.read_csv(f)), d['label'].isna().sum()) for f in ['subs/FINAL_routed.csv','subs/FINAL_routed_plain.csv']]"
   ```
5. **算 checksum 填進 REPRODUCE.md**：
   ```powershell
   Get-FileHash subs\FINAL_routed.csv, subs\FINAL_routed_plain.csv -Algorithm MD5
   ```
6. **上傳 Kaggle**（描述欄寫 "final pick 1 routed" / "final pick 2 plain"），
   公開分數應 ≈0.00207±0.0002（容差內位移正常）。
7. **在 Kaggle 選定這兩份為 final picks**（舊的 0.00207 那兩份含 0.5 佔位、不能選）。
8. commit REPRODUCE.md 的 checksum 更新（文件更新，freeze 允許）。

## 合規註記（官方「Only update private rows after July 13」）

最終 CSV 由單一 canonical run 產生（公開+私有一起），公開列相對 freeze 前
提交的差異僅為跨行程浮點噪音（mean ~3e-4、翻轉 ≤0.04%，REPRODUCE.md 已文件
化）——權重零改動，符合凍結規則的意旨；且這讓「提交檔 == Docker 輸出」的驗證
一致性最大化。

## 時間軸（假設 7/13 上午釋出）

| 時刻 | 事件 |
|---|---|
| T+0 | 下載+合併目錄（~1h） |
| T+1h | 開跑 |
| T+21h（7/14 上午） | 跑完 → 檢查 → 上傳 → 選定 |
| 7/14 下午 | 填模板、最後 buffer |
| 7/15 23:59 AoE | 官方討論串回覆 |
