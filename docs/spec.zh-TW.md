# Attar 規格書

Nov 28, 2026 · @JR · 對應版本 1.0.9

> 細節表格不在這裡重複：引擎與評估器見 [features.md](features.md)，事實與能力見 [concepts.md](concepts.md)，zoo 見
> [benchmarks.md](benchmarks.md)，比對結果見 [validation/](validation/)。

## 0. 現況摘要

- **套件**：PyPI `attar`，最新版 1.0.9；MIT；Python 3.8+、PyTorch 2.0+，CPU 與 CUDA。
- **資料**：Hugging Face `RayChang987/attar-zoo`，262 個設計、24 個 suite，第一次使用時下載。
- **引擎**：die sizing、packing、macro／IO placement、global placement、legalization（含 PDN 與 pin access）、detailed
  placement、global routing（含增量）、CTS、gate sizing、buffering、debuffering。
- **評估器**：HPWL、legality、setup／hold timing（placement 與 GR 寄生）、pin／net slack、DRV、power、propagated clock
  與 CRPR、GR overflow、pin access 等，全部在同一份 netlist 與 placement 上和 OpenROAD／OpenSTA／Innovus 比過。
- **研究重現**：LATTE、P11、DATE25 在 Attar 上重現，各有逐步講解的 Colab notebook。
- **測試**：196 項，ruff 無錯誤；完整測試在 tcdl5（`~/attar_DAC2026/attar_109`）上跑。

## 1. 定位

Attar 是物理設計演算法的快迴圈：把各種 benchmark 與 PDK 蒸餾成同一份張量資料，在 GPU 上秒級跑演算法、量結果；結果以
DEF 交回使用者的 flow（OpenROAD、Innovus、ICC2）做正式驗證。名稱取自 attar（玫瑰精油）：數千片花瓣蒸餾成一滴。

RosettaStone 2.0 往真實 PDK 與 signoff 走，可信但每次要跑完整 flow。Attar 反過來只保留算指標必需的數據（尺寸、routing
容量、Liberty 表、線的 RC、PDN 形狀），但在保留的範圍內**評估要和參考工具一致**，而不只是「有相關性」。

|  | RosettaStone 2.0（慢迴圈） | Attar（快迴圈） |
| --- | --- | --- |
| 評估 | 完整 RTL-to-GDS flow | 各環節評估器，與 OpenROAD／OpenSTA 一致 |
| 單次時間 | 分鐘到小時 | 毫秒到秒（百萬 cell 為秒到分鐘） |
| 安裝 | 編譯 OpenROAD、ORFS | `pip install attar` |
| 角色 | 裁判 | 練習場、快速迭代 |

Attar 的定位是「一致的評估加上可替換的基本引擎」：評估器照 OpenSTA／OpenROAD 原始碼移植到張量上，逐 endpoint 一致。

## 2. 目標使用者

主要是寫物理設計演算法論文的研究生與研究員：今天要花數週處理格式與 IO 才能公平比較（benchmark 與演算法各有格式、
正式 flow 門檻高且慢）。次要是 ML for EDA 研究者。三篇論文的重現證明了用法：演算法本身只有數百行，資料、評估、
合法化、routing 由 Attar 提供。

## 3. 設計原則與不做的事

1. **蒸餾，不是刪除**：原始檔以檔名與 sha256 記在 manifest；IR 自給自足，快迴圈只讀 IR，不讀環境變數。
2. **照參考工具原始碼移植，不擬合常數**；仍不一致處記在結果的 `limitations`。
3. **資料全程在張量**，CPU 與 GPU 同一套程式。
4. **百萬 cell 在幾分鐘內**，CPU 也一樣；必要時犧牲一點精度。
5. **安裝簡單**：不需編譯 CUDA extension；少數 C++ 小工具第一次使用時以 g++ 編譯，GPU timing 用 Triton。
6. **結果以 DEF 交回**：寫進來源 DEF 的框架；ORFS 交接工具只用於 Attar 自己的驗證。
7. **演算法是使用者的**：Attar 提供資料、評估器、基本引擎與掛鉤（GP 的 `extra`／`callback`、timer 唯讀資料）。
8. **先跑通、合法，再談品質；不手調參數**，用對照實驗找根本原因。
9. **可重現**：LG、STA、評估、RUDY、容量、export 逐位元相同；GPU 上的 GP 需 `deterministic=True`。

| 不做 | 原因 | 交給 |
| --- | --- | --- |
| Signoff STA、detailed routing、DRC、PDN 合成、detailed route 寄生萃取 | 需完整幾何或規則，非全域可張量化問題 | 使用者的 flow |
| 從 IR 重建 LEF／Liberty、OpenDB 後端 | 蒸餾掉的資訊無法還原；需編譯 OpenROAD | 原始檔、ORFS |
| OpenTimer 相容模式 | STA 只跟 OpenSTA | — |
| 內建可微分 timer | 屬於演算法（原則 7） | 使用者的程式 |

暫緩：partitioning、自動補資料（`attar.derive`）、3D placement（IR 已預留 die 陣列）、遵守 fence region。

## 4. 範圍與接下來

兩種題目：`before_fp`（die、IO、macro 是答案；Attar 不畫電源網）與 `after_fp`（cell 位置是答案，之後 sizing、buffering、
routing、timing）。引擎與評估器的完整清單與精度見 [features.md](features.md)。再次評估只重算有變動的 net；GPU 上搬動
cell 後只重算受影響的 timing 範圍（結果與完整重算逐位元相同）：範圍小時在 CPU 上掃（C++ 的延遲計算與 GPU kernel 逐位元
相同），大時在 GPU 上掃；CPU 與 GPU 的 timer 結果也逐位元相同。netlist 編輯（插 buffer、clone、拆 buffer）後由 CPU
引擎跟著更新（自己不變的 pin 編號、可插入的排序鍵），GPU 的 level 與 stage 結構先丟掉，有人讀或範圍太大時才重建；下一次
timing 只重算編輯過的 net 的影響範圍（結果與新建的 timer 逐位元相同；動到 clock、constant、latch 時才重建）。jpeg_x20
插一個 buffer：編輯約 140 ms、timing 約 30 ms。

接下來：

- 編輯後的 required time（`pin_slack`）在 CPU 上增量算，不必重建 level（目前讀了就改回完整更新）；propagated clock
  （CRPR）下也走 CPU 引擎。
- 剩下的固定成本：`Edit.apply` 的 O(P) 陣列複製、`wires()` 約 13 ms、checks 與結果約 6 ms（jpeg_x20）。
- 增量 routing 的 negotiation：重繞的 net 很多時不比全部重繞快；改策略會影響品質，需另行決定。
- 匯出 Verilog、SDC 與精簡 Liberty，方便和外部 baseline（Rsyn、DREAMPlace 4.0）比較。
- zoo 的設計註記（是否有 reference placement、contest SDC 下是否違規）。
- fence region 與 placement blockage 存入 IR 並納入 `legality()`（目前 21 個 zoo 設計違反卻被判合法）。

## 5. 資料層

**前端海納百川，Design 是狀態，有什麼題目就做什麼答案。** 每個 reader 只交出它有的事實；事實決定能力；引擎的產出寫回
Design，再解鎖下一個能力。缺的事實不自動補，明確報 `MissingFacts`（推導套件 `attar.derive` 尚未做）。題目與解答狀態
分開，解答可存成具名 snapshot；改 netlist 的操作以批次交易（`d.edit()`）進行，cell 與 net 編號不動。

固定慣例：die 一律是陣列；IO pin 只收 signal pin；cell 左下角為原點；pin offset 相對 cell 中心；磁碟存整數 DBU；數值在
張量、名稱在 CPU 對照表。

**Tech Profile** 只保留：routing（gcell、每層方向與容量、pin／OBS 形狀、special net、via 電阻）、線的 RC、cell 的 NLDM 表與
pin 電容、power 表與 cell function、SDC。同一平台 corner 的 cell 存一份共用 library（`zoo/libraries/`）。

| 保真度 | 資料 | 可信的指標 | 範例 |
| --- | --- | --- | --- |
| F1 | LEF + Liberty（+ SDC） | placement、routing、timing、power | ORFS、NS3K、ICCAD 2015／2025 |
| F2 | 有 routing 容量，無 Liberty | placement、routing | ISPD 2018／2019／2024 |
| F3 | 只有尺寸與連接 | placement | ISPD 2005、GSRC、MCNC |

散布：zoo 在 Hugging Face（zstd 壓縮，下載時解壓），不進 git 與 pip；格式只做加法，舊版讀新 zoo 不受影響。實測：118 萬
cell 的 bp_quad 蒸餾 34 秒，之後載入 0.3 秒。

## 6. 系統架構

編譯器式：多個前端、一份 IR、多個後端；N 種格式與 M 種演算法只需 N + M 個轉接器。

```text
來源（LEF/DEF、Bookshelf、YAL、GSRC blocks、Liberty、SDC）
   └─ reader ─> IR（safetensors + manifest）─ attar.load ─> Design（張量）
         ├─ attar.naive（引擎）   ├─ Evaluator、d.timing()、d.global_route()（評估）
         └─ d.export ─> DEF ─> 使用者的 flow
```

## 7. 準確度驗證

**為什麼不用代理指標**：在 pa_pdn 的 routed run（NS3K、Innovus 21.13、16 個設計 × 2 種 PDN × 約 60 種 placement）
上，常用代理和真值的排名一致性如下：

| 代理 → 真值 | Kendall τ 平均 | 代理最佳者也是真值最佳 |
| --- | ---: | ---: |
| HPWL → routed WL | 0.78 | 65%（隨機約 4%） |
| HPWL → WNS | −0.03 | 3% |
| RUDY（最壅塞 1% bin）→ DRV | 0.04 | 32% |

HPWL 對 routed WL 可信但有解析度（差距 <0.1% 時同向率 55%，>5% 時 99.8%）；HPWL 不能代表 timing，RUDY 不能代表 DRV。
所以 timing 與 routing 移植參考工具的演算法，在同一份 netlist 與 placement 上逐項比對；參考工具是 ORFS 映像
（`openroad/orfs:26Q4-368-g9ca2a9e76`）裡的 OpenROAD 與 OpenSTA，pin access 則對 Innovus 的 DRC。結果見
[validation/](validation/)。

慢迴圈本身也有雜訊：隨機移動 1% cell 一個 site，ORFS 的 routed WL 變動 0.1–0.3%，repair_timing 之後的 WNS／TNS 變動
可達數十 %，所以比對在同一個 placement 上做，而不是在 flow 跑完後。

已知差異：

- **ICCAD 2015 與文獻**：論文回報 OpenTimer；Attar（等同 OpenSTA）的 TNS 約為其 3–4 倍，只能比方法之間的倍數。
- **Global routing**：線長比 OpenROAD 少 0.5–3.6%，但殘留少量 OpenROAD 沒有的 overflow。
- **Global placement 與 DREAMPlace**：ISPD 2005 合法化後 HPWL 高 2.2–6.6%；timing-driven 迴圈中與 DATE25-TDP 相當。
- **Pin access**：signal 對 signal 的 M2 DRV（約 58%）不在範圍內。

## 8. API 與使用體驗

目標：`pip install` 到第一個可比較的結果，五分鐘、五行程式。

```python
import attar
d = attar.load('after_fp/orfs_asap7/aes', device='cuda')    # 第一次從 zoo 下載
d.place()                                                    # 或把自己的結果寫進 d.pos
print(attar.Evaluator(d).hpwl(), d.timing().wns)             # 與 OpenSTA 一致的 WNS
d.export('aes.def')                                          # 交給自己的 flow
```

原則：演算法「張量進、位置出」，評估器接受明確的位置；研究掛鉤（GP 的 `extra`／`callback`，timer 唯讀資料含
`stages()`，帶版本號）；增量評估；錯誤訊息說清楚缺什麼、怎麼辦（`MissingFacts`、`GPUOutOfMemory`）。

交付物：PyPI 套件、HF zoo、五本 Colab notebook、`docs/`、README（英、繁、簡）。

## 9. 風險與開放問題

| 風險 | 對策 |
| --- | --- |
| 評估器在新設計或新 SDC 上與參考工具不一致 | 照原始碼移植；差異記在 `limitations`；發版時跑 `docs/validation/` 的工具 |
| 研究程式依賴 timer 私有欄位 | 唯讀資料 API 帶版本號，含 `stages()` |
| GPU 記憶體不足 | 錯誤訊息附解法；CPU 路徑也在幾分鐘內完成 |
| 與文獻數字無法直接比（ICCAD 2015） | 文件與 notebook 註明，只比倍數 |
| 重現研究時移植出錯 | 重現一律在 Attar 上跑；對照組也要驗證與官方實作相當 |
| 範圍蔓延 | 新功能先問是不是資料、評估器或基本引擎 |

開放問題：`attar.derive` 做到什麼程度；fence region 是否在 legalizer 中遵守；增量 routing 是否讓舊 net 也能移動；3D
placement 何時開始。
