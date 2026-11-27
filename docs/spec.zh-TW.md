# Attar 規格書

Nov 27, 2026 · @JR · 對應版本 1.0.7（加上未發布的 pin access）

> 本文最初是 2026-11-09 的 v0.1 MVP 規格，2026-11-22 依實際做出來的樣子改寫。原規劃中後來改變或放棄的部分，
> 在各節以「原規劃」註明，保留做決定的理由。

## 0. 現況摘要

- **套件**：PyPI `attar`，1.0.0（11-15）到 1.0.7（11-24）；MIT 授權；Python 3.8 以上、PyTorch 2.0 以上，CPU 與 CUDA 皆可。
- **資料**：Hugging Face `RayChang987/attar-zoo`，262 個設計、24 個 suite，第一次使用時自動下載。
- **引擎**（`attar.naive` 與 Design 上的編輯）：die sizing、packing、macro placement、IO placement、global placement、
  legalization（含 PDN 與 pin access）、detailed placement、global routing（含增量）、CTS、gate sizing、buffering 與 cloning、
  debuffering。
- **評估器**：HPWL、legality、setup／hold timing（placement 與 global-routing 寄生）、pin／net slack、DRV、power、
  propagated clock 與 CRPR、GR overflow、pin access（PDN 擋住的 pin）、RUDY 等。每一項都在同一份 netlist 與 placement 上和 OpenROAD／OpenSTA 比過
  （第 7 節）。
- **研究重現**：DAC 2026 的 LATTE、P11（混合 net–pin 加權）與 DATE 2025 的 pin 對加權都在 Attar 上重現，並有逐步
  講解的 Colab notebook。
- **測試**：194 項（2 項略過），ruff 無錯誤。

## 1. 精神與定位

Attar 是物理設計演算法的快迴圈：把各種 benchmark 與 PDK 蒸餾成同一份張量資料，在 GPU 上以秒級速度跑演算法、量結果；
結果以 DEF 交回使用者自己的 flow（OpenROAD、Innovus、ICC2）做正式驗證。

名稱取自 attar（玫瑰精油）：數千片花瓣，蒸餾成一滴精華。PDK 與完整 flow 是花瓣；Attar 只留下演算法真正需要的數據。

RosettaStone 2.0 往現實與 signoff 走：把學術與比賽 benchmark 翻譯成真實 PDK、合規的 LEF/DEF，跑完整 RTL-to-GDS
flow。它可信，但每次評估要跑完整流程，門檻高也慢。Attar 往反方向走：只保留算指標必需的數據（cell 與 macro 尺寸、
routing 容量、Liberty 表與線的 RC、PDN 形狀），但在保留的範圍內，**評估要和參考工具一致**，而不是只求「有相關性」。

原規劃：v0.1 定位為「代理指標」，只宣稱和慢迴圈有足夠相關性。實作後發現，在同一份 netlist 與 placement 上，可以把
OpenSTA、OpenROAD 的演算法照原始碼移植到張量上，得到逐 endpoint 一致的結果（第 7 節）；於是定位從「相關的代理」
改為「一致的評估加上可替換的基本引擎」。

|  | RosettaStone 2.0（慢迴圈） | Attar（快迴圈） |
| --- | --- | --- |
| 方向 | 往真實 PDK、signoff | 蒸餾 PDK、只留數據 |
| 評估 | 完整 RTL-to-GDS flow | 各環節的評估器，與 OpenROAD／OpenSTA 一致 |
| 單次時間 | 分鐘到小時 | 毫秒到秒（百萬 cell 為秒到分鐘） |
| 安裝 | 編譯 OpenROAD、ORFS | `pip install attar` |
| 角色 | 裁判、正式結果 | 練習場、快速迭代與除錯 |

一句話：**GPU 上秒級迭代，自己的 flow 上端到端驗證。** Attar 與慢迴圈是互補，不是替代。

## 2. 痛點與目標使用者

主要使用者是寫物理設計演算法論文的研究生與研究員；他們今天要花數週處理格式與 IO，才能做一次公平比較。

痛點有三個：

- **Benchmark 四散**：ISPD、ICCAD 各屆比賽、ORFS 設計、OpenCores 各有格式，同名 testcase 在不同論文用不同 library 與規則。
- **演算法四散**：Xplace、DREAMPlace、TritonRoute、各家 GPU GR、Abacus 各有自己的 IO 與評估腳本，無法互接。
- **門檻高、迭代慢**：正式 flow 要編譯 OpenROAD、學 Tcl 與 OpenDB；GPU 工具多半需編譯 C++/CUDA extension；一次
  完整評估從分鐘到小時。

次要使用者是 ML for EDA 研究者：需要能快速產生大量實驗、且資料已是張量的環境。

**實際驗證過的用法**：三篇論文在 Attar 上重現（LATTE、P11、DATE25），每篇的演算法本身只有數百行，資料、評估、
合法化與 routing 都由 Attar 提供；重現的過程也找出了 Attar 的缺口與 bug（`RECOMMENDATIONS_FROM_P11.md`、
`recommendations_from_implementing_LATTE.md`，多數已在 1.0.5、1.0.6 修正）。

## 3. 設計原則與明確不做的事

Attar 只做「全域、可張量化」的問題；需要精確幾何或 signoff 精度的，交給使用者的 flow。

設計原則：

1. **蒸餾，不是刪除**：只保留算指標必需的參數；原始 LEF、Liberty、DEF 以檔名與 sha256 記在 manifest（不記路徑），
   供重現使用。IR 自給自足：一個設計資料夾複製到別台機器就能用；快迴圈只讀 IR，不讀任何環境變數。
2. **評估照參考工具的原始碼移植，不擬合常數**：和 OpenSTA 不一致時，去讀 OpenSTA 對應的程式（例如 `dcalc/DmpCeff.cc`）
   照做，不加經驗修正項。已知仍不一致的地方，記在每次結果的 `limitations` 裡（`summary()['exact']` 為 False）。
3. **資料全程在張量**：只有讀檔與寫檔兩端做轉換；內層迴圈避免 host 同步。CPU 與 GPU 跑同一套程式。
4. **百萬 cell 在幾分鐘內**：一百萬 cell 以內的設計，載入、placement、timing、routing 每一步都要在幾分鐘內完成，CPU 也
   一樣；忠實移植在 CPU 上太慢時，寧可犧牲一點精度換速度。
5. **安裝簡單**：`pip install` 就能用，不需要編譯 CUDA extension。少數 C++ 小工具（OpenROAD 的 Steiner tree、DMP 驅動
   模型、power 的活動率、Abacus）在第一次使用時以 g++ 編譯並快取；GPU 的 timing kernel 用 Triton，第一次執行時編譯。
   沒有 g++ 時，各自退回較慢或較粗的做法，並在結果中註明。
6. **結果以 DEF 交回使用者的 flow**：`d.export(out_def)` 把結果寫進來源 DEF 的框架（`frame.def.gz`：PDN、IO pin 形狀、
   固定 cell 等原樣保留），可移動 cell 與 signal net 由 IR 寫出。只有 LEF/DEF 來源的設計能 export；Bookshelf、YAL、
   GSRC 沒有 cell library，結果停在 Attar。ORFS 的交接工具只用於 Attar 自己的驗證（`attar._slow_loop`、`tools/`），
   不在使用者介面上。
7. **演算法是使用者的**：Attar 提供資料、一致的評估器、可替換的基本引擎，以及讓研究程式接上的掛鉤（GP 的
   `extra`／`callback`、timer 的唯讀資料）。研究方法本身（例如可微分 timing）留在使用者的程式裡，不做進套件。
8. **先讓 flow 跑通、合法，再談品質；不手調參數**：flow 失敗時用對照實驗找根本原因，修方法，不加 heuristic 或上限。
9. **可重現**：LG、STA、評估、RUDY、容量、export 在同樣輸入下逐位元相同；GPU 上的 global placement 預設不保證
   （浮點 atomic 加法順序不定），需要時用 `global_place(..., deterministic=True)`；CPU 上執行緒數相同即可重現。

明確不做：

| 不做 | 原因 | 交給 |
| --- | --- | --- |
| Signoff STA（多 corner、完整時序例外、精確寄生） | Attar 是單一 corner、placement／GR 寄生的 OpenSTA 等價分析 | OpenSTA、Innovus、PrimeTime |
| Detailed routing、DRC | 規則密集、局部、序列化 | TritonRoute、商業工具 |
| PDN 合成 | 結構性設計決策 | 使用者的 flow |
| 從 detailed route 萃取寄生 | 需完整幾何 | OpenRCX、StarRC |
| 從 IR 重建完整 LEF、Liberty | 蒸餾掉的資訊無法還原 | 原始檔 |
| OpenDB（odb）讀寫後端 | 需編譯 OpenROAD | ORFS |
| OpenTimer 相容模式 | STA 只跟 OpenSTA（第 7.3 節說明兩者在 ICCAD 2015 上的差異） | — |
| 內建可微分 timer | 屬於演算法（原則 7） | 使用者的程式 |

暫緩（未來評估）：

- **Partitioning**：資料形態（純超圖）與其他能力格格不入。
- **自動補資料（`attar.derive`）**：缺的事實預設不補；推導套件之後再做，見 5.1。
- **3D placement**：IR 已預留 die 陣列，引擎未做。
- **Fence region**：讀得到、只計數，不保證遵守。

## 4. 範圍與現況

### 4.1 兩種題目

zoo 依使用者要解的題目分兩種模式（設計名稱為 `<模式>/<suite>/<設計>`）：

- **`before_fp`，floorplan 題目**：netlist 與 cell library 已知，die（利用率）、IO pin、macro 是答案。Attar 不畫電源網；
  有 cell library 的結果以 DEF 交給使用者的 flow 畫 PDN。
- **`after_fp`，placement 題目**：die、IO pin、macro、電源網已知，cell 的位置是答案；之後還有 sizing、buffering、routing
  與 timing。

### 4.2 引擎

| 環節 | 模式 | 呼叫 | 做法（摘要） |
| --- | --- | --- | --- |
| Die sizing | `before_fp` | `naive.make_die` | 依利用率與長寬比定 die，沿用原有 row 的 site 格點重建 row |
| Packing | `before_fp` | `naive.place_macros(packing=True)` | block 層級設計：B*-tree packing，退火塞進 core，再依 HPWL 調整 |
| Macro placement | `before_fp` | `naive.place_macros` | mixed-size GP 後，macro 在其 routing track 上合法化並加 halo |
| IO placement | `before_fp` | `naive.place_io` | IO pin 放在 die 邊界；cell 擺好後依連接重排 |
| Global placement | 兩者 | `naive.global_place` | ePlace／DREAMPlace 式靜電 placement：WA wirelength、DCT 解 Poisson、Nesterov 加 backtracking、CUDA graph；`routability=True` 為 DREAMPlace 的 cell inflation；研究掛鉤 `extra`、`callback`，callback 改 net 權重從下一次迭代生效，preconditioner 依 net 權重（DREAMPlace） |
| Legalization | `after_fp` | `naive.legalize` | multi-row height、power rail parity；預設 Abacus（C++），cell 若被 GP 留在 macro 上會移到最近的空位；之後讓 signal pin 避開 PDN 的 via 金屬（row 動態規劃、鏡射、推移）；`cells=` 只重排指定 cell |
| Detailed placement | `after_fp` | `naive.detailed_place` | OpenROAD dpo 的 global swap、vertical swap、local reordering、cell flipping，在 GPU 上批次進行，保持合法 |
| Global routing | `after_fp` | `d.global_route()` | OpenROAD 的 gcell 格與容量；Prim-Dijkstra 樹、negotiated Z／L pattern routing、分段 layer assignment；`previous=routes` 只重繞 pin 換了 gcell 或層的 net（增量，類似 `grt -incremental`） |
| CTS | `after_fp` | `naive.cts` | 每個 ideal clock net 建 buffered clock tree（OpenROAD 的 clock buffer、TritonCTS 的 clustering），在 propagated-clock timer 上逐層平衡 |
| Gate sizing | `after_fp` | `d.resize`、`d.equivalents` | 同等價類的 cell 互換（驅動強度、Vt），timer 原地更新 |
| Buffering、cloning | `after_fp` | `d.edit()` | 批次 netlist 編輯，cell 與 net 的編號不變，下一次 `d.timing()` 即看到 |
| Debuffering | `after_fp` | `d.debuffer()` | 去掉 buffer 與雙反相，如 OpenROAD `remove_buffers` |

Design 上的捷徑：`d.floorplan(utilization=, aspect_ratio=)`（die、IO、macro 一次完成）、`d.place()`（GP 加 LG，用來源
flow 自己的目標密度）、`d.plot()`、`d.export()`。

原規劃：v0.1 只做 wirelength 這條線，DRV 與 timing 代理當作研究題目。實際上 timing 以移植 OpenSTA 的方式做成一致的
評估器（第 7 節），routing、CTS、sizing、buffering 也都在 11 月內加入。

### 4.3 評估器

`attar.Evaluator(d)` 與 Design 上的方法。精度一律指「同一份 netlist、同一個 placement 上，與參考工具的差距」，完整表格在
`docs/validation/` 與 `docs/features.md`。

| 評估 | 呼叫 | 參考工具 | 精度（摘要） |
| --- | --- | --- | --- |
| HPWL | `ev.hpwl` | OpenROAD 資料庫的 HPWL，23 個 ORFS 設計（最大 105 萬 cell） | 0.00002% 以內 |
| Legality | `ev.legality` | OpenROAD `check_placement` | 23 個 ORFS 設計與 62 個 NS3K floorplan 判定一致；故意弄壞的 placement 雙方都拒絕 |
| Timing（placement 寄生） | `d.timing()` | OpenSTA，`estimate_parasitics -placement`，31 個設計 | WNS 誤差中位數 0.00%（ORFS）、0.11%（NS3K），最大 0.34%；hold slack MAE 最多 0.36 ps；比 OpenSTA 快 5–30 倍 |
| Timing（ICCAD 2015，Late／Early library） | `d.timing()` | OpenSTA `-max Late -min Early`，8 個 superblue | WNS 誤差最大 0.33%、TNS 最大 0.13%、hold MAE 0.11–0.21 ps |
| Pin／net slack | `pin_slack()`、`net_slack()` | OpenSTA 每個 pin 的 `slack_max` | Pearson 至少 0.9995；負 slack 的 pin 上 MAE 最多 1.03 ps |
| Sizing、netlist 編輯後的 timing | `d.resize`、`d.edit()` 之後 | OpenROAD `replace_cell`、匯出的 DEF + OpenSTA | WNS 誤差最多 0.23%（sizing）、1.18%（NS3K 上插 buffer 與 clone） |
| DRV | `d.drv()` | `report_check_types` | 21 個設計違規 pin 完全相同；電容 slack 誤差 1.1 fF 以內 |
| Power | `d.power()` | `report_power` | 20 個設計總和 0.01% 以內、每個 instance 相同 |
| Propagated clock、CRPR | `d.propagate_clocks()` | OpenSTA + TritonCTS 的 clock tree | clock arrival 0.24 ps 以內，setup slack MAE 最多 0.03 ps |
| Global-routing 寄生 | `d.timing(routes=...)` | `estimate_parasitics -global_routing`（OpenROAD 自己的 routes） | 每條 net 的 R、C 與 SPEF 相同；WNS 誤差最多 0.08% |
| Global routing | `d.global_route()` | OpenROAD `global_route`，14 個設計 | 線長比 OpenROAD 少 0.5–3.6%，每條 net 的線長 Pearson 至少 0.998；排序 placement 和 OpenROAD routed WL 一致（差距 1% 以上的配對 100%；HPWL 只有 74–88%）；快 16–41 倍；殘留少量 OpenROAD 沒有的 overflow |
| GR overflow | `d.gr_overflow()` | OpenROAD 的 gcell 容量與 global route | 無 macro 時容量完全相同、有 macro 時 0.16% 以內；設計排序 Spearman 0.97 |
| Pin access（PDN 擋住的 pin） | `ev.pin_access` | Innovus routeDesign + verify_drc，i3dPDK_NS3K 988 個 run（pa_pdn） | 標記的 pin 與 signal 對 PG 的 M1/M2 DRV 一一對應（156,665／156,665，無誤報）；OpenROAD dpl 的 PDN 規則只標出 66 個 cell；signal 對 signal 的 DRV 不在範圍內 |
| Steiner WL、RUDY、density overflow | `ev.*` | 尚未比對 | — |

再次評估只重算有變動的部分（1.0.6）：移動 cell 或 resize 後，`d.timing()` 只重算 pin 有變的 net 的寄生；換一組 routes 後
只重算 route 有變的 net，結果與重新建 timer 逐位元相同。

### 4.4 接下來

- **下一版**：公開、帶版本號的 `TimingResult.stages()`（timer 的逐層排程與它自己算的 gate／wire delay），讓 LATTE 這類
  可微分 timer 不必讀 timer 的私有欄位；之後把 LATTE notebook 換上去，驗收標準是結果逐輪不變。
- 增量 routing 的 negotiation：重繞的 net 很多時（netcard_fast 移動 1% cell，1 萬條 net）不比全部重繞快；要改策略會影響
  繞線品質，需另行決定。
- 匯出 Verilog、SDC 與精簡的 Liberty，方便和外部 baseline（Rsyn、DREAMPlace 4.0）比較。
- zoo 的設計註記（是否有 reference placement、在 contest SDC 下是否有違規等）。
- 尚未做的引擎：delay-balanced clock tree。

## 5. 資料層：事實、能力與來源

**設計哲學：前端海納百川，Design 是狀態，有什麼題目就做什麼答案。**

- **前端海納百川**：任何 benchmark 格式都應該吃得進來，轉成同一套張量。每個 reader 只交出它手上有的「事實」，不要求齊全。
- **Design 是狀態**：一個 Design 擁有哪些事實，就決定它能做哪些事。框架負責這套機制，每個能力頂多附一個基本演算法。
- **有什麼題目，做什麼答案**：演算法的產出寫回 Design 的解答狀態，於是又解鎖下一個能力（例如 GP 產生 placement
  之後，才能做 legalization）。

### 5.1 事實、能力與來源

**事實（facts）**，見 `attar/facts.py`：

| 事實 | 內容 | 類別 |
| --- | --- | --- |
| `netlist` | cell、net、pin 的連接，net 權重 | 題目 |
| `cell_geometry` | cell 寬高、旗標（movable、fixed、macro） | 題目 |
| `pin_offsets` | pin 相對 cell 的位置（master 原始 offset 加方向） | 題目 |
| `pin_direction` | 每個 pin 是 driver 還是 sink（LEF `DIRECTION`、Bookshelf `I`/`O`）；timing 的前提 | 題目 |
| `floorplan.die` | die 陣列（每個 die 的外框） | 題目 |
| `floorplan.io` | IO pin 位置（die 邊界上） | 題目（可由 IO placer 產生） |
| `floorplan.macros` | macro 與固定 cell（如 tapcell）的位置 | 題目（可由 macro placer 產生） |
| `rows` | row 位置、site 寬、row 高、所屬 die | 題目 |
| `constraints` | fence region、placement blockage | 題目（只計數，未儲存） |
| `routing` | 擋 routing 容量的東西：pin 與 OBS 形狀、routing 層上的 special net、routing 層範圍與調整係數 | 題目 |
| `timing` | Liberty arc 與 pin 電容、clock、I/O delay（SDC） | 題目 |
| `library` | 共用的 cell library（`zoo/libraries/<名稱>`）：平台 corner 的每個 cell 及其 Liberty 資料，sizing 與 buffering 用 | 題目 |
| `cell_library` | cell 是真實 cell library（LEF master）的實例：結果可以寫回 DEF | 題目 |
| `tracks` | 各層 routing track，與每個 master pin 的 routing 層（macro 放在 track 上） | 題目 |
| `pdn` | 最低幾層 routing 層上的 PDN 金屬：signal pin 要避開 | 題目 |
| `def_frame` | 來源 DEF 去掉 Attar 自己會寫的部分，原樣保留：export 的模板 | 題目 |
| `die_stack` | 3D：die stack | 題目（預留） |
| `placement` | 所有可移動 cell 的位置與方向 | 解答狀態 |

**能力（capabilities）**：每個引擎與指標宣告它需要哪些事實，Design 由此推出 `capabilities()`。`d.missing('legalize')`
列出缺哪幾項；`d.status()` 印出全部事實、來源與下一步提示；沒有事實就呼叫引擎時，丟出帶提示的 `MissingFacts`。

| 能力 | 需要 |
| --- | --- |
| `hpwl` | netlist、cell_geometry、pin_offsets、placement |
| `die_size` | cell_geometry |
| `io_place` | netlist、floorplan.die |
| `macro_place` | 再加 cell_geometry、pin_offsets、floorplan.io |
| `global_place` | 再加 floorplan.macros |
| `legalize`、`detailed_place` | netlist、cell_geometry、pin_offsets、floorplan.die、rows、placement |
| `global_route` | netlist、cell_geometry、pin_offsets、floorplan.die、routing、placement |
| `sta` | netlist、pin_direction、timing（有 placement 才估線的 RC） |
| `resize`、`netlist_edit` | netlist、cell_library、pin_offsets、timing、library |
| `gr_overflow` | netlist、pin_offsets、placement、floorplan.die、tracks、routing |
| `pin_access` | cell_geometry、pin_offsets、placement、tracks、pdn |
| `export` | netlist、cell_library、def_frame |

**題目與解答狀態分開**：題目基本不變；解答狀態（位置、方向）由演算法改寫，可以存成具名 snapshot（例如蒸餾時保留的
來源 placement 是 `reference`）。會改 netlist 的操作（sizing、buffering）以批次交易進行（`d.edit()`），cell 與 net 的編號
不會移動；timer 能原地更新的就原地更新，否則重建。

**來源（provenance）與不自動補**：manifest 記錄每項事實的來源（`def`、`lef`、`bookshelf`，或產生它的引擎，例如
`naive:place_io(nets)`）。缺的事實預設不補、明確報錯，不用假資料。

- **TODO：推導套件 `attar.derive`**。很多缺的事實可以推出來（沒有 row 就用 die 加 cell 高度生成、Bookshelf 掛一份
  蒸餾過的 tech profile）。只能明確呼叫，推導出的事實標記為 `derived:<方法>`。尚未實作。

**前端（readers）**：

| 前端 | 提供的事實 |
| --- | --- |
| LEF/DEF（`attar.io.distill`） | netlist、幾何、pin offset、die、IO 與 macro（有擺時）、row、cell library、track、pin direction、PDN、DEF 框架；Liberty 與 SDC 經 `tools/distill_timing.py` 附上 timing |
| Bookshelf（`attar.io.bookshelf`） | netlist、幾何、pin offset、pin direction、die（row 加固定物件的外框）、IO、macro、row；無 cell library |
| YAL（MCNC，`attar.io.fpformats`） | netlist、block 幾何、pin offset、die（parent outline）、pad |
| GSRC／UCSC blocks（`attar.io.fpformats`） | netlist；硬 block 有幾何；軟 block 只有面積與長寬比範圍 |

一開始就定死、現在仍成立的慣例：die 一律是陣列（2D 為 1 個 die）；IO pin 只收 signal pin；cell 左下角為原點；
pin offset 相對 cell 中心；磁碟存整數 DBU、載入時轉浮點；方向存 master 原始 offset 加 orientation；數值在張量、名稱在
CPU 對照表。

### 5.2 Tech Profile：蒸餾 PDK

Tech Profile 是一小組張量，只保留算指標必需的參數：

| 用途 | 保留的參數 |
| --- | --- |
| Routing | gcell 尺寸、每層方向與容量、pin 與 OBS 形狀、special net、via 電阻 |
| 線的 timing | 每層單位長度 R、C（`set_layer_rc`），via R |
| Cell 的 timing | NLDM 延遲與 slew 表、pin 電容、slew 門檻、clock tree 路徑、timing type 與 sense |
| Power | internal power 表、leakage、cell function（活動率傳遞） |
| 約束 | SDC：clock、I/O delay、clock latency、uncertainty、path delay 例外；尚未套用的指令記在 `unsupported` |

共用 library（`zoo/libraries/`）：同一平台 corner 的 cell 只存一份，設計只記引用與 sha256；ICCAD 2015 的 8 個
superblue 另有 Early library（`library_min`）供 hold 分析。

### 5.3 保真度分級

| 等級 | 資料 | 可信的指標 | 範例 |
| --- | --- | --- | --- |
| F1 真實製程 | LEF + Liberty（+ SDC） | placement、routing、timing、power，可與參考工具逐項比對 | ORFS ASAP7／Nangate45、NS3K、ICCAD 2015、ICCAD 2025 |
| F2 有 routing 資訊 | 有 routing 容量，無 Liberty | placement、routing | ISPD 2018／2019／2024 |
| F3 純幾何 | 只有尺寸與連接 | 只有 placement 指標 | ISPD 2005、GSRC、MCNC |

### 5.4 磁碟格式

每個設計一個資料夾：`core.safetensors`（整數陣列）、`names.json.gz`、`tech_profile.safetensors`、`manifest.json`（事實、來源、
數量、來源檔的 sha256），有 timing 時另有 `timing.json` 與 library 引用。zoo 位置：原始碼 checkout 中的 `zoo/`，否則
`~/.cache/attar/zoo`；每個讀 zoo 的函式都接受 `root=`。沒有任何環境變數。

實測：118 萬 cell、220 個 macro 的 bp_quad，蒸餾一次 34 秒，之後從快取載入 0.3 秒；890 萬 pin 的 HPWL 3 ms。

### 5.5 Benchmark Zoo（262 個設計，24 個 suite）

**`after_fp`（placement 題目）**

| suite | 設計數 | cell 數 | 起點 | Attar 結果 |
| --- | ---: | --- | --- | --- |
| ispd2005 | 8 | 21 萬–220 萬 | 未擺放 | 合法 |
| ispd2015 | 20 | 2.9 萬–130 萬 | 未擺放 | 合法；fence 不遵守 |
| ispd2018、ispd2019 | 20 | 0.8 萬–90 萬 | 已擺放 | 重新擺放合法 |
| ispd2024 | 5 | 12 萬–72 萬 | 已擺放 | 重新擺放合法 |
| iccad2015 | 8 | 77 萬–193 萬 | 已擺放（Encounter） | 重新擺放合法；附 timing（Late 與 Early library） |
| iccad2025 | 13 | 0.2 萬–30 萬 | 已擺放 | 重新擺放合法；附 timing，與比賽官方評估一致 |
| gsrc_place | 4 | 0.08 萬–2.8 萬 | 未擺放 | 合法 |
| ns3k、ns3k_{pl,st}_u{60,70,80} | 14 + 84 | 0.03 萬–118 萬 | 未擺放（Innovus floorplan） | 全部合法；OpenROAD `check_placement` 與 global routing 驗證 |
| orfs_asap7 | 15 | 0.04 萬–105 萬 | 未擺放（ORFS 3_3） | 合法，`check_placement` 通過；6 個交給 ORFS routing 0 DRC |
| orfs_nangate45 | 8 | 0.07 萬–11 萬 | 未擺放 | 合法，`check_placement` 通過 |

**`before_fp`（floorplan 題目）**

| suite | 設計數 | 規模 | 起點 | Attar 結果 |
| --- | ---: | --- | --- | --- |
| orfs_asap7 | 5 | 0.9 萬–5.2 萬 cell | ORFS 2_1 | die、IO、macro；交給 ORFS 畫 PDN 後 `check_placement` 通過 |
| orfs_nangate45 | 8 | 3.4 萬–21 萬 cell，10–136 macro | ORFS 2_1 | 16 種利用率設定中 15 個通過 ORFS 的 PDN 與 detailed placement |
| mcnc、gsrc_hard | 5 + 6 | 9–300 block | 未擺放 | floorplan，無重疊 |
| ispd2006_hb | 28 | 81–8,800 block | 硬加軟 block | 少量重疊殘留 |
| gsrc_soft、mcnc_soft | 11 | 9–300 block | 軟 block | 尚無引擎（block 形狀是答案） |

蒸餾 zoo 的腳本讀本地的原始 benchmark、ORFS run 與 NS3K kit，不在 repository 裡；每個 manifest 記錄它的來源。使用者自己的
設計走同一套 reader：`attar.distill(def, lefs, out_dir)`。

### 5.6 散布：Hugging Face Datasets

蒸餾後的資料放在 Hugging Face `RayChang987/attar-zoo`（公開），不進 git、不進 pip 套件。陣列與大型 JSON 以 zstd 壓縮
（level 19：4.8 GB 變成約 1.1 GB），下載時解壓，所以本地 zoo 永遠是原始檔。`attar.load(name)` 先查本地，沒有就下載；
`attar.fetch(suite, force=True)` 重新下載整個 suite（zoo 資料更新時用，CHANGELOG 的升級說明會提示）。

發布流程：`python tools/zoo_release.py`（檢查 manifest、打包 `zoo_hub/`、寫 `index.json` 與 dataset card），再
`hf upload-large-folder`（`docs/releasing.md`）。zoo 的格式只做加法：舊版 Attar 讀新的 zoo 不受影響。

## 6. 系統架構

架構採編譯器式設計：多個前端、一份 IR、多個後端；N 種格式與 M 種演算法只需 N + M 個轉接器，而不是 N × M 個。

```text
來源（LEF/DEF、Bookshelf、YAL、GSRC blocks、Liberty、SDC）
   └─ attar.io.distill 等 reader ─> IR（zoo 資料夾，safetensors + manifest）
         └─ attar.load ─> Design（GPU／CPU 張量）
               ├─ attar.naive（引擎：改動 Design，加入事實）
               ├─ Evaluator、d.timing()、d.global_route() ...（評估）
               └─ d.export ─> DEF ─> 使用者的 flow（OpenROAD、Innovus、ICC2）
```

模組：`attar.io`（reader、writer）、`attar.timing`（STA、DMP、寄生、sizing、power、CTS 用的 timer）、`attar.route`
（global router 與 routes）、`attar.rules`（routing 容量、rail、PDN 規則）、`attar.naive`（引擎）、`attar.gpu`（GPU 錯誤
說明）。

## 7. 準確度驗證

### 7.1 為什麼這樣驗證：pa_pdn 的 951 次 routed run

v0.1 規劃時，用 pa_pdn 專案留下的 routed run 量了幾種常用代理指標和真值的排名一致性（i3dPDK_NS3K、Innovus 21.13、
16 個設計 × 2 種 PDN × 約 60 種 placement 方法）：

| 代理 → 真值 | Kendall τ 平均 | τ 中位數 | 代理最佳者在真值也最佳 |
| --- | ---: | ---: | ---: |
| HPWL → routed WL | 0.78 | 0.85 | 65%（隨機約 4%） |
| HPWL → WNS | −0.03 | −0.05 | 3% |
| HPWL → TNS | 0.02 | −0.01 | — |
| RUDY（最壅塞 1% bin）→ DRV | 0.04 | 0.02 | 32% |

依 HPWL 差距分組，routed WL 同向的比例：

| HPWL 差距 | <0.1% | 0.1–0.25% | 0.25–0.5% | 0.5–1% | 1–2% | 2–5% | >5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 同向率 | 55% | 69% | 74% | 84% | 86% | 97.5% | 99.8% |

結論是：HPWL 對 routed WL 可信但有解析度；HPWL 不能代表 timing、RUDY 不能代表 DRV。這決定了後來的方向：timing 和
routing 不靠代理，而是把參考工具的演算法移植過來，在同一份 netlist 與 placement 上逐項比對（7.2）。

### 7.2 與參考工具逐項比對

參考工具是 ORFS docker 映像（`openroad/orfs:26Q4-368-g9ca2a9e76`）裡的 OpenROAD 與 OpenSTA；平台為 ASAP7 與
Nangate45，另以 OpenROAD 直接讀 NS3K 的 DEF。每項比對都在**同一份 netlist、同一個 placement**上進行，結果見 4.3 與
`docs/validation/`：

- `orfs_check.md`：HPWL 與 legality（23 個 ORFS 設計）；
- `timing_routing.md`：timing（ASAP7、Nangate45、NS3K、ICCAD 2015、ICCAD 2025）、sizing 與 netlist 編輯後的 timing、
  GR 容量與 overflow；
- `global_routing.md`：GR 寄生與 router；
- `drv_power_cts.md`：DRV、power、propagated clock、CRPR、CTS；
- `fp_handoff.md`：`before_fp` 的 floorplan 交給 ORFS 後的結果；
- `slow_loop.md`：交接工具與慢迴圈雜訊。

**Router 的排序測試**：真值是 OpenROAD 在同一份 netlist 上對每個 placement 的 global route。7 個設計各 7 個 placement：
差距 1% 以上的配對，Attar 的 GR 線長排序和 OpenROAD 一致 100%（HPWL 只有 74–88%），差距 1% 以下也有 91%。

**慢迴圈本身的雜訊**：ORFS 是 deterministic 的，換 seed 量不到雜訊，所以隨機挑 1% 的 cell 各移 1 個 site 重跑：

| 設計 | cell 數 | routed WL | GRT WL | WNS | TNS |
| --- | ---: | ---: | ---: | ---: | ---: |
| aes | 1.2 萬 | 0.29% | 0.70% | 89% | 98% |
| ibex | 2.1 萬 | 0.25% | 0.16% | 34% | 104% |
| jpeg | 5.8 萬 | 0.26% | 0.27% | 9.8% | 7.6% |
| ethmac | 6.0 萬 | 0.09% | 0.18% | 7.2% | 4.5% |

routed WL 的單次雜訊約 0.1–0.3%；ORFS 跑完 repair_timing 後的 WNS、TNS 受混沌行為主導，單次 run 不能當真值。這也是
Attar 在同一個 placement 上比對、而不在 flow 跑完後比對的原因。

原規劃：v0.1 打算在 ORFS 上做「10 個設計 × 15 種 placement × 3 次微擾」的跑到底相關性實驗，並在報告中把低於解析度的
差距標為「無法判斷」。評估器改為逐項一致之後，這個實驗沒有照原樣執行，`compare` 與解析度判讀也沒有實作；routing
的部分由上面的排序測試取代。

### 7.3 已知差異

- **ICCAD 2015 與文獻**：DREAMPlace 4.0 及其後的論文回報的是 OpenTimer 的數字。在同一個 placement 上，Attar（等同
  OpenSTA）的 TNS 約是 OpenTimer 的 3–4 倍，差異集中在少數 endpoint：superblue 的 `lcb*` local clock buffer 是反相器，
  OpenSTA 下 register 在下降沿取樣，primary I/O 路徑只剩半個週期。所以 Attar 的絕對數字不能和這些論文比，只能比方法
  之間的倍數。
- **Global routing**：線長比 OpenROAD 少 0.5–3.6%，但會殘留少量 OpenROAD 沒有的 overflow。
- **Global placement 與 DREAMPlace**：只最佳化線長時，superblue18 的 TNS 約差 1.28 倍；ISPD 2005 合法化後的 HPWL 高
  2.2–6.6%。在 timing-driven 迴圈中（DATE 2025 的 pin 對加權），結果與 DATE25-TDP 官方程式相當（TNS 平均 0.93 倍）。
- 個別 timing 結果無法照 OpenSTA 建模的部分（例如尚未套用的 SDC 指令），記在 `TimingResult.limitations`。

## 8. API 與使用體驗

目標：從 `pip install` 到第一個可比較的結果，五分鐘、五行程式。

```python
import attar
d = attar.load('after_fp/orfs_asap7/aes', device='cuda')    # 第一次從 zoo 下載，之後讀本地
d.place()                                                    # 或把自己的演算法結果寫進 d.pos
print(attar.Evaluator(d).hpwl(), d.timing().wns)             # HPWL，以及和 OpenSTA 一致的 WNS
d.export('aes.def')                                          # 交給 OpenROAD、Innovus、ICC2 ...
```

三個核心物件：**Design** 是狀態（題目加解答）；**Evaluator** 與 Design 上的評估方法量測它；**`attar.naive`** 是每種能力的
基本引擎。指標和引擎都宣告需要哪些事實，所以能量什麼、能做什麼由 Design 有什麼決定。

API 原則：

- **自己的演算法一個函式就能接**：演算法是「張量進、位置出」，直接寫 `d.pos`；評估器也接受明確給的位置，可以在最佳化
  迴圈裡用而不改動 Design。
- **研究掛鉤**：`global_place(extra=, callback=)`（加自己的目標項、每次迭代後改 net 權重）；timer 的唯讀資料
  `TimingResult.graph()`、`library()`、`arrivals(per_tag=)`、`endpoint_required()`、`parasitics()`、`trees()`、
  `critical_paths()`，帶版本號 `TIMING_DATA_VERSION`；`d.set_timing()` 在記憶體中改 timing 設定、`d.ideal_clocks()`；
  `d.pin_names`、`d.pin_index`、`d.read_placement(def)` 等名稱與 placement 工具。
- **增量**：再次 `d.timing()` 只重算有變動的 net；`d.global_route(previous=routes)` 只重繞 pin 換了 gcell 的 net。
- **錯誤要說清楚**：缺事實時丟 `MissingFacts`，附缺哪些事實與怎麼取得；GPU 記憶體不足時丟 `attar.gpu.GPUOutOfMemory`，
  說明是哪一步、哪個設計、GPU 記憶體的使用狀況與解法。
- **可重現模式**：`global_place(..., deterministic=True)`（GPU 上每次相同，較慢）。

交付物：PyPI 套件、Hugging Face zoo、四本 Colab notebook（`examples/colab/`：15 行的 timing-driven placement、LATTE、
P11、DATE25，逐步講解）、`docs/`（concepts、features、benchmarks、api、validation、releasing）、README（英文、繁中、簡中）。

原規劃中沒有實作的介面：函式式的 `attar.place.global_place(d, engine=...)`、引擎字串切換、`attar.compare` 與解析度
判讀、`attar.set_deterministic`、Xplace／DREAMPlace adapter。實際介面以 Design 方法與 `attar.naive` 為主，第三方工具
經 DEF 或使用者自己的程式接上。

## 9. 時程與里程碑

| 日期 | 里程碑 |
| --- | --- |
| 11-09 | v0.1 規格；NS3K 原型（15 個設計） |
| 11-10–11-14 | Bookshelf、LEF/DEF、YAL、GSRC reader；GP、LG、macro／IO placement、die sizing；ORFS 交接與驗證工具 |
| 11-13 | STA 與 OpenSTA 一致（139 個設計附 timing，共用 library bundle） |
| 11-15 | 1.0.0 上 PyPI；zoo 公開上 HF |
| 11-16 | 1.0.2：DRV、power、propagated clock、CRPR、naive CTS |
| 11-17 | global router、GR 寄生；`before_fp` 的 Nangate45 macro 設計 |
| 11-19 | 1.0.4：與 ICCAD 2025 官方評估一致；GP 研究掛鉤、`critical_paths` |
| 11-21 | 1.0.5：Abacus 預設、Early library、deterministic GP、timer 唯讀資料；三篇論文的重現回饋 |
| 11-22 | 1.0.6：GR 寄生快 9–17 倍、增量寄生與增量 routing、GPU OOM 說明、英文逐步 Colab |
| 11-24 | 1.0.7：長 PG stripe 設計的 detailed placement 不再 OOM；遠離空位的增量合法化 1,250 s → 15 s |
| 11-27 | （未發布）pin access 評估與保持 pin 可達的合法化／detailed placement；LEF 層寬讀錯修正；ISPD 2018／2019 從 metal1 繞線 |

成功標準（v0.1 時訂的，現況）：

- 乾淨的 Colab 環境，`pip install` 到第一個評估結果不超過 5 分鐘：達成（timing 第一次要編譯 kernel 約 2 分鐘）。
- 3 種以上輸入格式、至少 20 個公開設計：達成（4 種格式、262 個設計）。
- 匯出的 DEF 能在 ORFS 跑完 routing：達成（`orfs_asap7` 6 個設計 0 DRC；`before_fp` 的 floorplan 交接）。
- 驗證結果公開：達成（`docs/validation/`）。
- 內建 GP 與其他 placer 的差距照實公布：達成（7.3）。

## 10. 風險與開放問題

| 風險 | 影響 | 對策 |
| --- | --- | --- |
| 評估器與參考工具在新設計或新 SDC 上不一致 | 結果不可信 | 照 OpenSTA／OpenROAD 原始碼移植；不一致之處記在 `limitations`；每次發版跑 `docs/validation/` 的比對工具 |
| 研究程式依賴 timer 的私有欄位 | timer 重構時研究程式壞掉 | 唯讀資料 API 帶版本號；下一版補上 `stages()` |
| GPU 記憶體不足（共用 GPU、百萬 cell 設計） | 跑不起來 | 1.0.6 起錯誤訊息附解法；CPU 路徑也要在幾分鐘內完成（原則 4） |
| 與文獻的數字無法直接比（ICCAD 2015 用 OpenTimer） | 使用者誤判 | 文件與 notebook 註明，只比方法間的倍數 |
| 重現研究時移植出錯（例如 P11 重現中 DATE25 pin 對只用了第一次更新） | 錯誤結論 | 重現一律在 Attar 上跑、以 Attar 評估；對照組也要驗證和官方實作相當 |
| Benchmark 授權 | 部分子集不能上 HF | 上架前逐一確認；不允許者只提供本地蒸餾 |
| 範圍蔓延 | 品質下降 | 演算法留給使用者（原則 7）；新功能先問是不是資料、評估器或基本引擎 |

開放問題：

- `attar.derive`（自動推導缺的事實）要不要做、做到什麼程度。
- Fence region 是否要在 legalizer 中遵守。
- 增量 routing 是否要讓擋路的舊 net 也能移動（品質與速度的取捨）。
- 3D placement 何時開始。
