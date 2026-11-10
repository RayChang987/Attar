# Attar MVP 規格書

Nov 9, 2026 · @JR

## 1. 精神與定位

Attar 是物理設計演算法的快迴圈：把 PDK 蒸餾成最少的數據，在 GPU 張量上以秒級速度驗證演算法；正式結論仍交給 RosettaStone 這類慢迴圈。

名稱取自 attar（玫瑰精油）：數千片花瓣，蒸餾成一滴精華。PDK 與完整 flow 是花瓣；Attar 只留下那一滴，也就是演算法真正需要的數據。

RosettaStone 2.0 往現實與 signoff 走：把學術與比賽 benchmark 翻譯成真實 PDK、合規的 LEF/DEF，跑完整 RTL-to-GDS flow。它可信，但每次評估要編譯、跑完整流程，門檻高也慢。

Attar 往反方向走：不是加上 PDK，而是蒸餾 PDK。只保留算代理指標必需的數據，例如 cell 和 macro 多大、GCell 容量多少、延遲表與線的 RC。結果不追求 signoff 精度，而追求「和真實結果有足夠相關性」。

|  | RosettaStone 2.0（慢迴圈） | Attar（快迴圈） |
| --- | --- | --- |
| 方向 | 往真實 PDK、signoff | 蒸餾 PDK、只留數據 |
| 評估 | 完整 RTL-to-GDS flow | 各環節的代理指標 |
| 單次時間 | 分鐘到小時 | 秒級 |
| 安裝 | 編譯 OpenROAD-Research、ORFS | pip install，零編譯 |
| 角色 | 裁判、正式結果 | 練習場、快速迭代與除錯 |

一句話：**GPU 上秒級迭代，ORFS 上端到端驗證。** 只要快迴圈的結論與慢迴圈有可量化的一致性，它帶來的方便與效率就值得。Attar 與 RosettaStone 是互補，不是替代。

## 2. 痛點與目標使用者

主要使用者是寫物理設計演算法論文的研究生與研究員；他們今天要花數週處理格式與 IO，才能做一次公平比較。

痛點有三個：

- **Benchmark 四散**：ISPD、ICCAD 各屆比賽、ORFS 設計、OpenCores 各有格式，同名 testcase 在不同論文用不同 library 與規則。
- **演算法四散**：Xplace、DREAMPlace、TritonRoute、各家 GPU GR、Tetris、Abacus、GPU legalizer 各有自己的 IO 與評估腳本，無法互接。
- **門檻高、迭代慢**：正式 flow 要編譯 OpenROAD、學 Tcl 與 OpenDB；GPU 工具多半需編譯 C++/CUDA extension；一次完整評估從分鐘到小時。

次要使用者是 ML for EDA 研究者：需要能快速產生大量實驗、且資料已是張量的環境。教學用途不是 MVP 目標。

## 3. 設計原則與明確不做的事

Attar 只做「全域、可張量化、容許估計誤差」的問題；需要精確幾何或 signoff 精度的，一律交給慢迴圈。代理指標分階段上線：先 wirelength，再 DRV 與 timing；每個代理都要先通過第 7 節的驗證，才會出現在評估報告的預設欄位。

設計原則：

1. **蒸餾，不是刪除**：只保留算代理指標必需的參數；原始 LEF、Liberty、DEF 以檔名與雜湊記在 manifest（不記路徑），供重現使用；IR 自給自足，快迴圈只讀 IR，不讀任何環境變數。
2. **資料全程在 GPU 張量**：只有讀檔與寫檔兩端做轉換；內層迴圈不呼叫 `.item()`、`.cpu()`，收斂判斷每 k 次迭代才同步一次。
3. **零編譯安裝**：核心只依賴 PyTorch 原生運算，不寫自訂 C++/CUDA kernel；Xplace、DREAMPlace 只當選配 adapter。
4. **每個指標都標保真度與解析度**：使用者永遠知道數字可信到什麼程度，以及差距多大才算數（第 7 節）。低於解析度的差距，報告直接標「無法判斷」。
5. **單向蒸餾，placement 可回寫**：PDK 只蒸餾、不重建。回到慢迴圈的只有演算法的輸出（cell 座標與方向，以及引擎改過的 floorplan）：`d.export(template_def, out_def)` 把它們「印」在使用者給的一份該設計的 DEF 上，其餘內容原封不動；寫出前檢查 cell library、單位、netlist 指紋與 die，不符就拒絕（`force=True` 才強寫並警告）。只有具 `cell_library` 事實（LEF/DEF 來源）的設計能 export，Bookshelf、YAL、GSRC 沒有。

回寫的理由：快迴圈有解析度上限（第 7.1 節：HPWL 差距在 1–2% 以內時，routed WL 同向率只有約 85%），而論文的改進常落在這個範圍。回寫讓使用者在移植演算法到 OpenROAD 之前，就能用 ORFS 驗證；第 7 節的驗證實驗本身也需要這條路。

慢迴圈的驗證鏈固定為 **ASAP7 + ORFS**，ORFS 版本以 docker 映像釘住（目前為 `openroad/orfs:26Q4-368-g9ca2a9e76`）。Attar 公布的解析度與相關性都只針對這條鏈；Nangate45、SKY130、RosettaStone 等其他流程可以吃回寫的 DEF，但不在 Attar 的驗證保證範圍內。

明確不做（MVP 與長期皆然）：

| 不做 | 原因 | 交給 |
| --- | --- | --- |
| Signoff STA | 需完整時序例外、多 corner、精確寄生 | OpenSTA、Innovus |
| Detailed routing、DRC | 規則密集、局部、序列化 | TritonRoute、商業工具 |
| CTS、PDN | 結構性設計決策，非全域最佳化 | ORFS flow |
| 精確寄生萃取 | 需完整幾何 | OpenRCX |
| 從 IR 重建完整 DEF、LEF、Liberty | 蒸餾掉的資訊無法還原；回寫只需替換座標 | 使用者給的 DEF 模板（manifest 記錄檔名與雜湊） |
| OpenDB（odb）讀寫後端 | 需編譯 OpenROAD，違反零編譯 | ORFS |
| RosettaStone 專屬整合 | 它吃 ORFS 格式；回寫的 DEF 能進 ORFS 即可 | ORFS 注入範例腳本 |

暫緩（不在 MVP，未來評估）：

- **Buffering**：它會改變 netlist 拓撲，需要批次編輯交易或虛擬 buffer 機制。
- **DRV 與 timing 代理**：現成代理經實測不可信（RUDY 對 DRV、HPWL 對 WNS 的排名相關都接近 0，第 7.1 節），需要另行研究，不在 v0.1 宣稱。
- **Partitioning**：資料形態（純超圖）與其他能力格格不入，等框架穩定後再評估。
- **自動補資料（`attar.derive`）**：缺的事實預設不補；推導套件之後實作，見 5.1。

## 4. 範圍：分階段上線的代理指標

MVP（v0.1，一個月）只交付 wirelength 這條線：2D placement（GP、LG）、HPWL 評估、placement 回寫，以及第 7 節的驗證。DRV、timing 代理與其他環節依序進路線圖，schema 從第一天預留。

分階段的依據是既有資料（第 7.1 節）：在 951 次 routed run 上，HPWL 對 routed WL 的排名一致性高，但 HPWL 對 WNS/TNS、RUDY 對 DRV 都接近隨機。所以 v0.1 只宣稱 wirelength，DRV 與 timing 代理各自當成研究題目，驗證通過才上線。

每個環節都宣告它的快速代理指標，以及在慢迴圈對應的真值：

| 環節 | 版本 | Attar 內的做法 | 快速代理指標 | 慢迴圈真值 |
| --- | --- | --- | --- | --- |
| Global placement（standard cell） | v0.1，已實作（`attar.naive.global_place`） | ePlace 式 electrostatic GP（參照 Xplace、DREAMPlace），全部純 PyTorch：密度以矩形角點 scatter 加 2D 前綴和精確計算；Poisson 方程以 DCT 基底矩陣乘法求解；密度梯度以後綴和 gather 精確積分；WA 梯度為解析式；Nesterov 加 backtracking；整個 objective 以 CUDA graph 重放。沒有 row 的區域（macro halo、通道）與 fixed 物件同視為障礙；預設目標密度 0.9。macro 固定 | HPWL、density overflow | routed WL |
| Legalization | v0.1，已實作（`attar.naive.legalize`） | 預設所有 cell 都是 multi-row height（k = 高 / row 高，單列高只是 k = 1）。依 (高度, rail) 分組、由高到低：rail 規則照 OpenROAD dpl（master 的上下 rail 由 PG pin 幾何判斷；R0 row 用參考單列高 cell 的 rail，翻轉 Y 的 row 互換；偶數高 cell 只能放在 rail 相容的 row，方向依 rail 決定）。每組：k 列一組的 band → 欄內垂直平衡（PAVA）→ band 內沿空段攤開 → 少量空段溢出 → 放不下的以佔用格點找最近空位（窗口逐輪放大，保證合法）→ 空段內 isotonic regression（平行 PAVA，即 Abacus 目標）。最後是平行位移改善：所有 cell 同時找 GP 位置附近的空位，重疊提案只取增益最大者（參考 OpenROAD dpl 的 negotiation legalizer），與空段重排交替三輪。全部張量化。尚無 fence 與 detailed placement | displacement、overlap 為零、rail 正確 | 合法性檢查通過（含 rail） |
| Placement 回寫 | v0.1 | 以原始 DEF 為模板，只替換可移動 cell 的座標與方向 | 回寫前後 HPWL 一致 | ORFS 從注入點跑完 routing |
| RUDY | v0.1（僅供參考） | RUDY 壅塞圖；報告中標「未驗證」，不宣稱代表 DRV | RUDY 分布 | 不設真值 |
| Detailed placement | v0.2 | 列內重排、ISM、global swap | HPWL 改善量 | routed WL |
| Global routing | v0.2 | GPU L/Z pattern routing；前綴和算路徑成本、差分陣列更新 demand；negotiation 迭代；不做 maze | GCell overflow、GR wirelength | routed WL；對 DRV 的相關性另行驗證 |
| DRV 代理 | v0.2–v0.3 | pin access 感知的指標（例如被 PDN 擋住的 pin、鄰近 pin 衝突），結合 GR overflow | 待研究 | DRV |
| Coarse STA + sizing | v0.3 | 張量版 Kahn levelization；NLDM 雙線性插值 + Elmore；sizing 改 master index 或連續鬆弛 | 粗估 WNS、TNS | ORFS WNS、TNS |
| Macro placement | v0.3 | GP 納入 macro，加上 macro legalization | HPWL、overflow | routed WL |
| 3D placement | v0.4 | GP 加上每個 cell 的 die 分配（連續鬆弛的 z 變數） | 跨層 net 數、HPWL | Pin-3D flow 的 HBT 數、QoR |

v0.1 明確不處理：macro 擺放（macro 固定）、fence region。含這些約束的 benchmark 在 v0.1 可以載入與評估，但內建引擎不保證合法。Multi-row height 與 rail parity 已支援。

原型實測（RTX 3080 Ti，目標密度 0.9，GP 後接 LG）：zoo 內可做 GP 的 35 個設計全部合法（含 rail 檢查）。ASAP7 的 aes、ibex、riscv32i、swerv_wrapper、tinyRocket 合法化後的 HPWL 比 ORFS 未合法化的 GP 低 7–20%，cva6、ethmac、jpeg 高 0.4–6%；ISPD 2005 八個設計比 DREAMPlace 的合法化結果高 2.2–6.6%。速度：aes（1.2 萬 cell）GP 0.4 秒、LG 0.16 秒；bigblue4（217 萬 cell）GP 37 秒、LG 12 秒。已知弱點：2 row 高 cell 比例很高的 NS3K dynamic_node（22%），LG 讓 HPWL 增加 30%（目標密度 0.88 時 21%）。

v0.1 的成功不是「環節都有」，而是「wirelength 的快迴圈結論在 ORFS 上站得住，而且清楚知道站得住的範圍」，見第 7 節。

## 5. 資料層：事實、能力與來源

**設計哲學：前端海納百川，Design 是狀態，有什麼題目就做什麼答案。**

- **前端海納百川**：任何 benchmark 格式都應該吃得進來，轉成同一套張量。吃 DEF、吃 odb、吃 Bookshelf 都只是選項之一；每個 reader 只負責交出它手上有的「事實」，不要求齊全。
- **Design 是狀態**：一個 Design 擁有哪些事實，就決定它能做哪些事。給了 floorplan 和 netlist，就能做 global placement；再有 row 和 site，就能做 legalization、detailed placement；再有 routing 資源，就能做 global routing；有 timing 資料，就能做 timing 分析與最佳化。框架負責這套機制，每個能力頂多附一個最簡單的演算法。
- **有什麼題目，做什麼答案**：演算法的產出寫回 Design 的解答狀態，於是又解鎖下一個能力（例如 GP 產生 placement 之後，才能做 legalization）。

### 5.1 事實、能力與來源

**事實（facts）**：Design 的資料拆成一組細的事實，每項可以有也可以沒有。

| 事實 | 內容 | 類別 |
| --- | --- | --- |
| `netlist` | cell、net、pin 的連接，net 權重 | 題目 |
| `cell_geometry` | cell 寬高、cell 旗標（movable、fixed、macro） | 題目 |
| `pin_offsets` | pin 相對 cell 的位置（存 master 原始 offset 加方向） | 題目 |
| `floorplan.die` | die 陣列（每個 die 的外框） | 題目 |
| `floorplan.io` | IO pin 的位置（die 邊界上） | 題目（可由 IO placer 產生） |
| `floorplan.macros` | macro 與 fixed cell（如 tapcell）的位置 | 題目（可由 macro placer 產生） |
| `rows` | row 位置、site 寬、row 高度、所屬 die | 題目 |
| `constraints` | fence region、placement blockage | 題目 |
| `routing` | layer 方向與 pitch、GCell grid 與容量、pin access | 題目 |
| `timing` | Liberty 的 arc 與 pin cap、clock、I/O delay | 題目 |
| `stack` | 3D：各 die 的製程、HBT 參數 | 題目 |
| `cell_library` | cell 是真實 cell library（LEF master）的實例，結果可印回 DEF | 題目 |
| `tracks` | 各層 routing track 與每個 macro pin 的 layer（macro pin 對齊 track） | 題目 |
| `placement` | 所有可移動 cell 的位置與方向 | 解答狀態 |

**能力（capabilities）**：每個演算法宣告它需要哪些事實，Design 由此推出 `capabilities()`。缺事實時，`d.missing("global_route")` 會列出缺哪幾項，不只是報錯。

| 能力 | 需要 |
| --- | --- |
| 評估 HPWL | `netlist`、`cell_geometry`、`pin_offsets`、`placement` |
| IO pin placement | `netlist`、`floorplan.die` |
| macro placement | `netlist`、`cell_geometry`、`pin_offsets`、`floorplan.die`、`floorplan.io` |
| global placement | `netlist`、`cell_geometry`、`pin_offsets`、`floorplan.die`、`floorplan.io`、`floorplan.macros` |
| legalization、detailed placement | 再加 `rows`、`placement` |
| global routing | 再加 `routing`、`placement` |
| STA、sizing | `netlist`、`timing`（有 `placement` 時才估線的 RC） |
| export（印回 DEF） | `netlist`、`cell_library` |

**題目與解答狀態分開**：題目（netlist、幾何、floorplan、tech）基本不變；解答狀態（位置、方向、die 分配，之後還有 route 與 sizing 結果）由演算法改寫。解答狀態可以存成具名 snapshot，例如蒸餾時保留的來源 placement 就是一個叫 `reference` 的 snapshot。會改 netlist 的操作（buffering、sizing）同時動到題目，需要另外的交易機制，列為後續工作。

**來源（provenance）與不自動補**：manifest 記錄每項事實的來源（例如 `def`、`lef`、`bookshelf`）。缺的事實預設不補、明確報錯，不用假資料。

- **TODO：推導套件 `attar.derive`**。很多缺的事實可以推出來：沒有 row 就用 die 加 cell 高度生成；沒有 die 就用利用率推算；沒有 GCell 容量就用 LEF track 數算；Bookshelf 沒有製程資料就掛一份蒸餾過的 tech profile。這些都只能明確呼叫（例如 `d.derive("rows")`），推導出的事實標記為 `derived:<方法>`，報告與 `compare` 一律印出來源。v0.1 不實作，只保留介面位置。

**前端（readers）**：每個 reader 宣告它提供哪些事實；多個來源可以疊加成一個 Design（例如 Verilog netlist 加 DEF floorplan 加 Liberty），用正規化後的名稱對齊。

| 前端 | 提供的事實 | 版本 |
| --- | --- | --- |
| LEF/DEF | 視 DEF 內容，通常全部題目事實，可回寫；有座標時另含 placement | v0.1 |
| Bookshelf（ISPD 2005/2006、ICCAD 2004） | `netlist`、`cell_geometry`、`pin_offsets`、`floorplan`、`rows`；`.pl` 有座標時另含 placement。Bookshelf 沒有 die 外框，以 row 和 fixed 物件的外框為 die，來源標為 `bookshelf(rows+fixed)` | v0.1，已實作（ISPD 2005 全 8 個設計） |
| Bookshelf + `.route`（ISPD 2011、DAC 2012） | 再加 `routing` | v0.2 |
| Verilog + LEF + Liberty | `netlist`、`cell_geometry`、`pin_offsets`、`timing`，沒有 floorplan | 之後 |
| MCNC、YAL、GSRC | block 與軟模組，需另加 floorplan 視圖 | 之後 |
| odb | 經 ORFS 匯出 DEF，或選配的 OpenROAD Python 綁定 | 經 DEF |

Partitioning 暫不列入：它的資料（純超圖）與其他能力的資料形態差距大，等框架穩定後再評估。

一開始就定死的慣例：

- **die 一律是陣列**：2D 設計就是 1 個 die（DEF 的 DIEAREA），3D 設計有多個 die（tier）。die 外框存成 `(D, 4)`；row 帶所屬 die，cell 帶所屬 die（2D 時全為 0，3D placement 會改這個變數）；manifest 的 `dies` 列表記錄每個 die 對應哪份 tech profile。依 die 計算的量一律以 die 為第一維（例如密度圖是 `(D, nx, ny)`），所以 2D 和 3D 共用同一套程式碼。bonding terminal 不是 die，放在 3D 層另一張表。
- **IO pin 只收 signal pin**：VDD、VSS 這類 PG pin 不屬於 signal netlist，蒸餾時排除（manifest 記錄排除的數量）。
- **座標**：cell 左下角為原點，與 DEF 一致；Bookshelf 載入時轉換。
- **Pin offset**：一律相對 cell 中心，並寫進文件。
- **單位**：磁碟存整數 DBU；載入 GPU 轉浮點並記錄縮放係數，輸出轉回整數。
- **方向**：存 master 原始 pin offset 加 orientation，有效 offset 需要時才算。
- **數值與字串分離**：數值在 GPU 張量；cell、net、master 名稱放 CPU 對照表，只在 IO 時使用。

### 5.2 Tech Profile：蒸餾 PDK

Tech Profile 是一小組張量，只保留算代理指標必需的參數；線寬、間距、pin 形狀、DRC 規則、power 表一律丟掉。

| 用途 | 保留的參數 |
| --- | --- |
| Routing | GCell 尺寸、每層方向與每 GCell track 容量、via 成本 |
| 線的 timing | 每層單位長度 R、C，via R |
| Cell 的 timing | 每個 cell type 的 NLDM 延遲表（或固有延遲 + 驅動電阻）、pin 輸入電容 |
| 約束 | clock period、基本 I/O delay |

來源有兩種：

- **從真實 PDK 蒸餾**：先做驗證鏈用的 ASAP7，讀 ORFS 平台的 LEF 與 Liberty 萃取，保住與慢迴圈的對應；Nangate45、SKY130 等平台之後依需求加入。
- **合成掛載**：Bookshelf 等無製程資料，借用一份蒸餾過的 profile 按比例調整後掛上，讓 routing、timing 代理跑得起來。這屬於推導，只能經 `attar.derive` 明確呼叫（TODO，見 5.1）。

### 5.3 保真度分級

每個 benchmark 與每個輸出指標都帶保真度標籤，寫在報告裡。保真度等級是 5.1 各事實來源的摘要：F1、F2、F3 由有哪些事實、事實是否為推導而來決定。

| 等級 | 資料 | 可信的指標 | 範例 |
| --- | --- | --- | --- |
| F1 真實製程 | 有 LEF + Liberty（+ SDC） | placement、routing 代理、粗略 timing，可與慢迴圈對照 | ORFS ASAP7 設計、ICCAD 2015 |
| F2 有 routing 資訊 | 有 GCell 容量，無 Liberty | placement、routing 代理；timing 為合成，只能相對比較 | ISPD 2011、DAC 2012 |
| F3 純幾何 | 只有尺寸與連接 | 只有 placement 指標 | ISPD 2005/2006 Bookshelf |

### 5.4 磁碟格式

每個設計一個資料夾：張量存 safetensors，加一份 JSON manifest，記錄 schema 版本、來源格式、DBU、存在哪些層、保真度等級、原始檔案的檔名與雜湊（不記路徑）。首次讀原始格式時寫入快取，之後直接讀快取。散布方式見 5.6。

原型實測（RTX 3080 Ti）：118 萬 cell、220 個 macro 的 bp_quad，從 1.15GB DEF 蒸餾 34 秒（一次性），之後從快取載入到 GPU 0.3 秒、IR 95MB、HPWL 1.5 ms、WA wirelength 前向加反向 8.5 ms、GPU 記憶體 0.9GB。ASAP7 的 9 個 ORFS 設計（含 4 個 macro 設計，最大 12.5 萬 cell）載入都在 0.5 秒內，HPWL 與 OpenROAD GPL 回報值差 −0.66% 到 +0.03%。manifest 另記資料品質警告，例如 ORFS `AUTO_MEMORIES` 產生的 macro LEF 有 signal pin 超出外框。

### 5.5 Benchmark Zoo：收錄範圍

過去散落各處的 contest 與學術 benchmark 全部蒸餾進同一份 IR；v0.1 先收 placement 類，其他類別依 IR 層的實作進度加入。

| 類別 | 代表 benchmark | 原格式 | IR 層 | 保真度 | 收錄版本 |
| --- | --- | --- | --- | --- | --- |
| 經典 placement | ISPD 2005、ISPD 2006 | Bookshelf | Core | F3 | v0.1 |
| IBM mixed-size | ICCAD 2004 ibm01 到 ibm18 | Bookshelf | Core | F3 | v0.1 |
| Routability-driven | ISPD 2011、DAC 2012 superblue | Bookshelf + .route | Core + Routing | F2 | v0.1 Core；Routing 層 v0.2 |
| 現代 LEF/DEF placement | ISPD 2015（fence）、ICCAD 2017（multi-row height） | LEF/DEF | Core + Constraint | F2 | v0.1 |
| Timing-driven | ICCAD 2015 | LEF/DEF + Liberty + SDC | + Timing | F1 | v0.1 Core；Timing 層 v0.3 |
| Macro placement | MacroPlacement（Ariane、BlackParrot、MemPool 等） | LEF/DEF + Liberty | 全部 | F1 | v0.1 |
| 真實 flow 設計 | ORFS ASAP7 設計（`flow/designs/asap7`） | ORFS `3_3_place_gp.odb` 匯出的 DEF | 全部 | F1 | v0.1（驗證鏈） |
| 其他 ORFS 平台 | Nangate45、SKY130 設計 | 同上 | 全部 | F1 | v0.1 之後 |
| Floorplanning | MCNC（ami33、ami49）、GSRC、IBM-HB | YAL、GSRC 格式 | 需新增 Floorplan 視圖 | — | v0.1 之後 |
| 3D placement | ICCAD 3D placement 題目、Pin-3D 設計 | 各自格式、DEF | + 3D | F1 到 F3 | v0.4 |

Floorplanning benchmark 是 block 層級。硬模組直接放進 Core 層（每個 block 是一顆未擺放的 macro）；軟模組只有面積和長寬比範圍，形狀是待決定的答案而不是事實，所以存成 `soft_area`、`soft_aspect` 兩個陣列，不宣稱 `cell_geometry`。

**現況（2026-11-09，`tools/distill_benchmarks.py`，共 125 個設計）**

| zoo suite | 讀取器 | 設計數 | 有的事實 | 一載入就能做 |
| --- | --- | --- | --- | --- |
| `ispd2005` | Bookshelf | 8 | 全部 Core（tier C） | GP、LG |
| `ispd2015` | LEF/DEF | 20 | Core + cell_library + tracks；fence、blockage 只計數（`not_stored`），`placement.constraints` 的 maximum_utilization 記為 `target_density` | GP、LG、export |
| `ispd2018`、`ispd2019` | LEF/DEF | 10 + 10 | Core + cell_library + tracks + placement；route guide 未讀 | HPWL、合法性、重新擺放 |
| `ispd2024` | LEF/DEF（Nangate45） | 5 | 同上；`.cap/.net` 未讀 | 同上 |
| `iccad2025` | LEF/DEF（ASAP7 + Liberty 路徑，F1） | 13 | 同上；Verilog、SDC、SPEF 未讀 | 同上（11 個輸入的 cell 全是 N，rail 不合法） |
| `gsrc_place` | Bookshelf | 4 | 全部 Core | GP、LG |
| `ispd2006_hb` | Bookshelf（IBM-HB+、Calypto） | 28 | Core；可動的硬 block 讓 `floorplan.macros` 缺席；軟 block 以 .nodes 形狀、只計數 | macro placement |
| `mcnc` | YAL | 5 | netlist、幾何、pin offset、die（parent outline）、pads | macro placement（floorplan） |
| `gsrc_hard` | GSRC blocks | 6 | netlist、幾何（pin 在 block 中心）、pads；無 die | 開 die → floorplan |
| `gsrc_soft`、`mcnc_soft` | GSRC blocks | 6 + 5 | 只有 netlist（與 pads 或 outline） | 幾乎沒有：缺 `cell_geometry` |

讀取器都只交出格式本身有的事實。DEF 中狀態為 PLACED 的 macro 視為 floorplan 的一部分而凍結；Bookshelf 中不在 row 範圍內的 terminal 視為 IO pad；Bookshelf 座標的 DBU 依資料選 2、10、100、1000（IBM-HB+ 的 √面積 尺寸需 1/1000 並記錄捨入誤差）。尚未收：ICCAD 2022/2023 3D（每個 cell 在兩個 die 上尺寸不同，需要 per-die 幾何）、ICCAD 2020/2021 cell move 與 ICCAD 2018 bus routing（routing 層）。

同一設計出現在多屆比賽時（例如 superblue），netlist 只存一份；各屆不同的約束或容量另存小檔，manifest 記錄 lineage。

### 5.6 散布：Hugging Face Datasets

所有蒸餾後的資料放在 Hugging Face Datasets（暫名 `attar-zoo`），不進 git、不進 pip 套件；授權不允許散布的子集只提供本地蒸餾腳本。

三層散布：

1. **pip 套件內**：一兩個極小的 demo 設計（數 MB 內），裝完即可跑。
2. **Hugging Face dataset**：授權允許的子集，按設計懶下載、雜湊驗證、本地快取。
3. **本地蒸餾**：授權不明或不允許的子集，使用者從官方下載後執行 `attar distill`。

Dataset 目錄結構：`<suite>/<design>/` 下放 `core.safetensors.zst`、`names.zst`、`tech_profile.safetensors`、`manifest.json`。每個 suite 一份 dataset card，寫明原始出處、引用格式與授權。

載入流程：`attar.load("ispd2005/adaptec1")` 先查本地快取，沒有就經 `huggingface_hub` 下載並驗證雜湊；若該子集只允許本地蒸餾，回傳明確訊息，告訴使用者去哪下載原始檔、執行哪個指令。

版本管理：每個 Attar 版本釘住一個 dataset revision；manifest 記錄 schema 版本與原始檔雜湊；所有蒸餾結果都能由 repo 內腳本重現。

體積控制：座標存 int32 DBU，不用 float16；名稱字串獨立存並以 zstd 壓縮；整套經典 benchmark 蒸餾後預估為數 GB 未壓縮、壓縮後約 1 GB 量級（待實測）。

授權盤點（每個子集上架前必須確認）：

| 子集 | 授權狀態 | 處理 |
| --- | --- | --- |
| ISPD 2005、2006 | 可散布 | 確認後決定 HF 或本地蒸餾 |
| ICCAD 2004 IBM | 可散布 | 同上 |
| ISPD 2011、DAC 2012 superblue | 可散布 | 同上 |
| ISPD 2015 | 可散布 | 同上 |
| ICCAD 2015、2017 | 可散布 | 同上 |
| MacroPlacement | 可散布 | 逐一 testcase 確認 |
| ORFS 設計 | 可散布 | 依各設計與 PDK 授權 |
| MCNC、GSRC、IBM-HB | 可散布 | 同上 |

## 6. 系統架構

架構採編譯器式設計：多個前端、一份 IR、多個後端；N 種格式與 M 種演算法只需 N + M 個轉接器，而不是 N × M 個。

&#91;embedded content: Attar 架構 · 讀寫兩端之外全在 GPU\]

Loader 把各種 benchmark 轉成同一份 IR；引擎與評估器只讀寫張量，全部建在同一組 primitive 上；writer 把結果交回 ORFS/RosettaStone 慢迴圈或比賽官方 evaluator。v0.1 只實作前三個引擎。

## 7. 準確度驗證：快慢迴圈一致性

Attar 的核心主張要證明兩件事：快迴圈說 A 比 B 好時，慢迴圈大多也這麼說；以及差距要多大，這句話才成立。這是整個專案最重要的實驗。

為什麼必要：ChiPBench 發現中間指標常與端到端 PPA 不一致，主張應重視端到端結果。Attar 的回應不是反駁，而是量出哪些代理指標在什麼條件下可靠，並把這個條件寫進每一份報告。

### 7.1 既有證據：pa_pdn 的 951 次 routed run

pa_pdn 專案留下的 routed run 已經可以當第一輪證據：

- **製程與工具**：i3dPDK_NS3K（3nm 級，M1 只做 pin access，M2 以上單向繞線），Innovus 21.13。
- **規模**：16 個 ORFS 設計 × 2 種 PDN × 約 60 種 placement 方法，共 34 組（設計 × PDN × 利用率）。
- **代理的量法**：從交給 router 的最終 placement 計算（flow 在 routing 之後沒有任何優化）。

各組內的排名相關（Kendall τ，已調整正負號，+1 表示代理與真值完全一致）：

| 代理 → 真值 | τ 平均 | τ 中位數 | 代理最佳者在真值也最佳 |
| --- | ---: | ---: | ---: |
| HPWL → routed WL | 0.78 | 0.85 | 65%（隨機約 4%） |
| HPWL → WNS | −0.03 | −0.05 | 3% |
| HPWL → TNS | 0.02 | −0.01 | — |
| RUDY（最壅塞 1% bin 平均）→ DRV | 0.04 | 0.02 | 32% |

兩兩比較，依 HPWL 相對差距分組，看 routed WL 是否同向：

| HPWL 差距 | <0.1% | 0.1–0.25% | 0.25–0.5% | 0.5–1% | 1–2% | 2–5% | >5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 同向率 | 55% | 69% | 74% | 84% | 86% | 97.5% | 99.8% |

結論：

1. **HPWL 對 routed WL 可信，但有解析度。** 差距 ≥ 2% 幾乎一定同向；0.5–2% 約 85%；低於 0.25% 接近丟銅板。
2. **量測點很重要。** 改用 CTS 階段的 HPWL（尚未經過最後的 placement refinement），τ 從 0.78 降到 0.35，差距超過 5% 才可靠。Attar 在 GP/LG 後量，ORFS 之後還會做 resize、DP 與 CTS，預期會落在這兩者之間，必須實測。
3. **HPWL 不能代表 timing，RUDY 不能代表 DRV。** WNS 與 DRV 在每個差距區間的同向率都在 40–60%。本資料的 DRV 主要來自 pin access 與 PDN，而非全域壅塞，所以對 RUDY 不利；但這正是先進製程的常態。

這是 NS3K + Innovus 上的初步證據，用來決定 v0.1 的範圍；正式的驗證鏈是 7.2 的 ASAP7 + ORFS。

限制：單一 PDK、單一工具（Innovus 而非 ORFS）、方法集中在 pin access 研究；重跑樣本只有 8 對，量不出慢迴圈本身的雜訊。各組方法差異大（少數方法讓 HPWL 差到 70%），會把 τ 灌高，因此以差距分組表為準。

### 7.2 v0.1 實驗設計（ASAP7 + ORFS）

0. **環境**：ORFS 以 docker 映像釘住版本（`openroad/orfs:26Q4-368-g9ca2a9e76`，ASAP7 平台）；每次實驗的 manifest 記錄映像版本與各設計的 `config.mk` 雜湊。asap7/gcd 的完整 flow 約 33 秒，可當每日煙霧測試。
1. **設計**：從 ORFS `flow/designs/asap7` 選 8 到 12 個，涵蓋小型、中型與含 macro 的設計（v0.1 macro 固定）。
2. **組合**：每個設計 10 到 20 種 placement。第 1 週先用 ORFS 自己的 global placement 換 target density、seed、timing-driven 開關產生組合，在 Attar GP 完成前就把量測流程跑通；之後加入 Attar 內建 GP 與 Xplace adapter。
3. **注入點**：替換 ORFS `3_3_place_gp.odb` 裡的 placement。
   - **蒸餾來源**：`3_3_place_gp.odb` 匯出的 DEF，不用 `3_2`。ORFS 的 `global_place.tcl` 在 GP 時還會做 port buffering 並擺放 IO，所以只有 `3_3` 的 netlist 是下游實際使用的那份。`3_2_place_iop.odb` 有保留，但內容依設定而異：`GPL_PLACE_IOS=1` 的設計（gcd、aes、jpeg、ethmac、tinyRocket）的 `3_2` 只是 `3_1` 的複本，IO 要到 `3_3` 才擺；其餘設計（ibex、riscv32i、swerv_wrapper、cva6）在 `3_2` 擺 IO。ORFS 的 GP 結果另存為參考 placement，Attar 引擎則從不帶座標的狀態開始。
   - **注入**：Attar 回寫 DEF 後，ORFS 容器內用 `read_def -incremental` 把座標套回 `3_3_place_gp.odb`。只餵 COMPONENTS 段，否則 PINS 會被重複加入，導致 routing 失敗。之後由 ORFS 接著做 resize、DP、CTS 與 routing。
   - **odb**：轉換只在 ORFS 端進行，Attar 本身不碰 odb。
   - **已在 asap7 的 gcd、aes、ibex、jpeg、ethmac 驗證**：原樣回寫後重跑，下游所有指標與 ORFS 原始結果逐位相同（routed WL、DRC、WNS、TNS、DP HPWL）；位移擾動後的結果則確實不同，代表注入生效。ORFS 是 deterministic 的，同一輸入必得同一輸出。
   - **floorplan 注入點（2_1）**：Attar 擺的 die、row、IO pin、macro 由 `tools/orfs_handoff.py` 套到 `2_1_floorplan.odb`：
     - **怎麼套**：die 變了就用 `initialize_floorplan` 加平台的 `make_tracks`；IO pin 用 `place_pin`，狀態 FIXED；macro 狀態 FIRM。
     - **跳過 ORFS 的 macro placer**：同一份 odb 也存成時間較新的 `2_2_floorplan_macro.odb`，make 就不會重跑 2_2。
     - **ORFS 接著做**：tapcell、PDN、GP、CTS、routing。`place_pins` 和 `-place_ios` 找不到要擺的 pin，所以 IO 不會被動。
     - **已驗證**：asap7 的 aes、riscv32i、tinyRocket、cva6，以及 aes、riscv32i 改過 die 大小（`make_die`）後，都繞通、0 DRC，最終 DEF 裡的 macro 與 pin 都沒動。
     - **前提**：macro pin 必須落在其層的 track 上（`Design.macro_grid`）。對照實驗中，ORFS 自己的 macro 擺法平移 24 dbu 讓 pin 離開 track，detailed routing 就失敗。
   - **兩個量測點**：同時記錄兩個量測點的 HPWL：Attar 輸出時與 ORFS routing 前，用來分開「代理本身的誤差」和「下游步驟造成的漂移」。
4. **慢迴圈雜訊**：ORFS 是 deterministic 的，換 seed 量不到雜訊，所以用微擾：隨機挑 1% 的可移動 cell 各移 1 個 site（HPWL 變化約 0.005%），重跑 3 次，量慢迴圈對近乎相同輸入的反應幅度，當作雜訊底線。asap7 四個設計的實測（baseline 加 3 次微擾的最大減最小，相對 baseline）：

   | 設計 | cell 數 | routed WL | GRT WL | WNS | TNS |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | aes | 12k | 0.29% | 0.70% | 89% | 98% |
   | ibex | 21k | 0.25% | 0.16% | 34% | 104% |
   | jpeg | 58k | 0.26% | 0.27% | 9.8% | 7.6% |
   | ethmac | 60k | 0.09% | 0.18% | 7.2% | 4.5% |

   - **Wirelength**：單次 ORFS run 的 routed WL 雜訊約 0.1–0.3%，因此 HPWL 差距在約 0.5% 以下時，慢迴圈本身就判斷不了誰比較好，這和 7.1 中差距 <0.25% 接近丟銅板的結果一致。把 10% 的 cell 移 ±3 sites（HPWL +0.1–0.2%），routed WL 的變化仍落在雜訊範圍內。
   - **Timing**：單次 run 的 WNS、TNS 由 repair_timing 的混沌行為主導，微擾就能變動 5–100%。比較 timing 時，每個 placement 至少要跑 3 次微擾，用平均與範圍比較；單次 run 的 WNS/TNS 不能當真值。
   - **算力**：asap7 中型設計（12k–60k cell），每次從 3_4 跑到 finish 約 10–60 分鐘（2–4 核）。10 個設計 × 15 種 placement × 3 次微擾約 450 次 run，以 20 核估計約需 1–2 天。
5. **快迴圈**：Attar 算 HPWL、overflow、displacement；RUDY 照算，但只作參考。
6. **慢迴圈**：收 routed WL、DRV、WNS、TNS。
7. **指標**：主指標是依代理差距分組的兩兩同向率（同 7.1）。輔助指標是各設計內的 Kendall's τ 與 top-1 一致率，並附隨機基準。

報告方式：README 首頁放「解析度曲線」（代理差距 vs. 同向率）與「代理 × 真值」相關係數熱圖，並列出不一致的案例。不一致案例本身是研究題材，例如 pin access 造成的落差。

合格門檻（暫定，第一輪數據後調整）：

- **解析度**：在 ORFS 上，HPWL 差距 ≥ 2% 的配對，routed WL 同向率 ≥ 90%（pa_pdn 實測 97.5%）。
- **報告整合**：實測出的解析度寫進評估器；低於解析度的差距自動標「無法判斷」（第 8 節）。
- **τ 只作參考**：pa_pdn 的 0.78 輕鬆超過原先 0.6 的門檻，不具鑑別力。
- **WNS、TNS、DRV 照收照報、不設門檻**：它們是 v0.2 之後 DRV 與 timing 代理的基準數據；timing 一律以多次微擾的平均與範圍報告。
- **解析度不得低於雜訊**：公布的解析度門檻至少是該設計慢迴圈雜訊範圍的 2 倍（以上表約為 0.5%）。
- **未達門檻**：照實公布，用實測解析度發布，並在 README 首頁說明哪些差距不可信。

## 8. API 與使用體驗

目標體驗：從 `pip install` 到第一個可比較的結果，不超過五分鐘、五行程式；覺得方法可行時，再用一行回寫，交給 ORFS 驗證。

```python
import attar
d = attar.load("orfs/asap7/aes", device="cuda")        # 首次下載並轉換，之後讀快取
d = attar.place.global_place(d, engine="builtin")       # 可換 "xplace"、"dreamplace"
d = attar.place.legalize(d, engine="tetris")
print(attar.compare(d, baseline="builtin"))             # HPWL 等差距，附保真度與「可信／無法判斷」
d.export("aes.def", "out.def")                          # 印在該設計的 DEF 上，交給 ORFS 驗證
```

API 原則：

- **函式式**：每個環節吃 design、回傳 design，方便任意串接與替換。
- **引擎即字串**：同一環節的不同演算法用 `engine=` 切換；第三方引擎做成選配 extra，例如 `pip install attar[xplace]`。
- **自己的演算法一個函式就能接**：使用者只要寫一個吃張量、回傳座標的函式，就能用同一評估器與內建 baseline 比較。
- **比較附解析度**：`attar.evaluate(d)` 回傳指標與保真度標籤；`attar.compare` 回傳相對差距，並依第 7 節實測的解析度判讀為「可信」或「無法判斷」。未經驗證的指標（v0.1 的 RUDY）標「未驗證」，不參與判讀。
- **回寫只換座標**：`attar.export` 以 manifest 記錄的原始 DEF 為模板（先驗證雜湊），只替換可移動 cell 的座標與方向；fixed cell、tapcell、IO、PDN、netlist 原封不動。來源是 Bookshelf 的設計呼叫 `export` 會明確報錯，並說明原因。
- **ORFS 注入範例**：文件附一份在 ORFS docker 映像內執行的腳本：把回寫 DEF 的 COMPONENTS 套回 `3_3_place_gp.odb`，接著跑 resize、DP、CTS 與 routing，並收回 routed WL、DRV、WNS、TNS。預設平台是 ASAP7；RosettaStone 吃 ORFS 格式，不另做整合。
- **評估報告結構化**：輸出欄位命名對齊 METRICS2.1，方便與慢迴圈對照。
- **三個核心物件：Design、Evaluator、Naive**：Design 是狀態（題目加解答）；Evaluator 量測它；Naive 是每種能力的一套基本引擎，讓 Design 動起來。指標和引擎都宣告需要哪些事實，所以能量什麼、能做什麼由 Design 有什麼決定（`ev.available()`、`ev.missing("timing")`、`d.capabilities()`）。
  - Evaluator 已實作：HPWL、star wirelength、Steiner wirelength（3 pin 以下精確；以上取 single-trunk、RMST、star 三種真實樹的最短者，為 RSMT 上界）、RUDY、density overflow、legality（multi-row 感知）。待實作：global routing 的 wirelength 與 overflow、timing、pin access、熱分析。
  - Naive 已實作：IO pin placement（die 四邊接成一條線，依連接 cell 的重心或名稱順序定目標，平行 PAVA 排到 pitch 格點上）、macro placement（mixed-size GP 後，所有 macro 兩兩平行推開重疊、加 halo、對齊格點、固定）、global placement、legalization。Naive 擺過 IO 或 macro 後，Design 不再有 `writeback` 與 `slow_loop`（export 只寫 COMPONENTS，不寫 PINS 與 macro）。Naive macro placer 只看 wirelength，不考慮繞線通道與 pin access；推開後仍重疊的 macro 依面積由大到小，在 site 格點的佔用格上找最近空位（2D prefix sum 一次算出所有空位），block 密度約 80% 的 floorplan（IBM-HB+、Calypto）仍會剩少量重疊。Die sizing（`make_die`）：core 面積 = std-cell 面積 / utilization + macro 面積，依 aspect ratio 定長寬，沿用設計原有 row 的 site、row 高、row 方向交替與 die 到 core 的 margin，需要能力 `die_size`（`cell_geometry`；有 row 時沿用其 site 格點，沒有 row 的 block 設計則 die 即 core）；新 die 讓舊的 IO、macro、fixed cell（tap cell 等）、placement 與 snapshot 都失效，也拿掉 `writeback` 與 `slow_loop`。detailed placement 暫不做。
  - Design 上的捷徑：`d.floorplan(utilization=, aspect_ratio=)`（有給 utilization 就 make_die，再補上缺的 IO、macro；IO 先依名稱、粗略 GP 後再依連接重新擺；結束時 cell 回到未擺放）、`d.place()`（global placement + legalization）、`d.plot()`。範例：`examples/quickstart.py`。
- **HPWL 求快**：一次 scatter 對 `(x, y, −x, −y)` 取最大值，精確且無近似。實測（RTX 3080 Ti）：約 100 萬 pin 以下，單次呼叫受啟動開銷限制，約 0.15 ms；CUDA graph（`fast=True`）降到 0.05 ms；`hpwl_batch` 一次評估多組 placement，小設計每組只要 3–14 µs。大設計受計算限制：`torch.compile`（`compile=True`）快約 1.8 倍，890 萬 pin 的 bigblue4 為 1.8 ms，有效頻寬約 200 GB/s，約為顯卡峰值的四分之一。剩下的空間要靠不用 atomic 的 CSR 分段 kernel（Triton），列為選配。
- **可重現模式**：`attar.set_deterministic(True)` 開啟 PyTorch deterministic algorithms，代價是稍慢；不支援 deterministic 的運算列在文件中。
- **選配加速**：`torch.compile` 作為可開關選項，不是預設依賴。

交付物：PyPI 套件、Colab demo、套件內附一個小型 demo 設計、文件站、ORFS 注入範例腳本、Hugging Face dataset（attar-zoo）與本地蒸餾腳本。

## 9. 時程、里程碑與成功標準

v0.1 在四週內交付，最關鍵的閘門在第 3 週末：相關性實驗過門檻，才進入發布。

&#91;embedded content: v0.1 四週計畫與後續版本 · 2 個閘門\]

若第 3 週相關性未達門檻，第 4 週優先改進代理指標，發布時照實列出哪些指標可信。

v0.1 成功標準：

- 在乾淨的 Colab 環境，`pip install` 到第一個評估結果不超過 5 分鐘。
- 3 種輸入格式可用；至少 20 個公開設計可載入並通過 validator。
- 匯出的 DEF 能在 ORFS 跑完 routing。
- 相關性實驗結果與不一致案例公開於 README。
- 內建 GP 與 Xplace 在同一組設計上的 HPWL 與速度差距照實公布。
- 發布後一個月內 30 到 50 個 GitHub stars。

## 10. 風險與開放問題

最大的風險是代理指標與真值不相關；其次是被資源更多的團隊搶先佔位。

| 風險 | 影響 | 對策 |
| --- | --- | --- |
| 代理指標與慢迴圈相關性低 | 核心主張不成立 | 第 7 節實驗提前到第 3 週；照實公布，改進代理（加入 RUDY、GR） |
| 純 PyTorch 速度遠輸 Xplace | 「快」的說服力下降 | 量出實際倍數放 README；強調「安裝到第一個結果」的時間；保留可抽換 backend |
| 北大等團隊推出類似整合平台 | 定位被佔 | 盡早發布 placement 版；強調零編譯與對接慢迴圈 |
| Benchmark 授權限制 | 部分子集不能上架 HF | 逐一授權盤點（5.6）；不允許者只提供本地蒸餾腳本 |
| LEF/DEF 子集解析器遇到不支援語法 | 部分設計載入失敗 | 明確列出支援範圍；odb Python 綁定做選配後端 |
| 範圍蔓延 | 一個月交不出可用版本 | v0.1 只承諾 placement + RUDY；其他只在 roadmap |

開放問題：

- 名稱 attar 目前在 PyPI 未被佔用（2026-11-09 查詢），GitHub org 與 PyPI 名稱應盡快註冊。
- Pin offset 以 cell 中心為準是否會讓 DEF 轉換複雜化，需實作後確認。
- 是否主動聯繫 RosettaStone 團隊，爭取在 ORFS-Research 文件中互相引用。
- 合格門檻（τ ≥ 0.6、top-1 ≥ 70%）是否合理，待第一輪數據。
