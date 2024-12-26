# 情緒指標建立
本分析採用字典法，適用於各種中文資料來源的情緒計算，主要情緒計算的方式參考自 Barbaglia (2024)，並採用兩套中文情緒字典，分別建構自 Bian (2019) 以及 Du (2021)，並將簡體中文轉換為繁體中文。本程式使用字典法來示範建立每日新聞情緒指標，根據不同計算邏輯，我們將分別定以下五種不同計算方式的情緒指標。而在定義這些指標之前，我們先定義相關的符號。

假設每日新聞的總數為 $N_t$，其中 $t$ 表示日期時間戳記（例如 2024 年 1 月 1 日）。對於當天的每一篇新聞，標記為 $A_{t,j}$，其中 $j = 1, \dots, N_t$。每篇新聞 $A_{t,j}$ 首先經由 CKIP 工具進行斷詞（Word Segmentation）和詞性分析（POS Tagging），以得到更精細的詞彙單位。每篇新聞 $A_{t,j}$ 可表示為一組斷詞結果 $\{ WS_{t,ji} \mid i = 1, \dots, M_{t,j} \}$，其中 $M_{t,j}$ 表示在時間點 $t$ 下，新聞 $A_{t,j}$ 的總斷詞數。

為了容易理解本範例情緒分數計算的邏輯，我們將字典來自於 Bian (2019) 以及 Du (2021) 分別定義為字典 $A$ 與 $B$，並分別延伸兩本字典的交集與聯集，分別定義為 $A \cap B$ 以及 $A \cup B$。而本範例所關心的關鍵字亦包含三種組合：產業、總體經濟與不分類，分別標注為 $\mathbb{K}_{\mathrm{ind}}$、$\mathbb{K}_{\mathrm{macro}}$ 與 $\mathbb{K}$。

## 情緒指標一：

第一個指標，以情緒詞彙集合字典 $A$ ($\mathrm{DICT}_{A,\mathrm{ps}}$) 為例，每日情緒分數的計算基於正面情緒詞彙集合 $\mathrm{DICT}_{A,\mathrm{ps}}$ 中字詞的出現頻率。具體而言，對於每篇新聞 $A_{t,j}$，正面情緒分數 $S_{t,j}^{\mathrm{A,ps}}$ 定義為新聞中所有斷詞 $WS_{t,ji}$ 屬於正面情緒字典 $\mathrm{DICT}_{A,\mathrm{ps}}$ 的詞彙總數：

$$
S_{t,j}^{\mathrm{A,ps}} = \sum_{i=1}^{M_{t,j}} \mathbf{1}\{ WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{ps}} \}
$$

其中，指示函數 $\mathbf{1}\{ \cdot \}$ 當 $WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{ps}}$ 時取值為 1，否則取值為 0。以同樣的方式亦可以根據負面情緒詞彙集合 $\mathrm{DICT}_{A,\mathrm{neg}}$ 計算：

$$
S_{t,j}^{\mathrm{A,neg}} = \sum_{i=1}^{M_{t,j}} \mathbf{1}\{ WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{neg}} \}
$$

最後我們可以定義時間點 $t$ 下的情緒分數為：

$$
S_{t}^{\mathrm{A}} = \sum_{j=1}^{N_t} \left( S_{t,j}^{\mathrm{A,ps}} - S_{t,j}^{\mathrm{A,neg}} \right)。
$$
