# GRU


## Description
GRU（Gated Recurrent Unit）は、リカレントニューラルネットワーク（RNN）の一種であり、主に時系列データを扱うためのモデルである。GRUは、LSTM（Long Short-Term Memory）と同様に、シーケンスデータの長期的な依存関係をモデル化するために設計されるが、より単純な構造を持ち、パラメータの数が少ないという特徴がある。

GRUは、以下の構成要素からなる：

・更新ゲート（Update Gate）： ネットワークが前のタイムステップの情報をどれだけ重要視するかを制御
・リセットゲート（Reset Gate）： ネットワークが以前の情報をどれだけ忘れるかを制御

以下にGRUセルの図と更新式を示す。  

$$z_t = \sigma(W_z [x_t,h_{t-1}])$$

\bm{r}_t &amp;= \sigma (\bm{W}_{r}[\bm{x}_t, h_{t-1}]) 
\tilde{\bm{h}}_t &amp;= \tanh (\bm{W}_{h}[\bm{x}_{t}, \bm{r}_t \circ \bm{h}_{t-1}  ]) 
\bm{h}_t &amp;= (1 -\bm{z}_t) \circ \tilde{\bm{h}}_{t-1} + \bm{z}_t \circ \bm{h}_t



GRUは、LSTMと比較してメモリ消費量が少なく、計算量も少ないため、実装が比較的簡単であり、小さなデータセットや計算資源が限られている環境での利用に適している。
