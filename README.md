# GRU

## Description
GRU（Gated Recurrent Unit）は、リカレントニューラルネットワーク（RNN）の一種であり、主に時系列データを扱うためのモデルである。GRUは、LSTM（Long Short-Term Memory）と同様に、シーケンスデータの長期的な依存関係をモデル化するために設計されるが、より単純な構造を持ち、パラメータの数が少ないという特徴がある。

GRUは、以下の構成要素からなる：

・更新ゲート（Update Gate）： ネットワークが前のタイムステップの情報をどれだけ重要視するかを制御　　
・リセットゲート（Reset Gate）： ネットワークが以前の情報をどれだけ忘れるかを制御

以下にGRUセルの図と更新式を示す。  
<p align="center">
<img src="https://github.com/rimao-uni/GRU/assets/117995370/500c72ad-873f-43d1-b854-3345385b700b" >
</p>

Source : [http://dprogrammer.org/rnn-lstm-gru](http://dprogrammer.org/rnn-lstm-gru)

$$z_t = \sigma(W_z [x_t,h_{t-1}])$$

$$r_t = \sigma(W_r[x_t, h_{t-1}])$$

$$h_t = \tanh(W_h[x_t,r_t \circ h_{t-1}])$$

$$h_t = (1 -z_t) \circ \tilde h_{t-1} + z_t \circ h_t$$


GRUは、LSTMと比較してメモリ消費量が少なく、計算量も少ないため、実装が比較的簡単であり、小さなデータセットや計算資源が限られている環境での利用に適している。

## Requirement
```
torch==1.9.0
pandas==1.3.3
matplotlib==3.4.3
```

## References
Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling : [Paper](https://arxiv.org/abs/1412.3555)
