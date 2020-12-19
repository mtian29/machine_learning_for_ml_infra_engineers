https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/index.html#coursework

## Recurrent Neural Networks and Language Models
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture06-rnnlm.pdf

- language model (LM): a system that predicts the next word
- n gram: a chunk of n consecutive words.
- 5 gram based on conditional probalistic model
- why rnn for LM? 
take any input length. Apply same weight on each step. Can optionally produce output on each step
- how to train rnn? 
Get a large text, feed into RNN-LM, compute output distribution for every step, calcuate loss between predicted distribution vs true 1 hot vector

## Vanishing Gradients and Fancy RNNs
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture07-fancy-rnn.pdf

Vanishing gradients in RNN --> How to solve them (LSTM) 
RNN variants (bidirectional, multi-layer RNN)
- vanishing gradient

chain rule, the gradient signal gets smaller and smaller as it backpropagates further
- why it is a problem in RNN?

Gradient signal from faraway is lost because it’s much smaller than gradient signal from close-by. 
it’s too difficult for the RNN to learn to preserve information over many timesteps
- solution for graident expoding

gradient clipping 
- LSTM, long short term memory -> resilient to vanishing gradients

if the forget gate is set to remember everything on every timestep, then the info in the cell is preserved indefinitely
Popular in 2013 - 2015
- GRU (faster): an alternative to LSTM
- vanishing gradient is a problem for very deep nn.

Due to chain rule / choice of nonlinearity function, gradient can become
vanishingly small as it backpropagates
Thus lower layers are learnt very slowly (hard to train)
Solutions: resnet (skip connections), dense connections (directly connect every layer to layer)
- Bidirectional RNNs

bidirectional RNNs are only applicable if you have access to the entire input sequence.
Use bidirectional when possible
Cancatenated hidden states from forword rnn hidden states with backword rnn.
BERT (Bidirectional Encoder Representations from Transformers)
- Multi-layer RNNs

Transformer-based networks (e.g. BERT) can be up to 24 layers 
Multi-layer RNNs are powerful, but you might need skip/dense-connections if it’s deep
