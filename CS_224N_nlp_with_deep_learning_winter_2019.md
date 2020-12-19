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
