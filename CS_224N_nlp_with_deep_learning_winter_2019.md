https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/index.html#coursework
https://www.youtube.com/playlist?list=PL75e0qA87dlFJiNMeKltWImhQxfFwaxvv

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

## Machine Translation, seq2seq, Attention
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf

Machine translation is a major use case of sequence to sequence model (improved by attention)

- Statistical Machine Translation, SMT (1990 - 2010)

given x, find translation of x in language y.
`Find argmax p(y|x) = argmax p(x|y) * p(y)` -> get y

- alignment

Alignment is the correspondence between particular words in the
translated sentence pair.

- Neural Machine Translation (NMT) - better than SMT

The neural network architecture is called sequence-to-sequence (aka seq2seq) and it involves two RNNs. Decoder RNN (provides an vector of source sentence feed into encoder) and Encoder RNN (generate the target sentence).

- beam search decoding, beam search size = k

For each of the k hypotheses, find top k next words and calculate scores. Of these k^2 hypotheses, just keep k with highest scores

- bottleneck problem

Encoding (vector) of the source sentence. This needs to capture all information about the source sentence. Information bottleneck!

- Attention: solution for bottleneck problem. Focus on particular parts of the input

on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence.
Given a set of vector values, and a vector query, attention is a technique to compute a weighted sum of the values, dependent on the query.

## Transformer, Bert
### transformer: replace LSTM.
https://www.youtube.com/watch?v=TQQlZhbC5ps&ab_channel=CodeEmporium

https://www.tensorflow.org/tutorials/text/transformer

http://nlp.seas.harvard.edu/2018/04/03/attention.html

https://arxiv.org/pdf/1706.03762.pdf

- RNN is slow. Transformer training is fast, in parallel.
- All words in. Then the embedding for all words out from encoder
- Input word embedding + positional encoder -> the input for transformer.
- Encoders: (1) Attention vector for each word: which other words is more relavant to this word
- Decoders: predict the next word given the previous word in the output and input embedding. -> get the probably distribution.

### Bert
https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html

https://www.youtube.com/watch?v=xI0HHN5XKDo&ab_channel=CodeEmporium

https://github.com/google-research/bert
https://arxiv.org/abs/1810.04805
https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb

- Bidirectional Encoder Representations from Transformers
- Consider the full context of a word by looking at the words that come before and after it
- Just a stack of transformer encoders for language understanding 
- pretraining using unannotated text (input is 2 sentences). (1) masked language model (2) next sentence prediction 
- After pretraining, it generates an embedding for each word (considering contextual words as well)
- Fine tunning: for specific nlp tasks. -> supervised learning. e.g. Q&A 
- A “sequence” refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together
- tf-hub: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3
  - pooled_output for classification: returns [batch_size, hidden_size(768)]. --> dropout + full connected layer(num_labels node) + softmax --> use for classification
  - sequence_output returns [batch_size, sequence_length, hidden_size]
  - batch size: number of sequences (sentences)
  - sequence_length: max 128, == number of tokens in this sequence
  - hidden size: 768, the feature size of this token. (the embedding of this token)

### word pieces
Breaking word into smaller pieces. (i.e. "calling" -> ["call", "##ing"])

if these get segmented as walk@@ ing, walk@@ ed, etc., notice that all of them will now have walk@@ in common, which will occur much frequently while training, and the model might be able to learn more about it.

### Word embeddings
- static word embedding: one word -> one vector
Word2Vec, GloVe, 

- Contextualized (Dynamic) Word Embedding (word, word, word) -> (vector, vector, vector)
Meaning of a word depends on the words surrounding it. 
CoVe, ELMo, BERT

### Transfer learning
Transfer learning — a technique where instead of training a model from scratch, we use models pre-trained on a large dataset and then fine-tune them for specific natural language tasks.

## text generation
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture15-nlg.pdf

### Decoding algrorithms
- greedy decoding
- beam search
- sampling methods
- softmax temperature

### Sumarization
given input text x, write a summary y which is shorter and
contains the main information of x.

### Dialogue
