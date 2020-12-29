## Word embedding

https://www.tensorflow.org/tutorials/text/word_embeddings

- Explain this example.
- Use model.get_layer() to inspect the output for each layer.

```

For text or sequence problems, the Embedding layer takes a 2D tensor of integers, of shape (samples, sequence_length), where each entry is a sequence of integers. It can embed sequences of variable lengths. You could feed into the embedding layer above batches with shapes (32, 10) (batch of 32 sequences of length 10) or (64, 15) (batch of 64 sequences of length 15).

The returned tensor has one more axis than the input, the embedding vectors are aligned along the new last axis. Pass it a (2, 3) input batch and the output is (2, 3, N)
N = feature space = embedding demesion = each token/word has a embedding demension of N

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])


Model: "sequential"
Input, (None, a string). None -> batch size
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
text_vectorization (TextVect (None, 100)               0   

convert words to index in the vocab. Look up each word in the vocab for its index. 
I dont like this movie. I -> 10, dont -> index 89, ....
I dont like this movie -> [10, 89, 38, 11, 19, 0 , 0 ...]. Pad remaining to 0. 100 is max sequence length.

_________________________________________________________________
embedding (Embedding)        (None, 100, 16)           160000    

None - batch size
100 -- tokens in this sequence(sentence), sequence_length
16 - embedding size (represent a token/word into a 16-length vector)

I -> index 10 -> embedding for I, vector of 1*16
dont -> index 89 ->
... do this for 100 tokens.
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0       

take the averge of the 100 demension ( the sequence_length). So that we get (None,16), each sequence/sentence is a length 16 vector.  
_________________________________________________________________
dense (Dense)                (None, 16)                272       

16 nodes to 16 nodes layer
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        

16 nodes to 1 node layer -> this number represent good or bad review.
=================================================================
Total params: 160,289
Trainable params: 160,289
Non-trainable params: 0
_________________________________________________________________

```
