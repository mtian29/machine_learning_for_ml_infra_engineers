## text_classification_rnn
https://www.tensorflow.org/tutorials/text/text_classification_rnn

```
Bidirectional RNN 
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), ## output is 64*2 vector to represent this sequence

VOCAB_SIZE=1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))
encoded_example = encoder(example).numpy()
print(encoded_example.shape) ## (none, 998)
print(encoded_example[16])
print(example[16]) ## example 16 is the longest, and have length 998. This is why all encoded example has length 998 (paddled with 0)


# predict on a sample text with padding, ## With a long sentence of 2000 length, this won't affect the short senctence, so the result for the short sentence is the same
as without the*2000.

##padding = "the " * 2000 + "there" * 10
padding = "the " * 2000
predictions = model.predict(np.array([sample_text, padding])) ## the long padding sentence won't affect the result of sample_text
print(predictions[0])
print(predictions)

```
