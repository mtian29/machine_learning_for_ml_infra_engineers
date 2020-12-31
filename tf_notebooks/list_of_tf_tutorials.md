https://www.tensorflow.org/tutorials
# ML basics with keras 

## Basic text classification
Done.

## Text classification with TensorFlow Hub: Movie reviews
Maps from text to 50-dimensional embedding vectors.

```
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
```

## Save and load
https://www.tensorflow.org/tutorials/keras/save_and_load
```
train_images.shape
(1000, 784)

## it doesn't have a input layer. But use a input_shape
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size) ## 
    
!ls {checkpoint_dir}
```

## Introduction to the Keras Tuner
https://www.tensorflow.org/tutorials/keras/keras_tuner

# hub tutorials
https://www.tensorflow.org/hub/tutorials#text-related-tutorials

# Advanced text tutorials

## https://www.tensorflow.org/tutorials/text/word_embeddings
Done

## https://www.tensorflow.org/tutorials/text/text_classification_rnn
Done

# https://developers.google.com/machine-learning/recommendation
https://developers.google.com/machine-learning/guides/rules-of-ml

# Guides https://www.tensorflow.org/guide
