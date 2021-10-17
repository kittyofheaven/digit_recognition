import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(training_data, training_labels), (test_data, test_labels) = mnist.load_data()
training_data, test_data = training_data / 255, test_data / 255

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # use softmax so it doesnt give strange output
])

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(training_data, training_labels, epochs=5)
model.evaluate(test_data, test_labels)

# test mode accuracy for new data
predictions = model.predict(test_data)
np.set_printoptions(suppress=True)
print(test_labels[0])
print(predictions[0])