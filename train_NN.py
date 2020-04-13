import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=64)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"[*] Test Accuracy: {test_acc} Test Loss: {test_loss}")

# Convert layer from logits to probabilities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Save model
probability_model.save("MNIST.h5")
print("[+] Saved model as: MNIST.h5")
