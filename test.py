import os
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.utils import np_utils

model = ''

if os.path.exists('model.h5'):
    model = load_model('model.h5')
else:
    np.random.seed(42)
    # noinspection NonAsciiCharacters
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model = Sequential()

    model.add(Dense(800, input_dim=784, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))

    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

    print(model.summary())

    model.fit(x_train, y_train, batch_size=200, epochs=20, validation_split=0.2, verbose=2)

    model.save('model.h5')


scores = model.evaluate(x_test, y_test, verbose=0)
print("To4nost' raboti na testovyh dannih: %.2f%%" % (scores[1] * 100))

prediction = model.predict('test3.jpeg')
print("Na kartinke izobrazheno: %.2f%%" % prediction)