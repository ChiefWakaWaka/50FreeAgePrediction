from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import keras
import numpy as np

model = Sequential()
model.add(Dense(1, activation='relu', input_dim=1))
model.add(Dense(10, activation='softmax'))
model.add(Dense(25, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


import  pandas as pd
trainData = pd.read_csv('50FreeDataset.csv')
testData = pd.read_csv('testDataset.csv')

train_x = trainData['Time'].to_numpy()
train_y = trainData['Age'].to_numpy()
test_x = testData['Time'].to_numpy()
test_y = testData['Age'].to_numpy()

train_y = keras.utils.to_categorical(train_y)
test_y = keras.utils.to_categorical(test_y)

model.fit(train_x, train_y, epochs=50, batch_size=10, use_multiprocessing=True)

'''
score = model.evaluate(test_x, test_y, batch_size = 10)
predictions = keras.predict_classes(model, test_x)
print(predictions)
'''
