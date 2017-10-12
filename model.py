from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, Merge, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Conv2D(8, kernel_size=(5, 5), activation='tanh', input_shape=(60, 160, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), activation='elu', dilation_rate = (2, 3)))
model.add(Conv2D(64, (5, 5), activation='elu', dilation_rate = (2, 3)))
model.add(Conv2D(96, (5, 5), activation='elu', dilation_rate = 2))
model.add(BatchNormalization())
model.add(Conv2D(128, (5, 5), strides=(2, 3), activation='elu'))
model.add(Conv2D(256, (5, 5), strides=(2, 3),activation='elu'))
model.add(Conv2D(256, (5, 5), strides=(2, 2),activation='elu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'elu'))
model.add(Dense(256, activation = 'elu'))
model.add(Dense(128, activation = 'elu'))
model.add(Dense(96, activation = 'elu'))
model.add(Dense(32, activation = 'elu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer= Adam(lr = 0.00003), loss='mse', metrics=['accuracy'])
