from keras import regularizers
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

def conv4():
    inputs = Input(shape=(224, 224, 1))

    x = Conv2D(filters=16, kernel_size=(3,3), padding='valid')(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='valid')(x)
    
    x = Conv2D(filters=32, kernel_size=(3,3), padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='valid')(x)
    
    x = Conv2D(filters=64, kernel_size=(3,3), padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='valid')(x)

    x = Conv2D(filters=64, kernel_size=(3,3), padding='valid', name='last_conv')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='valid')(x)

    x = Flatten()(x)

    x = Dense(units=150, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(units=100, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
    x = Dropout(0.25)(x)

    outputs = Dense(units=2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
