import tensorflow as tf
from read_data import BSData
import pandas as pd
import numpy as np
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split


def DNN_model(shape=9):
    input = tf.keras.Input(shape=shape)
    # x = tf.keras.layers.Flatten()(input)
    x = tf.keras.layers.Dense(1024, activation='relu')(input)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='relu')(x)
    model = tf.keras.Model(input, x)
    return model


if __name__ == "__main__":
    NW = BSData("PWM500_NW.txt")
    NW2 = BSData("PWM500_NW2.txt")
    BAR_C = BSData("PWM500_BAR_CENTERED.txt")
    BAR_1L = BSData("PWM500_BAR_1CML.txt")
    BAR_2L = BSData("PWM500_BAR_2CML.txt")
    BAR_1R = BSData("PWM500_BAR_1CMR.txt")
    BAR_2R = BSData("PWM500_BAR_2CMR.txt")

    print("Reading data")
    NW.read_data()
    NW2.read_data()
    BAR_C.read_data()
    BAR_1L.read_data()
    BAR_2L.read_data()
    BAR_1R.read_data()
    BAR_2R.read_data()

    print("Converting data")
    NW = pd.DataFrame(NW.data)
    NW_Y = np.zeros(NW.shape[0])
    NW.drop('Time', axis=1, inplace=True)

    NW2 = pd.DataFrame(NW2.data)
    NW2_Y = np.zeros(NW2.shape[0])
    NW2.drop('Time', axis=1, inplace=True)

    BAR_C = pd.DataFrame(BAR_C.data)
    BAR_C_Y = np.zeros(BAR_C.shape[0])
    BAR_C.drop('Time', axis=1, inplace=True)

    BAR_1L = pd.DataFrame(BAR_1L.data)
    BAR_1L_Y = np.ones(BAR_1L.shape[0])
    BAR_1L.drop('Time', axis=1, inplace=True)

    BAR_2L = pd.DataFrame(BAR_2L.data)
    BAR_2L_Y = np.ones(BAR_2L.shape[0])
    BAR_2L.drop('Time', axis=1, inplace=True)

    BAR_1R = pd.DataFrame(BAR_1R.data)
    BAR_1R_Y = np.ones(BAR_1R.shape[0])
    BAR_1R.drop('Time', axis=1, inplace=True)

    BAR_2R = pd.DataFrame(BAR_2R.data)
    BAR_2R_Y = np.ones(BAR_2R.shape[0])
    BAR_2R.drop('Time', axis=1, inplace=True)

    print("Concatenating data")
    X = pd.concat([i for i in [NW, NW2, BAR_C, BAR_1L, BAR_2L, BAR_1R, BAR_2R]])
    y = pd.concat([pd.DataFrame(i) for i in [NW_Y, NW2_Y, BAR_C_Y, BAR_1L_Y, BAR_2L_Y, BAR_1R_Y, BAR_2R_Y]])

    print("Creating model")
    model = DNN_model()
    model.compile(optimizer='adam',
                  loss=tf.losses.binary_crossentropy,
                  metrics=['accuracy', F1Score(num_classes=1, threshold=0.5)])

    checkpoint_filepath = './tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True
    )

    print("Spliting sets")
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=10,
              batch_size=1024,
              callbacks=[model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)
    model.save('DNN_1D.h5')

    print("Predicting values")
    y_pred = model.predict(X_val).reshape(-1)

    metric = F1Score(num_classes=1, threshold=0.5)
    metric.update_state(y_val, y_pred)
    result = metric.result()
    print("F1Score for val set:")
    print(result.numpy())
