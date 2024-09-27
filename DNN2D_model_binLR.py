import tensorflow as tf
from read_data import BSData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import os
import logging
from keras.callbacks import CSVLogger


def DNN_model(shape=(9,)):
    mult = 1
    for i in shape:
        mult *= i
    din = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Flatten()(din)
    x = tf.keras.layers.Dense(mult, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(mult, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(32, activation='linear')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(32, activation='linear')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(8, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(8, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(din, x)
    return model


def SequencingData(x_data: pd.DataFrame, y_data: pd.DataFrame, seq_len: int = 10, name='') -> tuple:
    out_x = list()
    out_y = list()
    max_len = len(x_data) - seq_len
    step = 1 if seq_len < 1000 else int(seq_len / 300)
    for i in range(seq_len, max_len, step):
        if i % 100000 < step:
            print(f'{name}:{i}/{max_len}')
        out_x.append(x_data.iloc[i - seq_len:i:20][['Current', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'Dir']]
                     .__array__().reshape([-1, 8, 1]))
        # out_x.append(x_data.iloc[i - seq_len:i][['Current', 'Dir']])
        out_y.append(y_data[i - 1])
    return out_x, out_y


def F1_score(true_val, pred_val, threshold=0.5):
    true_val, pred_val = true_val > threshold, pred_val > threshold
    TP = sum(true_val & pred_val)
    # TN = sum((~true_val - min(~true_val)) & (~pred_val - min(~pred_val)))
    FN = sum(true_val & ~pred_val)
    FP = sum(~true_val & pred_val)

    if TP:
        return (2 * TP) / (2 * TP + FP + FN)
    else:
        return 0


def Normalization(data: pd.DataFrame) -> pd.DataFrame:
    data.drop('Time', axis=1, inplace=True)
    data['Current'] = data['Current']
    data['AX'] = data['AX']
    data['AY'] = data['AY']
    data['AZ'] = data['AZ']
    data['GX'] = data['GX'] / 10
    data['GY'] = data['GY'] / 10
    data['GZ'] = data['GZ'] / 2
    data['Dir'] = data['Dir']
    data['TimeDiff'] = data['TimeDiff'] / 10
    return data.astype('float16')


if __name__ == "__main__":
    logging.basicConfig(filename='./DNN2D_model_binLR.log', level=logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    logging.info(f"{os.times()}")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    NW = BSData("PWM500_NW.txt")
    NW2 = BSData("PWM500_NW2.txt")
    BAR_C = BSData("PWM500_BAR_CENTERED.txt")
    BAR_1L = BSData("PWM500_BAR_1CML.txt")
    BAR_2L = BSData("PWM500_BAR_2CML.txt")
    BAR_1R = BSData("PWM500_BAR_1CMR.txt")
    BAR_2R = BSData("PWM500_BAR_2CMR.txt")

    print("Reading data")
    # NW.read_data()
    # NW2.read_data()
    BAR_C.read_data()
    BAR_1L.read_data()
    BAR_2L.read_data()
    BAR_1R.read_data()
    BAR_2R.read_data()

    print("Converting data")
    # NW = Normalization(pd.DataFrame(NW.data))
    # NW_Y = np.zeros(NW.shape[0], dtype=np.float16)
    #
    # NW2 = Normalization(pd.DataFrame(NW2.data))
    # NW2_Y = np.zeros(NW2.shape[0], dtype=np.float16)

    BAR_C = Normalization(pd.DataFrame(BAR_C.data))
    BAR_C_Y = np.zeros(BAR_C.shape[0], dtype=np.float16)

    BAR_1L = Normalization(pd.DataFrame(BAR_1L.data))
    BAR_1L_Y = np.ones(BAR_1L.shape[0], dtype=np.float16)

    BAR_2L = Normalization(pd.DataFrame(BAR_2L.data))
    BAR_2L_Y = np.ones(BAR_2L.shape[0], dtype=np.float16)

    BAR_1R = Normalization(pd.DataFrame(BAR_1R.data))
    BAR_1R_Y = np.ones(BAR_1R.shape[0], dtype=np.float16)

    BAR_2R = Normalization(pd.DataFrame(BAR_2R.data))
    BAR_2R_Y = np.ones(BAR_2R.shape[0], dtype=np.float16)

    X = list()
    y = list()
    temp = list()
    print("Sequencing data")
    seq_len = 5000
    with Pool(5) as p:
        for res in p.starmap(SequencingData,
                             [# (NW, NW_Y, seq_len, 'NW'),
                             #  (NW2, NW2_Y, seq_len, 'NW2'),
                              (BAR_C, BAR_C_Y, seq_len, 'BAR_C'),
                              (BAR_1L, BAR_1L_Y, seq_len, 'BAR_1L'),
                              (BAR_2L, BAR_2L_Y, seq_len, 'BAR_2L'),
                              (BAR_1R, BAR_1R_Y, seq_len, 'BAR_1R'),
                              (BAR_2R, BAR_2R_Y, seq_len, 'BAR_2R')]):
            temp.append(res)

    # del NW, NW_Y, NW2, NW2_Y
    del BAR_C, BAR_C_Y, BAR_1L, BAR_1L_Y, BAR_2L, BAR_2L_Y, BAR_1R, BAR_1R_Y, BAR_2R, BAR_2R_Y

    print("Appending")
    for tx, ty in temp:
        for dx, dy in zip(tx, ty):
            X.append(dx)
            y.append(dy)

    X, y = np.array(X), np.array(y)
    logging.info("Data preprocessed")
    print("Spliting sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=256)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=256)

    print("Creating model")
    model = DNN_model(shape=(X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer='Adam',
                  loss=tf.losses.binary_crossentropy,
                  metrics=['accuracy', F1Score(num_classes=1)])
    # model = tf.keras.models.load_model('DNN_2D_binLR.h5')

    stringlist = []
    model.summary()
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    logging.info(short_model_summary)

    checkpoint_filepath = './tmp3/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    # model.load_weights(checkpoint_filepath)
    csv_logger = CSVLogger('./DNN2D_model_binLR_epochs.log', append=True, separator=';')
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=50,
              batch_size=2048,
              callbacks=[model_checkpoint_callback, csv_logger])

    model.load_weights(checkpoint_filepath)
    model.save('DNN_2D_binLR.h5')

    print("Predicting values")
    y_pred = model.predict(X_val).reshape(-1)

    # logging.info(f"MAE: {tf.losses.mae(y_val, y_pred)} MSE: {tf.losses.mse(y_val, y_pred)}")
    # print(f"MAE: {tf.losses.mae(y_val, y_pred)} MSE: {tf.losses.mse(y_val, y_pred)}")

    result = F1_score(y_val, y_pred)
    print("F1Score for val set:")
    print(result)

    plt.scatter(range(len(y_val)), y_val)
    plt.scatter(range(len(y_pred)), y_pred)
    plt.show()
