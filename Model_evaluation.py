import tensorflow as tf
from read_data import BSData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import F1Score
from multiprocessing import Pool
import os
import time


def SequencingData(x_data: pd.DataFrame, y_data: pd.DataFrame, seq_len: int = 10, name='') -> tuple:
    out_x = list()
    out_y = list()
    max_len = len(x_data) - seq_len
    step = 1 if seq_len < 1000 else int(seq_len / 100)
    for i in range(seq_len, max_len, step):
        if i % 100000 < step:
            print(f'{name}:{i}/{max_len}')
        out_x.append(x_data.iloc[i - seq_len:i:20][['Current', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'Dir']]
                     .__array__().reshape([-1, 8, 1]))
        # out_x.append(x_data.iloc[i - seq_len:i][['Current', 'Dir']])
        out_y.append(y_data[i - 1])
    return out_x, out_y


def BalAcc(true_val, pred_val, threshold=0.5):
    true_val, pred_val = true_val > threshold, pred_val > threshold
    TP = sum(true_val & pred_val)
    TN = sum(~true_val & ~pred_val)
    FN = sum(true_val & ~pred_val)
    FP = sum(~true_val & pred_val)
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)

    return (Sensitivity + Specificity)/2


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


def acc_score(true_val, pred_val, threshold=0.5):
    true_val, pred_val = true_val > threshold, pred_val > threshold
    TP = sum(true_val & pred_val)
    TN = sum(~true_val & ~pred_val)
    # FN = sum(true_val & ~pred_val)
    # FP = sum(~true_val & pred_val)

    if TP or TN:
        return (TP + TN) / len(true_val)
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
    return data.astype(np.float32)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    BAR_C = BSData("PWM500_BAR_CENTERED.txt")
    BAR_1L = BSData("PWM500_BAR_1CML.txt")
    BAR_2L = BSData("PWM500_BAR_2CML.txt")
    BAR_1R = BSData("PWM500_BAR_1CMR.txt")
    BAR_2R = BSData("PWM500_BAR_2CMR.txt")

    print("Reading data")
    BAR_C.read_data()
    BAR_1L.read_data()
    BAR_2L.read_data()
    BAR_1R.read_data()
    BAR_2R.read_data()

    print("Converting data")
    BAR_C = Normalization(pd.DataFrame(BAR_C.data))
    BAR_C_Y = np.zeros(BAR_C.shape[0], dtype=np.float32)

    BAR_1L = Normalization(pd.DataFrame(BAR_1L.data))
    BAR_1L_Y = np.ones(BAR_1L.shape[0], dtype=np.float32)

    BAR_2L = Normalization(pd.DataFrame(BAR_2L.data))
    BAR_2L_Y = np.ones(BAR_2L.shape[0], dtype=np.float32)

    BAR_1R = Normalization(pd.DataFrame(BAR_1R.data))
    BAR_1R_Y = np.ones(BAR_1R.shape[0], dtype=np.float32)

    BAR_2R = Normalization(pd.DataFrame(BAR_2R.data))
    BAR_2R_Y = np.ones(BAR_2R.shape[0], dtype=np.float32)

    X = list()
    y = list()
    temp = list()
    print("Sequencing data")
    seq_len = 5000
    with Pool(5) as p:
        for res in p.starmap(SequencingData,
                             [  # (NW, NW_Y, seq_len, 'NW'),
                                 #  (NW2, NW2_Y, seq_len, 'NW2'),
                                 (BAR_C, BAR_C_Y, seq_len, 'BAR_C'),
                                 (BAR_1L, BAR_1L_Y, seq_len, 'BAR_1L'),
                                 (BAR_2L, BAR_2L_Y, seq_len, 'BAR_2L'),
                                 (BAR_1R, BAR_1R_Y, seq_len, 'BAR_1R'),
                                 (BAR_2R, BAR_2R_Y, seq_len, 'BAR_2R')]):
            temp.append(res)

    del BAR_C, BAR_C_Y, BAR_1L, BAR_1L_Y, BAR_2L, BAR_2L_Y, BAR_1R, BAR_1R_Y, BAR_2R, BAR_2R_Y

    print("Appending")
    for tx, ty in temp:
        for dx, dy in zip(tx, ty):
            X.append(dx)
            y.append(dy)

    X, y = np.array(X), np.array(y)

    model_f1 = tf.keras.models.load_model('CNN_2D_binLR_f32_2.h5')

    model_f1.save_weights("f32_2.data")
    from CNN2D_model_binLR import CNN_model
    model = CNN_model(shape=(X.shape[1], X.shape[2], X.shape[3]))
    model.compile(optimizer='Adam',
                  loss=tf.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.load_weights("f32_2.data")

    start_time = time.time()
    y_pred = model.predict(X)
    end_time = time.time()

    y_pred = y_pred.reshape(-1)
    exec_time = end_time - start_time

    acc = acc_score(y, y_pred)
    f1result = F1_score(y, y_pred)
    balacc = BalAcc(y, y_pred)

    print(f"Accuracy: {acc*100}%")
    print(f"F1 score: {f1result}")
    print(f"Balanced accuracy: {balacc}")
    print(f"Execution time: {exec_time} s")
    print(f"{len(y)/exec_time} FPS")
    print(f"{len(y)} frames calculated")
    model.save('CNN_2D_binLR_f32_2_no_f1.h5')
