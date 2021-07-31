# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import neuron

import seaborn as sns
from sklearn.metrics import confusion_matrix


if __name__=='__main__':
    data = pd.read_excel("d.xlsx")
    data = data[['Q1_性别', 'Q2_身高（厘米）','Q3_体重（斤）','Q4_学号','比例']]
    scaled_data = data.copy()
    scaled_data['Q1_性别'] = (scaled_data['Q1_性别']=='女') * 1
    scaled_data['Q3_体重（斤）'] = data['Q3_体重（斤）'] - data['Q3_体重（斤）'].mean()
    scaled_data['Q2_身高（厘米）'] = data['Q2_身高（厘米）'] - data['Q2_身高（厘米）'].mean()
    scaled_data['比例'] = data['比例'] - data['比例'].mean()

    training_set = scaled_data[(scaled_data.index+1)%2==1]
    testing_set = scaled_data[(scaled_data.index+1)%2==0]

    data = training_set[['Q3_体重（斤）','Q2_身高（厘米）','比例']].values
    all_y_trues = training_set[['Q1_性别']].values
    
    # Train our neural network!
    network = neuron.OurNeuralNetwork()
    print("L1N1w1：%f,L1N1w2：%f,L1N1b：%f,L1N2w1：%f,L1N2w2：%f,L1N2b：%f,L1N3w1：%f,L1N3b：%f,L2N1w1：%f,L2N1w2：%f,L2N1w3：%f,L2N1b：%f" % (network.L1N1w1,network.L1N1w2,network.L1N1b,network.L1N2w1,network.L1N2w2,network.L1N2b,network.L1N3w1,network.L1N3b,network.L2N1w1,network.L2N1w2,network.L2N1w3,network.L2N1b))
    network.train(data, all_y_trues)

    # Make some predictions
    pred_values = []
    for i, d in testing_set.iterrows():
        gender = "女" if d['Q1_性别']==1 else "男"
        obseration = d[['Q3_体重（斤）','Q2_身高（厘米）','比例']].values
        pred = network.feedforward(obseration)
        pred_values.append(pred)
        gender_pred = "女" if pred>=0.5 else "男"
        print("%s | 实际性别:%s,%.3f,预测性别:%s,%s" % (d['Q4_学号'], gender, pred, gender_pred,"" if gender==gender_pred else "xxxxxxxxxxxxxx错误！"))

    # Confusion Matrix
    sns.set()
    y_true = testing_set['Q1_性别'].values
    y_pred = np.array(pred_values)
    cm = confusion_matrix(y_true, y_pred>0.5, labels=[0, 1])
    print(cm) 
    i=cm[0][0]+cm[1][1]
    l=len(testing_set)
    rate=i/l
    print("准确率：%f" % rate)
    print("L1N1w1：%f,L1N1w2：%f,L1N1b：%f,L1N2w1：%f,L1N2w2：%f,L1N2b：%f,L1N3w1：%f,L1N3b：%f,L2N1w1：%f,L2N1w2：%f,L2N1w3：%f,L2N1b：%f" % (network.L1N1w1,network.L1N1w2,network.L1N1b,network.L1N2w1,network.L1N2w2,network.L1N2b,network.L1N3w1,network.L1N3b,network.L2N1w1,network.L2N1w2,network.L2N1w3,network.L2N1b))
    sns.heatmap(cm, annot=True)
