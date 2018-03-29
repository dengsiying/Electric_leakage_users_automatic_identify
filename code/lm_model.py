#-*- coding: utf-8 -*-
#LM神经网络模型
import pandas as pd
from random import shuffle

from keras.models import Sequential
from keras.layers.core import Dense,Activation

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

datafile = '../data/model.xls'
data = pd.read_excel(datafile)
data = data.as_matrix()
shuffle(data)

p = 0.8
train = data[:int(len(data)*p),:]
test = data[int(len(data)*p):,:]

netfile = '../tmp/net.model'

net = Sequential()
net.add(Dense(input_dim = 3, units= 10))
net.add(Activation('relu'))
net.add(Dense(input_dim = 10, units = 1))
net.add((Activation('sigmoid')))
net.compile(loss='binary_crossentropy',optimizer='adam')

net.fit(train[:,:3],train[:,3],nb_epoch=1000,batch_size=1)
net.save_weights(netfile)
#预测结果
predict_result  = net.predict_classes(train[:,:3]).reshape(len(train))
#print(predict_result.shape) #(232,1)-->(232,)

#混淆矩阵
cm = confusion_matrix(train[:,3],predict_result)

plt.matshow(cm,cmap = plt.cm.Blues)
plt.colorbar()
for x in range(len(cm)):
    for y in range(len(cm)):
        plt.annotate(cm[x,y],xy=(x,y),horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#绘制ROC曲线
test_predict_result = net.predict(test[:,:3]).reshape(len(test))
fpr,tpr,thresholds = roc_curve(test[:,3],test_predict_result,pos_label=1)#fpr 假正率 tpr 真正率
plt.plot(fpr,tpr,linewidth=2,label='ROC of LM')
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果
