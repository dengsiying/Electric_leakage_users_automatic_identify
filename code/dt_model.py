#-*- coding: utf-8 -*-
#CART决策树模型
import pandas as pd
import random
from random import shuffle

from sklearn.tree import DecisionTreeClassifier
#保存模型
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import  roc_curve
random.seed(1)
datafile = '../data/model.xls'
treefile = '../tmp/tree.pkl' #模型输出名字
data = pd.read_excel(datafile)
data = data.as_matrix()
shuffle(data)

p = 0.8
train = data[:int(len(data)*p),:]
test = data[int(len(data)*p):,:]

tree =  DecisionTreeClassifier()
tree.fit(train[:,:3],train[:,3])
#print(tree.predict_proba(test[:,:3]))
#print(tree.predict_proba(test[:,:3])[:,1]) #array of shape = [n_samples, n_classes] 第一列为分类为0的概率，第二列为分类为1的概率
#保存模型
joblib.dump(tree,treefile)

#混淆矩阵
cm = confusion_matrix(train[:,3],tree.predict(train[:,:3]))

plt.matshow(cm,cmap=plt.cm.Blues)
plt.colorbar()

for x in range(len(cm)): #数据标签
  for y in range(len(cm)):
    plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label') #坐标轴标签
plt.xlabel('Predicted label') #坐标轴标签
plt.show() #显示作图结果

#绘制ROC曲线
fpr,tpr,threshoulds = roc_curve(test[:,3],tree.predict_proba(test[:,:3])[:,1],pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0,1.05)
plt.xlim(0,1.05)
plt.legend(loc=4) #左下角
plt.show()