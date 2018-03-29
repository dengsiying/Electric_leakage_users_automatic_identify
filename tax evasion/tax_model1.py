#-*- coding: utf-8 -*-

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

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

datafile = 'Taxevasion identification.xls'

df = pd.read_excel(datafile)
#print(data.describe().T)
df_normal = df.iloc[:,3:16][df[u"输出"]=="正常"]
df_abnormal=df.iloc[:,3:16][df[u'输出']=='异常']

# df_normal.describe().T.to_excel('normal.xls')
# df_abnormal.describe().T.to_excel('abnormal.xls')

#数据预处理
df1 = pd.get_dummies(df[u'销售类型'],prefix='type')
df2 = pd.get_dummies(df[u'销售模式'],prefix='model')
res = pd.get_dummies(df[u'输出'],prefix='result')
df = pd.concat([df,df1,df2,res],axis=1)
df.drop([u'销售类型',u'销售模式',u'输出'],axis=1,inplace = True)
#正常列去除，异常列作为结果 1表示异常 0表示正常
df.drop([u'result_正常'],axis=1,inplace=True)
df.rename(columns={u'result_异常':'result'},inplace = True)

#决策树模型

data = df.as_matrix()
shuffle(data)

p = 0.8
train = data[:int(len(data)*p),:]
test = data[int(len(data)*p):,:]

tree =  DecisionTreeClassifier()
X = train[:,1:-1]
Y = train[:,-1]
tree.fit(X,Y)
#保存模型
joblib.dump(tree,'tree1.pkl')
#混淆矩阵
cm = confusion_matrix(Y,tree.predict(X))

plt.matshow(cm,cmap=plt.cm.Blues)
plt.colorbar()

for x in range(len(cm)): #数据标签
  for y in range(len(cm)):
    plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label') #坐标轴标签
plt.xlabel('Predicted label') #坐标轴标签
#plt.show() #显示作图结果

cm = confusion_matrix(test[:,-1],tree.predict(test[:,1:-1]))

plt.matshow(cm,cmap=plt.cm.Blues)
plt.colorbar()

for x in range(len(cm)): #数据标签
  for y in range(len(cm)):
    plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label') #坐标轴标签
plt.xlabel('Predicted label') #坐标轴标签
#plt.show()

#逻辑回归模型
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
lr.fit(X,Y)
#逻辑回归系数
w = pd.DataFrame({"columns":list(df.columns)[1:-1], "coef":list(lr.coef_.T)})
print(w)
cm = confusion_matrix(Y,lr.predict(X))

plt.matshow(cm,cmap=plt.cm.Blues)
plt.colorbar()

for x in range(len(cm)): #数据标签
  for y in range(len(cm)):
    plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label') #坐标轴标签
plt.xlabel('Predicted label') #坐标轴标签
plt.show() #显示作图结果

cm = confusion_matrix(test[:,-1],lr.predict(test[:,1:-1]))

plt.matshow(cm,cmap=plt.cm.Blues)
plt.colorbar()

for x in range(len(cm)): #数据标签
  for y in range(len(cm)):
    plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label') #坐标轴标签
plt.xlabel('Predicted label') #坐标轴标签
plt.show()

#两个模型的roc曲线
fig,ax=plt.subplots()
fpr, tpr, thresholds = roc_curve(test[:,-1], tree.predict_proba(test[:,1:-1])[:,1], pos_label=1)
fpr2, tpr2, thresholds2 = roc_curve(test[:,-1], lr.predict_proba(test[:,1:-1])[:,1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'blue') #作出ROC曲线
plt.plot(fpr2, tpr2, linewidth=2, label = 'ROC of LR', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果