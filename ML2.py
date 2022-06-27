#Dikshant buwa
import pandas as pd
df=pd.read_csv("C:/Users/CC-O79/Downloads/iris.csv")
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB(opt)
#from sklearn.naive_bayes import BernoulliNB(opt)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
rf=RandomForestClassifier(random_state=1)
logr=LogisticRegression(random_state=0)
GBC=GradientBoostingClassifier(n_estimators=10)
DTC=DecisionTreeClassifier(random_state=0)
SVM=svm.SVC()
nn=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
mnb=MultinomialNB()

X=df.drop('Species',axis=1)
Y=df['Species']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.3)
#print(X_train)
#print(X_test)
print(Y_train)
#print(Y_test)

train=logr.fit(X_train,Y_train)
Y_pred=logr.predict(X_test)
print("LR:")
print(accuracy_score(Y_test,Y_pred))
#print(precision_score(Y_test,Y_pred))

train=mnb.fit(X_train,Y_train)
Y_pred1=mnb.predict(X_test)
print("MNB:")
print(accuracy_score(Y_test,Y_pred1))

train=GBC.fit(X_train,Y_train)
Y_pred2=GBC.predict(X_test)
print("GBC:")
print(accuracy_score(Y_test,Y_pred2))


train=DTC.fit(X_train,Y_train)
Y_pred3=DTC.predict(X_test)
print("DTC:")
print(accuracy_score(Y_test,Y_pred3))

train=SVM.fit(X_train,Y_train)
Y_pred4=SVM.predict(X_test)
print("SVM:")
print(accuracy_score(Y_test,Y_pred4))

train=nn.fit(X_train,Y_train)
Y_pred5=nn.predict(X_test)
print("nn:")
print(accuracy_score(Y_test,Y_pred5))

train=rf.fit(X_train,Y_train)
Y_pred6=rf.predict(X_test)
print("rf:")
print(accuracy_score(Y_test,Y_pred6))


'''LR:
0.9777777777777777
MNB:
0.6
GBC:
0.9777777777777777
DTC:
0.9777777777777777
SVM:
0.9777777777777777
nn:
0.24444444444444444'''