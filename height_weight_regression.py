import  numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r"C:\Users\Sana Kanwal\Downloads\Compressed\weight-height.csv")
print(data .head())
summary=data.describe()
print(summary.transpose())
print(data.isnull().sum())
data['feet']=data['Height']//12
data['inches']=data['Height']%12
data['FEET']=data['feet'].astype(int)
print(data['FEET'])
print(data['inches'])
data['weight_in_Kgs']=data['Weight']/2.2046
print(data['weight_in_Kgs'])
print('After Conversion','\n',data.head())

print(plt.style.available)
mpl.style.use(['ggplot'])
'''Data Visualization'''
sns.countplot(data['Gender'])
plt.show()
#sns.pairplot(data,hue='Gender',palette='coolwarm')
#plt.show()
plt.subplot(1,2,1)
plt.boxplot(data['feet'])
plt.subplot(1,2,2)
plt.boxplot(data['weight_in_Kgs'])
plt.show()
#plt.scatter(x=data['feet'],y=data['weight_in_Kgs'])
plt.show()
plt.subplot(1,2,1)
plt.hist(data['Height'])
#plt.hist(data['feet'])
plt.subplot(1,2,2)
plt.hist(data['Weight'])
#plt.hist(data['weight_in_Kgs'])
plt.show()
#sns.lmplot(x='weight_in_Kgs',y='height_in_feet',data=data,hue='Gender',markers=['o','*'])
#from sklearn
#sns.regplot(x=data['height_in_feet'],y=data['weight_in_Kgs'])
#plt.show()
corr=data[['feet','inches','weight_in_Kgs']].corr()
sns.heatmap(corr,annot=True)
sns.lmplot(x='Height',y='Weight',data=data,hue='Gender',fit_reg=False)
plt.show()
sns.lmplot(x='feet',y='weight_in_Kgs',data=data,hue='Gender',fit_reg=False)
plt.show()
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(data['Gender'])
data['Gender']=encoder.fit_transform(data['Gender'])
X=data[['Gender','FEET','inches']]
y=data['weight_in_Kgs']
print(X)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
scaler=StandardScaler()
scaled=(scaler.fit_transform(X))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
model=LinearRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
print('Complete')
new=model.predict([[0,4,11.0]])
print(new)
