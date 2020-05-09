import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.preprocessing import MinMaxScaler
'''****Phase#01 Data Preparation***'''
data=pd.read_csv(r"C:\Users\Sana Kanwal\Downloads\Compressed\diabetes.csv")
print(data)
print(data.isnull().sum())
print(data.columns)

# Seprating independent and dependent variables
# X=data.drop('Outcome',axis=1)
# y=data['Outcome']
#***Feature Scaling in feature scaling we are changing the range of the data
# scalar=MinMaxScaler(feature_range=(0,1))
# rescaled_X=scalar.fit_transform(X)
# np.set_printoptions(precision=3)  # Setting the precision for output
# print('Min Max Standard Scalar Data  ',str(rescaled_X))

#*** Standardizing the attributes in Gussain distribution with Mean=0 and SD=1\
# from sklearn.preprocessing import StandardScaler
# scalar=StandardScaler().fit(X)
# XX=scalar.transform(X)
# print('Standard Scalar Data  ',str(XX))

# *** Normalizing in this we resclae each observation to a length of  1 unit form and changing the shape of the data
# from sklearn.preprocessing import Normalizer
# scalar=Normalizer().fit(X)
# rescaled_X=scalar.fit_transform(X)
# print(' Normalize Data  ',str(rescaled_X))

# Shape of the dataseet
print(data.shape)
# Grouped the data
print(' Grouped Data ',str(data.groupby('Outcome').size()))

'''*******************Phase2 02 Visualization**********************'''
'''====Count PLot====='''
sns.countplot(x = 'Outcome',data = data)

'''Histogram:group data into bins and give us an idea of how many observations each bin holds,
'''
import itertools
col = data.columns[:7]
plt.subplots(figsize = (20, 15))
length = len(col)
for i, j in itertools.zip_longest(col, range(length)):
    plt.subplot((length/2), 3, j + 1)
    plt.subplots_adjust(wspace = 0.1,hspace = 0.5)
    data[i].hist(bins = 20)
    plt.title(i)
plt.show()
''' It is also for Histogram but with different color'''
f, axes = plt.subplots(2,4, figsize=(40,20), sharex=True)
sns.distplot( data["Glucose"] , color="skyblue",ax=axes[0, 0],bins=20)
sns.distplot( data["BloodPressure"] , color="olive",ax=axes[0, 1],bins=20)
sns.distplot( data["SkinThickness"] , color="gold",ax=axes[0, 2],bins=20)
sns.distplot( data["BMI"] , color="teal",ax=axes[0, 3],bins=20)
sns.distplot( data["DiabetesPedigreeFunction"] , color="green",ax=axes[1, 0],bins=20)
sns.distplot( data["Age"] , color="cyan",ax=axes[1, 1],bins=20)
plt.title('Histogram of All Variables')
plt.show()

'''===Density PLot===='''
data.plot(kind='density',subplots=True,sharex=False,figsize=(20,10),color='yellow')
plt.title('Density Plot')
plt.show()

'''Box and Whisker plot summarize how each attribute is distributed 
Whiskers tell us how the data is spread, and the dots outside the whiskers give candidate outlier values. 
'''
data.plot(kind='box',subplots=True,sharex=False,sharey=False,figsize=(20,10))
plt.title('Box and Whisker Plot')
plt.show()
'''
Multivariate plots 
'''
correlations=data.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
sns.heatmap(correlations,annot=True,cmap='RdYlGn')
ticks=np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()
'''Scatter Plot how much one variable is affected by another variable to build a relation out of it. 
'''
'''****Phase#3 DATA CLEANSING****'''
'''Observation: 1. Count plot tells that dataset is imbalanced as the number of patient have diabetes is more than which don't have!
                2. By observing Blood Pressure histogram there are some zero values and  readings 
                of the data set seem wrong because a living person cannot have a diastolic blood pressure of zero.
                 By observing the data we can see 35 counts where the value is 
                3. Glusoce levels would not be as low as zero. Therefore zero is an invalid reading. By observing the data we can see 5 counts where the value is 0.
                4. For normal people, skin fold thickness can’t be less than 10 mm better yet zero. Total count where value is 0: 227.
                5. BMI: Should not be 0 or close to zero unless the person is really underweight which could be life-threatening.
                6. Insulin: In a rare situation a person can have zero insulin but by observing the data, we can find that there is a total of 374 counts.
'''
'''***Observation Implementation****
'''
print("Total : ", data[data.BloodPressure == 0].shape[0])
print(data[data.BloodPressure == 0].groupby('Outcome')['Age'].count())

print("Total : ", data[data.Glucose == 0].shape[0])
print(data[data.Glucose == 0].groupby('Outcome')['Age'].count())

print("Total : ", data[data.SkinThickness == 0].shape[0])
print(data[data.SkinThickness == 0].groupby('Outcome')['Age'].count())

print("Total : ", data[data.BMI == 0].shape[0])
print(data[data.BMI == 0].groupby('Outcome')['Age'].count())

print("Total : ", data[data.Insulin == 0].shape[0])
Total :  374
print(data[data.Insulin == 0].groupby('Outcome')['Age'].count())

'''Remove the rows which have zero vlaues in BloodPresssure, BMI and Glusoce columns '''
diabetes_mod = data[(data.BloodPressure != 0) & (data.BMI != 0) & (data.Glucose != 0)]
print(diabetes_mod.shape)
'''********Phase#03 Feature Engineering**** “ 
Feature engineering enables us to highlight the important features 
and facilitate to bring domain expertise on the problem to the table. 
It also allows avoiding overfitting the model despite providing many input features”.
    '''
feature_names = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome
print(X)
''' ****Phase#04 Model Selection***'''
# Importing  diffrenet classifers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))

'''Feature Scaling'''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(data)
dataset_scaled = pd.DataFrame(data)
X=dataset_scaled.drop('Outcome',axis=1)
y=dataset_scaled['Outcome']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)
'''KFold Cross Validation  split the dataset into equal portions (folds) then use 1 fold for testing and union of other folds as training '''

names = []
scores = []
for name, model in models:
    kfold = KFold(n_splits=10,random_state=None,shuffle=False)
    score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()

    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

'''PLotting Accuracy of Classiication Algorithm'''
plt.title('Accuaracies of different Classifiers')
axis = sns.barplot(x='Name', y='Score', data=kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width() / 2, height + 0.005, '{:1.4f}'.format(height), ha="center")

plt.show()
print(X_train)
y_new=model.predict([[120,63,25,1,29.6,0.667,25]])
print(y_new)
y_new1=model.predict([[148,72,35,0,33.6,0.627,50]])
print('Recognizing: if zero mean no diabetes, if one mean patient have diabetes',y_new1)

'''GUI'''
import tkinter as tk
from tkinter import *
from PIL import ImageTk,Image
from tkinter import messagebox
root= tk.Tk()
root.title("Diabetes Recognizer")
canvas1 = tk.Canvas(root, width = 900, height = 500)
root.geometry("500x250")
image=PhotoImage(file='C://Users//Sana Kanwal//Pictures//dia.png')
#====Variabels======
var_glu=IntVar()
var_Bld_pr=IntVar()
var_skin=IntVar()
var_Insulin=IntVar()
var_BMI=DoubleVar()
var_db_pd=DoubleVar()
var_age=IntVar()
root.title("Diabetes Recognizer")
img_lbl=Label(root,image=image).pack()
title=Label(root,text='DIABETES RECOGNIZER',font=('times new roman',30,'bold'),bg='grey',bd=10,relief=GROOVE)
title.place(x=0,y=0,relwidth=1)
frame=Frame(root,bg='white')
frame.place(x=500,y=80)
label1 = tk.Label(frame, text='Glucose',compound=LEFT,font=('times new roman',12,'bold'),bg='white',fg='blue').grid(row=1,column=0,padx=20,pady=10)
txtlbl1=Entry(frame,bd=5,textvariable=var_glu,relief=GROOVE,font=('','13')).grid(row=1,column=1,padx=10)
label2 = tk.Label(frame, text='Blood_Pressure',compound=LEFT,font=('times new roman',12,'bold'),bg='white',fg='blue').grid(row=2,column=0,padx=20,pady=10)
txtlbl1=Entry(frame,bd=5,textvariable=var_Bld_pr,relief=GROOVE,font=('','13')).grid(row=2,column=1,padx=10)

label3 = tk.Label(frame, text='Skin_Thickness',compound=LEFT,font=('times new roman',12,'bold'),bg='white',fg='blue').grid(row=3,column=0,padx=20,pady=10)
txtlbl1=Entry(frame,bd=5,textvariable=var_skin,relief=GROOVE,font=('','13')).grid(row=3,column=1,padx=10)

label4 = tk.Label(frame, text='Insulin',compound=LEFT,font=('times new roman',12,'bold'),bg='white',fg='blue').grid(row=4,column=0,padx=20,pady=10)
txtlbl1=Entry(frame,bd=5,textvariable=var_Insulin,relief=GROOVE,font=('','13')).grid(row=4,column=1,padx=10)

label5 = tk.Label(frame, text='BMI',compound=LEFT,font=('times new roman',12,'bold'),bg='white',fg='blue').grid(row=5,column=0,padx=20,pady=10)
txtlbl1=Entry(frame,bd=5,textvariable=var_BMI,relief=GROOVE,font=('','13')).grid(row=5,column=1,padx=10)

label6 = tk.Label(frame, text='Db_pd',compound=LEFT,font=('times new roman',12,'bold'),bg='white',fg='blue').grid(row=6,column=0,padx=20,pady=10)
txtlbl1=Entry(frame,bd=5,textvariable=var_db_pd,relief=GROOVE,font=('','13')).grid(row=6,column=1,padx=10)

label7 = tk.Label(frame, text='Age',compound=LEFT,font=('times new roman',12,'bold'),bg='white',fg='blue').grid(row=7,column=0,padx=20,pady=10)
txtlbl1=Entry(frame,bd=5,textvariable=var_age,relief=GROOVE,font=('','13')).grid(row=7,column=1,padx=10)

def func():
    var_glu.get()
    var_Bld_pr.get()
    var_skin.get()
    var_Insulin.get()
    var_BMI.get()
    var_db_pd.get()
    var_age.get()
    result = model.predict([[var_glu.get(),var_Bld_pr.get(),var_skin.get(),var_Insulin.get(),var_BMI.get(),var_db_pd.get(),var_age.get()]])
    label_Prediction = tk.Label(root, text=result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
    if result==[0]:
        messagebox.showerror('Output'," Since result is  negatice means Patient is fine & don't have diabetes")
    else:
        messagebox.showerror('Output','Patient have diabetes')
btn=Button(frame,text='Recognize',width=14,font=('times new roman',13,'bold'),bg='white',fg='blue',command=func).grid(row=8,column=1,pady=10)
root.mainloop()
