import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

data = pd.read_csv(r"C:\Users\Sana Kanwal\Downloads\binary1.csv")
print(data.head())
X=data['per%'].values.reshape(-1,1)
y=data['GPA'].values.reshape(-1,1)
#reshaping the x array into 2d array b/c sklearn not work on 1d array

scalar = StandardScaler()
scalar.fit(X,y)
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
regr = LinearRegression()
regr.fit(X_train,y_train)
#print('Intercept: \n',regr.intercept_)
#print('Cofficient:\n',regr.coef_)
y_pred=regr.predict(X_test)
score = regr.score(X_test,y_pred)

#cm=confusion_matrix(y_test,y_pred)
#print(cm)
##===== GUI=====#3
root= tk.Tk()
root.title("GPA Calculator")

canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.pack()



# New_Score label and input box
label1 = tk.Label(root, text='Enter input score:')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)
entry1.focus()



def values():
    global New_Score #our 1st input variable
    New_Score = float(entry1.get())

    # with sklearn
    Intercept_result = ('Intercept: ', regr.intercept_)
    label_Intercept = tk.Label(root, text=Intercept_result, justify='center')
    canvas1.create_window(260, 220, window=label_Intercept)

    # with sklearn
    Coefficients_result = ('Coefficients: ', regr.coef_)
    label_Coefficients = tk.Label(root, text=Coefficients_result, justify='center')
    canvas1.create_window(260, 240, window=label_Coefficients)
    # prediction of GPA
    Prediction_result  = ('Predicted GPA: ', regr.predict([[New_Score]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)




button1 = tk.Button(root, text='Predict GPA', command=values,
                    bg='orange')  # button to call the 'values' command above
canvas1.create_window(270, 150, window=button1)




# plot 1st scatter
figure = plt.Figure(figsize=(5, 4), dpi=100)
plt.scatter(X_test, y_pred,  color='gray',label='Scores')
plt.plot(X_test, y_pred, color='red', linewidth=2,label='GPA')
plt.title('Score v/s GPA')
plt.xlabel('Scores',fontsize=15,color='blue')
plt.ylabel('GPA',fontsize=15,color='blue')
plt.legend()
plt.show()
#ax = figure.add_subplot(111)
#ax.scatter(data['SAT'].astype(float), data['GPA'].astype(float), color='r')
#scatter = FigureCanvasTkAgg(figure, root)
#scatter.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
#ax.legend()
#ax.set_xlabel('SAT')
#ax.set_title('SAT Vs. GPA')
root.mainloop()
