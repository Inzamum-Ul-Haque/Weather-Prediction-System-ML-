#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics


#Importing Datasets
dataset = pd.read_csv("Boston_weather.csv")
dataset = dataset.replace('-',0.0)
X = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values
Y = dataset.iloc[:,-1].values
Y = Y.reshape(-1,1)  #change 1d list to 2d list


#Encoding dataset(because the machine learning algorithms won't work in the
#columns which are strings)
le1 = LabelEncoder()
Y = le1.fit_transform(Y)


#Feature Scaling(for plotting graph in certain range)
sc = StandardScaler()
X = sc.fit_transform(X)



#MACHINE LEARNING STEPS START
#Splitting dataset into training set and test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.6,random_state=0)


#Training Model
classifier = DecisionTreeClassifier()
classifier.fit(X_train,Y_train)


#Accuracy of the training(means how many % of data has been trained)
score = classifier.score(X_train,Y_train)
y_predict = classifier.predict(X_test)

#Accuracy of the model(if we want to increase the accuracy we can increase the
#n_estimators of the ramdom forest classifier which is the tree node for random
#forest classifier.But increasing the node will only take more time to run the
#algorithm but will not improve the accuracy much.)
accuracy = accuracy_score(Y_test,y_predict)


#Metrics of the model(Mean absolute error,Rms,Root mean squared error)
mae = metrics.mean_absolute_error(Y_test,y_predict)
mse = metrics.mean_squared_error(Y_test, y_predict)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_predict))

#Convert the data back to string
y_predict = le1.inverse_transform(y_predict)
Y_test = le1.inverse_transform(Y_test)
Y_test = Y_test.reshape(-1,1)
y_predict = y_predict.reshape(-1,1)

#Classification Report of the model
cs_report = classification_report(Y_test, y_predict)


#Map the values into dataframe as table
df = np.concatenate((Y_test,y_predict),axis=1)
dataframe = pd.DataFrame(df,columns=['Rain on Today','Prediction of Rain'])
dataframe.to_csv('Decision_Tree_Final.csv')


#Print the values of X and Y
print(dataframe)
print('Accuracy: ',accuracy*100)
print('Mean Absolute Error: ',mae*100)
print('Mean Squared Error: ',mse)
print('Root Mean Squared Error: ',rmse)
print(cs_report)


#Graph Plotting
data = pd.read_csv("Decision_Tree_Final.csv")
le2 = LabelEncoder()
le3 = LabelEncoder()
x = data["Rain on Today"]
y = data["Prediction of Rain"]
plt.plot(x,y,'o')
plt.xlabel('Rain on Today')
plt.ylabel('Prediction of Rain')

x = le2.fit_transform(x)
y = le3.fit_transform(y)
m,b = np.polyfit(x,y,1)
plt.plot(x, m*x + b)

plt.show()
