# Artificial Neural Network

# Importing Library
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv(r"E:\NIT Python (ALL Task)\DEEP LEARNING\17th Jan. Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1].values
print(X)
print(Y)

#Encoding categorical data
#label Encoding the 'Gender' columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
print(X)

#One Hot Encoding the "Geography"columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
print(X)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)


# splitting the datset into the Training set And test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Part 2 - Building the ANN

# Initializing the ANN
ann=tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))  

#Adding output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


#Part 3- Training the ANN
 
#Compiling the ANN
ann.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train,Y_train,batch_size=32,epochs=100)

# Part 4-Making the Prediction and evaluating the model

#Prediction the test set results
Y_pred=ann.predict(X_test)
Y_pred=(Y_pred > 0.5)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))
              
      
#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(Y_test,Y_pred)
print(ac)























