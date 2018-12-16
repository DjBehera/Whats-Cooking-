import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
#--------------DATA EXTRACTION FROM JSONS---------------------------------
s = open('train.json')
data = json.loads(s.read())

train_set = pd.DataFrame(data)

train_set = train_set.drop(['id'],axis = 1)

#DATA PREPROCESSING--------------

# CREATING A CORPUS FOR TFID VECTORIZER
corpus=[]
for i in range(len(train_set)):
    list = []
    for j in range(len(train_set['ingredients'][i])):
        list.append(train_set['ingredients'][i][j].replace(" ","-"))
    corpus.append(" ".join(list))
    
#Applying TFID and hot encoding to ylabels--------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cv = CountVectorizer(max_features = 1500)
cv = TfidfVectorizer(max_features = 1500,sublinear_tf=True, min_df = 5, norm = 'l2', encoding= 'latin')
X = cv.fit_transform(corpus).toarray()
y = train_set['cuisine']
y = pd.get_dummies(y,columns = ['cuisine'])

#---------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Applying ANN to the dataset
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu', input_dim = 1500))
classifier.add(Dropout(p = 0.5))
#Adding the second hidden layer
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.5))
# Adding the output layer
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 15)

y_pred = classifier.predict(X_test)


##--------convert hot encoded data to labeled data-------------------------------
y_pred_final = np.empty([len(y_pred),1]).astype(str)
y_test_final = np.empty([len(y_test),1]).astype(str)

cuisine = y_test.columns.values
for i in range(len(y_pred)):
    y_pred_final[i] = cuisine[y_pred[i].argmax()]
y_test_final = y_test.idxmax(axis = 1)


#------------------CONFUSION MATRIX VISUALIZATION using SEABORN---------------------
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_mat = confusion_matrix(y_test_final,y_pred_final)
plt.figure()
fig,ax = plt.subplots(figsize = (10,10))
sns.set_style('darkgrid')
sns.heatmap(conf_mat, annot=True, fmt = 'd',xticklabels = y_test.columns.values,yticklabels = y_test.columns.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



