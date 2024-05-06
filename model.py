import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
import numpy as np

#1-pozitivno 0-negativno

min_max_scaler = preprocessing.MinMaxScaler()

def model(train_data, k):
    ind_atr = train_data.columns[3:-1]
    dep_atr = train_data.columns[-1]
    
    X = train_data[ind_atr]
    y = train_data[dep_atr]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)    
    
    X_minmax_train = min_max_scaler.fit_transform(X_train.values)
    X_minmax_test = min_max_scaler.transform(X_test.values)
    
    #metric="minkowski", p=2
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # knn.fit(X_train, y_train)
    knn.fit(X_minmax_train, y_train)

    #print(knn.score(X_test, y_test))
    
    # y_pred_test = knn.predict(X_test)
    y_pred_test = knn.predict(X_minmax_test)
    
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
    
    neighbors = np.arange(1, 21)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
  
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        # knn.fit(X_train, y_train)
        knn.fit(X_minmax_train, y_train)
        # train_accuracy[i] = knn.score(X_train, y_train)
        # test_accuracy[i] = knn.score(X_test, y_test)
        train_accuracy[i] = knn.score(X_minmax_train, y_train)
        test_accuracy[i] = knn.score(X_minmax_test, y_test)
        
    return knn, confusion_matrix, neighbors, train_accuracy, test_accuracy
    
def predict(knn, data):
    pred_minmax= min_max_scaler.transform(data)
    #prediction = knn.predict(data)
    prediction = knn.predict(pred_minmax)
    predict_proba= knn.predict_proba(pred_minmax)
    return prediction, predict_proba

train_data=pd.read_csv('train_data.csv')

knn_model, c_matrix, neighbors, train_accuracy, test_accuracy = model(train_data, 3)

#print(c_matrix, neighbors, train_accuracy, test_accuracy)
