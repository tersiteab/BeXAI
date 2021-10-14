import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR,SVC
import matplotlib as plot
import sklearn.metrics
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow as tf
tf.compat.v1.disable_v2_behavior() 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.datasets import imdb
tf.compat.v1.disable_v2_behavior()



def to_rgb(x):
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb

def get_dataset(rgb):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1,28,28,1)).astype('float32') 
    x_test = x_test.reshape((-1,28,28,1)).astype('float32') 
    if rgb == False:
        return x_train, y_train, x_test, y_test
    else:
        return to_rgb(x_train),y_train,to_rgb(x_test),y_test


def loadDataset(dataset):
    if dataset == "boston":
        boston = sklearn.datasets.load_boston()
        boston_df = pd.DataFrame(boston['data'] )
        boston_df.columns = boston['feature_names']
        boston_df['PRICE']= boston['target']

        y = boston_df['PRICE']
        X = boston_df.iloc[:,0:13]
        return X, y
    elif dataset == "superconductivity":
        df = pd.read_csv("/home/teadem/ExplainableAI/datasets/train.csv")
        y = df['critical_temp']
        X = df.drop('critical_temp',axis=1)

        return X,y
    elif dataset == "diabetes":
        diabetes = sklearn.datasets.load_diabetes()
        diabetes_df = pd.DataFrame(diabetes.data )
        diabetes_df.columns = diabetes.feature_names
        diabetes_df['target']= diabetes['target']

        y = diabetes_df['target']
        X = diabetes_df.drop('target',axis=1)
        return X, y
    elif dataset == "imdb":
        words=20000
        X,y = imdb.load_data(num_words=words)
        return X,y
    # elif dataset == ""
    elif dataset == "breast cancer":
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    
        return X, y
    elif dataset == "wine":
        wine = sklearn.datasets.load_wine()
        X = wine.data
        y = wine.target
        target_names = wine.target_names
        feature_names = wine.feature_names

        return X,y,target_names,feature_names
    else: 
        return None #TODO return error message


def split_train_test(X,y,test_size=0.3,randomstate=42,scale=False):
    if scale == False:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=randomstate)
        return X_train,X_test,y_train,y_test
    else:
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        y_scaled = sc.fit_transform(y)

        return train_test_split(X_scaled,y_scaled,test_size=test_size,random_state=randomstate)

def train(rgb):
  if rgb == True:
    i = 3
  else:
    i = 1
  model = keras.Sequential(
      [
      keras.Input(shape=(28,28,i)),
      layers.Conv2D(16, i, activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(10,activation='sigmoid')
      ]
  )
  return model

def RNN(x_train,y_train,x_test,y_test,words):
    
    word_size=words
    embed_size=128

    imdb_model=tf.keras.Sequential()
    # Embedding Layer
    imdb_model.add(tf.keras.layers.Embedding(word_size, embed_size, input_shape=(x_train.shape[1],)))
    # LSTM Layer
    imdb_model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))
    # Output Layer
    imdb_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    print(imdb_model.summary())
    imdb_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    imdb_model.fit(x_train, y_train, epochs=5, batch_size=128)
    test_loss, test_acurracy = imdb_model.evaluate(x_test, y_test)
    print("Test accuracy: {}".format(test_acurracy))
    return imdb_model


def train_model(model, X,y):
    if model == "Linear Regression":
        lr = LinearRegression()
        lr.fit(X,y)
        return lr
    elif model == "Random Forest Regressor":
        rf_r = RandomForestRegressor()
        rf_r.fit(X,y)
        return rf_r
    elif model == "SVR":
        svr = SVR(kernel = "rbf")
        svr.fit(X,y)
        return svr
    elif model == "RNN":
        words=20000
        max_length=100

        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=words)
        """Padding the Text"""
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
        return RNN(x_train,y_train,x_test,y_test,words)
    elif model == "Logistic Regression":
        lr = LogisticRegression(max_iter=10000)
        lr.fit(X,y)
        return lr
    elif model == "Random Forest Classifier":
        rf_c = RandomForestClassifier()
        rf_c.fit(X,y)
        return rf_c
    elif model == "SVC":
        svc = SVC(
            kernel = 'rbf',
            C=1,
            probability=True,
            gamma = 0.1,
            decision_function_shape='ovr',  # n_cls trained with data from one class as postive and remainder of data as neg
            random_state = 0,
        )
        svc.fit(X,y)
        return svc
    else:
        return None

