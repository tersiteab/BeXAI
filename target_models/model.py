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

# import torch, torchvision
# from torchvision import datasets, transforms
# from torch import nn, optim
# from torch.nn import functional as F
# from torchviz import make_dot
import tensorflow as tf
from tensorflow.keras.datasets import imdb
tf.compat.v1.disable_v2_behavior()


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         # Convolution Layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 10, kernel_size=5),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             nn.Conv2d(10, 20, kernel_size=5),
#             nn.Dropout(),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#         )
        
#         # 
#         self.fc_layers = nn.Sequential(
#             nn.Linear(320, 50),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(50, 10),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(-1, 320)
#         x = self.fc_layers(x)
#         return x
# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output.log(), target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
            
            
#             test_loss += F.nll_loss(output.log(),
#                                     target).item()  
            
           
#             pred = output.max(1, keepdim=True)[1]  
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print(
#         '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset),
#             100. * correct / len(test_loader.dataset)))
    

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

# def CNN():
    model = Net()
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        'mnist_data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        'mnist_data',
        train=False,
        transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
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

