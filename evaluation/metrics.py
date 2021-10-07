import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lime
import shap
import matplotlib as plot
import sklearn.metrics
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pp


def plot(x):
  pp.plot(x,'-o')
  pp.show()

def faithfulness_metric_reg(model, x, coefs, base):
    predt = model.predict(np.transpose(x.reshape(-1,1)))
    ar = np.argsort(coefs) 
    pred_ts = np.zeros(x.shape[0])
    diff = []
    for ind in np.nditer(ar):
        x_copy = x.copy()
        d = x_copy[ind]-base[ind]
        if d<0:
            diff.append(-1)
        else:
            diff.append(1)
        x_copy[ind] = base[ind]
        x_copy_ts = model.predict(np.transpose(x_copy.reshape(-1,1)))
        pred_ts[ind] = x_copy_ts
    
    return -np.corrcoef(np.array(diff)*coefs, pred_ts)[0,1]

def faithfulness_metric_new_reg(model, x, coefs, base):
    predt = model.predict(np.transpose(x.reshape(-1,1)))
    ar = np.argsort(coefs) 
    pred_ts = np.zeros(x.shape[0])
    diff = []
    for ind in np.nditer(ar):
        x_copy = x.copy()
        d = x_copy[ind]-base[ind]
        if d<0:
            diff.append(-1)
        else:
            diff.append(1)
        x_copy[ind] = base[ind]
        x_copy_ts = model.predict(np.transpose(x_copy.reshape(-1,1)))
        pred_ts[ind] = x_copy_ts - predt
    
    return -np.corrcoef(coefs, pred_ts)[0,1]
def faithfulness_metrics_cls1(model,x,coefs,base):
#     pred_class = np.argmax(model.predict_proba(np.transpose(x.reshape(-1,1))), axis=1)[0]
#     pred_prob_og = model.predict_proba(np.transpose(x.reshape(-1,1)))
#     ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
#     pred_probs = np.zeros(x.shape[0])
#     diff = []
#     for ind in np.nditer(ar):
#         x_copy = x.copy()
#         x_copy[ind] = base[ind]
        
#         x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
#         pred_probs[ind] = x_copy_pr[0][pred_class]
#         # print(pred_probs)
#         # print(pred_class)

#     return -np.corrcoef(coefs, pred_probs-pred_prob_og)[0,1]
    pred_class = np.argmax(model.predict_proba(x.reshape(1,-1)), axis=1)[0]
    p = np.amax(model.predict_proba(x.reshape(1,-1)))
#     print("og pred",p)
    #find indexs of coefficients in decreasing order of value
    ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
    pred_probs = np.zeros(x.shape[0])
    isPos = [-1 for i in range(len(ar))]
    diff = np.zeros(x.shape[0])
    for ind in np.nditer(ar):
        if coefs[ind]<0:
            isPos[ind] = -1
        else:
            isPos[ind] = 1
        x_copy = x.copy()
        x_copy[ind] = base[ind]
        x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
        pred_probs[ind] = x_copy_pr[0][pred_class]
#         print(pred_probs)
        diff[ind]=p - pred_probs[ind]
#     print(coefs, np.array(diff))

    return -np.corrcoef(coefs, -np.array(isPos)*np.array(diff))[0,1]
def faithfulness_metrics_cls(model,x,coefs,base):
    pred_class = np.argmax(model.predict_proba(np.transpose(x.reshape(-1,1))), axis=1)[0]
    ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
    pred_probs = np.zeros(x.shape[0])
    for ind in np.nditer(ar):
        x_copy = x.copy()
        x_copy[ind] = base[ind]
        x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
        pred_probs[ind] = x_copy_pr[0][pred_class]
        # print(pred_probs)
        # print(pred_class)

    return np.corrcoef(coefs, pred_probs)[0,1]

def fai_cls(model,x,coefs,base):
    pred_class = np.argmax(model.predict(x.reshape(1,-1)), axis=1)[0]
    ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
    pred_probs = np.zeros(x.shape[0])
    for ind in np.nditer(ar):
        x_copy = x.copy()
        x_copy[ind] = base[ind]
        x_copy_pr = model.predict(x_copy.reshape(1,-1))
        pred_probs[ind] = x_copy_pr[0][pred_class]
        # print(pred_probs)
        # print(pred_class)

    return -np.corrcoef(coefs, pred_probs)[0,1]

def monotonicity_metric_reg(model, x, coefs, base):
    predict_ = model.predict(np.transpose(x.reshape(-1,1)))
    x_copy = base.copy()
    ar = np.argsort(coefs)
    isPos = [False for i in range(len(ar))]
    pred_tss = np.zeros(x.shape[0])
    
    for ind in np.nditer(ar):
        if coefs[ind]<0:
            isPos[ind] = False
        else:
            isPos[ind] = True
        x_copy[ind] = x[ind]
        x_copy_pr = model.predict(np.transpose(x_copy.reshape(-1,1)))
        pred_tss[ind] = x_copy_pr
    diff = np.diff(pred_tss[ar])
    final_ = []
    for i in range(len(diff)):
        if isPos[i] == False and diff[i] < 0:
            final_.append(True)
        elif isPos[i] == True and diff[i] >=0:
            final_.append(True)
        else:
            final_.append(False)
    return any(final_)
def monotonicity_metric_cls(model, x, coefs, base):
    pred_class = np.argmax(model.predict_proba(x.reshape(1,-1)), axis=1)[0]
    x_copy = base.copy()
    #find indexs of coefficients in increasing order of value
    ar = np.argsort(coefs)
#     print(coefs)
    isPos = [False for i in range(len(ar))]
    pred_probs = np.zeros(x.shape[0])
    for ind in np.nditer(ar):
        if coefs[ind]<0:
            isPos[ind] = False
        else:
            isPos[ind] = True
        x_copy[ind] = x[ind]
        x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
        pred_probs[ind] = x_copy_pr[0][pred_class]
    diff = np.diff(pred_probs[ar])
    final_ = []
    for i in range(len(diff)):
        if isPos[i] == False and diff[i] < 0:
            final_.append(True)
        elif isPos[i] == True and diff[i] >=0:
            final_.append(True)
        else:
            final_.append(False)
    return any(final_)

def metrics_reg(model,X,shap_val,explainer_type,metrics_type,dataset):
    cols = X.columns
    if dataset == "boston":
      base1 = X[cols].mean()
      base1['ZN'] = int(round(X['ZN'].mean()))
      base1['CHAS'] = 0
      base1['RAD'] = int(round(X['RAD'].mean()))
      base = base1.values
    else:
      base = X[cols].mean().values
    if metrics_type == "faithfulness":
        faithfulness = []
        if explainer_type == "shap":
            for i in range(X.shape[0]):
                x = np.array(X.iloc[i,:])
                if type(shap_val) == np.ndarray:
                    coefs = shap_val[i]
                else:# print(shap_val[i])
                    coefs = shap_val[i].values
                # coefs = shap_val[i].values
                f = faithfulness_metric_new_reg(model, x, coefs, base)
                # print(f)
                faithfulness.append(f)
        elif explainer_type == "kernel shap":
            for i in range(X.shape[0]):
                x = np.array(X.iloc[i,:])
                if type(shap_val) == np.ndarray:
                    coefs = shap_val[i]
                else:# print(shap_val[i])
                    coefs = shap_val[i].values 
                # coefs = shap_val[i].values
                f = faithfulness_metric_new_reg(model, x, coefs, base)
                # print(f)
                faithfulness.append(f)
        elif explainer_type == "lime":
            for i in range(X.shape[0]):
                x = np.array(X.iloc[i,:])
                coefs = np.array(shap_val)[i]
                f = faithfulness_metric_new_reg(model, x, coefs, base)
                # print(f)
                faithfulness.append(f)
        plot(faithfulness)
        return np.array(faithfulness).mean()
    elif metrics_type == "monotonicity":
        monotonicity = []
        if explainer_type == "shap":
            for i in range(X.shape[0]):
                x = np.array(X.iloc[i,:])
                coefs = shap_val.values[i]
                f = monotonicity_metric_reg(model, x, coefs, base)
                monotonicity.append(f)
        elif explainer_type == "lime":
            for i in range(X.shape[0]):
                x = np.array(X.iloc[i,:])
                coefs = np.array(shap_val)[i]
                f = monotonicity_metric_reg(model, x, coefs, base)
                monotonicity.append(f)
        return monotonicity


def metrics_cls(model,X,shap_val,explainer_type,metrics_type,dataset):
   
    base = np.mean(X,axis=0)
    if metrics_type == "faithfulness":
        faithfulness = []
        if explainer_type == "shap":
            for i in range(X.shape[0]):
                x = X[i,:]
                if type(shap_val) == np.ndarray:
                    coefs = shap_val[i]
                else:# print(shap_val[i])
                    coefs = shap_val[i].values
                f = faithfulness_metrics_cls1(model, x, coefs, base)
                # print(f)
                faithfulness.append(f)
        elif explainer_type == "kernel shap":
            for i in range(X.shape[0]):
                x = X[i,:]
                # print(shap_val[i])
                coefs = shap_val[i]
                f = faithfulness_metrics_cls1(model, x, coefs, base)
                # print(f)
                faithfulness.append(f)
        elif explainer_type == "lime":
            for i in range(X.shape[0]):
                x = X[i,:]
                coefs = shap_val[i]
                f = faithfulness_metrics_cls1(model, x, coefs, base)
                # print(f)
                faithfulness.append(f)
        plot(faithfulness)
        return np.array(faithfulness).mean()
    elif metrics_type == "monotonicity":
        monotonicity = []
        if explainer_type == "shap":
            for i in range(X.shape[0]):
                x = X[i,:]
                if type(shap_val) == np.ndarray:
                    coefs = shap_val[i]
                else:# print(shap_val[i])
                    coefs = shap_val[i].values
                # coefs = shap_val.values[i]
                f = monotonicity_metric_cls(model, x, coefs, base)
                monotonicity.append(f)
        elif explainer_type == "lime":
            for i in range(X.shape[0]):
                x = X[i,:]
                if type(shap_val) == np.ndarray:
                    coefs = shap_val[i]
                else:# print(shap_val[i])
                    coefs = shap_val[i].values
                # coefs = shap_val[i]
                f = monotonicity_metric_cls(model, x, coefs, base)
                monotonicity.append(f)
        return monotonicity
def fai_cls_forText(model,x,coefs,base):
    pred_class = np.argmax(model.predict(x.reshape(1,-1)), axis=1)[0]
    ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
    pred_probs = np.zeros(x.shape[0])
    for ind in np.nditer(ar):
        x_copy = x.copy()
        x_copy[ind] = base[ind]
        x_copy_pr = model.predict(x_copy.reshape(1,-1))
        pred_probs[ind] = x_copy_pr[0][pred_class]
        # print(pred_probs)
        # print(pred_class)

    return -np.corrcoef(coefs, pred_probs)[0,1]

def monotonicity_metric_txt(model, x, coefs, base):
    pred_class = np.argmax(model.predict(x.reshape(1,-1)), axis=1)[0]

    x_copy = base.copy()
    ar = np.argsort(coefs)
    isPos = [False for i in range(len(ar))]
    pred_tss = np.zeros(x.shape[0])
    
    for ind in np.nditer(ar):
        if coefs[ind]<0:
            isPos[ind] = False
        else:
            isPos[ind] = True
        x_copy[ind] = x[ind]
        x_copy_pr = model.predict(x_copy.reshape(1,-1))
        pred_tss[ind] = x_copy_pr
    diff = np.diff(pred_tss[ar])
    final_ = []
    for i in range(len(diff)):
        if isPos[i] == False and diff[i] < 0:
            final_.append(True)
        elif isPos[i] == True and diff[i] >=0:
            final_.append(True)
        else:
            final_.append(False)
    return any(final_)
    