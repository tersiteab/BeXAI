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
def pred_func(images,model):
    if images.shape[2] == 1:
        return model.predict(images.reshape((1,28,28,1)))
    elif images.shape[2] == 3:
        return model.predict(images.reshape(1,28,28,3))

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


def faithfulness_metrics_image_cls(model,X,coefs,base):
    
    pred_class = np.amax(pred_func(X,model))
    pred_probs = []
    shap_list = []
    
    # plot(X)
    for i in range(1,5):
      for j in range(1,5):
        #compute average shap_val for superpixel of size (4x4)
        avg_shap = np.average(coefs[7*(i-1) :7*i ,7*(j-1) :7*j])

        shap_list.append(avg_shap)                      
        x_copy = X.copy()
        x_copy[7*(i-1) :7*i ,7*(j-1) :7*j,0] = base[7*(i-1) :7*i ,7*(j-1) :7*j,0]
        
        # plot(x_copy)
        
        x_copy_pr = np.amax(pred_func(x_copy,model))
        pred_probs.append(x_copy_pr-pred_class)
    return -np.corrcoef(np.array(shap_list), np.array(pred_probs))[0,1]


def monotonicity(model,X,coefs,base):
    pred_class = np.amax(pred_func(X,model))
    shap_list = []
    # print("original")
    # plot(X)
    x_copy = base.copy()

    for i in range(1,8):
      for j in range(1,8):
        #compute average shap_val for superpixel of size (4x4)
        avg_shap = np.average(coefs[4*(i-1) :4*i ,4*(j-1) :4*j])
        shap_list.append(avg_shap) 

    ar = np.argsort(np.array(shap_list))
    pred_probs = np.zeros(np.array(shap_list).shape)#= []
    isPos = [False for i in range(len(ar))]

    for i in ar :  
      if shap_list[i]<0:
        isPos[i] = False
      else:
        isPos[i] = True
      r = i // 4
      c = i % 4
      x_copy[4*r :4*(r+1) ,4*c :4*(c+4),0] = X[4*r :4*(r+1) ,4*c :4*(c+4),0]
      # print("ablated")
      # plot(x_copy)
      x_copy_pr = np.amax(pred_func(x_copy,model))
      pred_probs[i]=x_copy_pr

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
    pred_class = np.argmax(model.predict_proba(x.reshape(1,-1)), axis=1)[0]
    p = np.amax(model.predict_proba(x.reshape(1,-1)))

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
        diff[ind]=p - pred_probs[ind]

    return -np.corrcoef(coefs, -np.array(isPos)*np.array(diff))[0,1]


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
                else:
                    coefs = shap_val[i].values
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
def metrics_image(metrics_type,X,model,explainer_type,shap_val,rgb=True):
    idx=[]
    for i in range(5):
        pred= pred_func(X[i],model)[0]
        max = np.amax(pred)
        _idx = np.where(pred == max)[0]
        idx.append(_idx[0])
    idx = np.array(idx) 
    base = np.zeros(X[0].shape) 
    if metrics_type == "faithfulness":
        if explainer_type == "SHAP":
            fidelity_shap = []
            for i in range(4):
                __idx = idx[i]
                coefs = shap_val[__idx,i,:,:,0]
                x = X[i]
                fidelity_shap.append(faithfulness_metrics_image_cls(model,x,coefs,base))
            return fidelity_shap
        elif explainer_type == "LIME":
            fidelity_lime = []
            for i in range(4):
                __idx = idx[i]
                coefs = shap_val[i]
                x = X[i]
                fidelity_lime.append(faithfulness_metrics_image_cls(model,x,coefs,base))
            return fidelity_lime
    elif metrics_type == "monotonicity":
        if explainer_type == "SHAP":
            mono_SHAP = []
            for i in range(4):
                __idx = idx[i]
                coefs = shap_val[__idx,i,:,:,0]
                x = X[i]
                mono_SHAP.append(monotonicity(model,x,coefs,base))
            return mono_SHAP
        elif explainer_type == "LIME":
            mono_LIME = []
            for i in range(4):
                __idx = idx[i]
                coefs = shap_val[i]
                x = X[i]
                # print(faithfulness_metrics_cls(model,X,coefs,base))
                mono_LIME.append(monotonicity(model,x,coefs,base))
            return mono_LIME
    
def metrics_text(metrics_type,X,model,explainer_type,shap_val):
    base = np.zeros(shape=(100))
    if metrics_type == "faithfulness":
        if explainer_type == "SHAP":
            f = []
            for i in range(10):
                x = X[i]
                coefs = shap_val[0][i]
                f.append(fai_cls_forText(model,x,coefs,base))
            return f
        #elif explainer_type == "LIME":
    elif metrics_type == "monotonicity":
        if explainer_type == "SHAP":
            mono_text = []
            for i in range(10):
                x = X[i]
                coefs = shap_val[0][i]
                mono_text.append(monotonicity_metric_txt(model,x,coefs,base))
            return mono_text