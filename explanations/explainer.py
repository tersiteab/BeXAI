import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lime
from lime.lime_text import LimeTextExplainer
from lime import lime_image
import shap
import matplotlib as plot
import sklearn.metrics
import sklearn.datasets
from sklearn.preprocessing import StandardScaler



def Explanation_reg(explainer,model,X,X_ref,dataSetType,task):
    if task == "Regression":
        if dataSetType == "tabular":
            if explainer == "SHAP":
                exp = shap.Explainer(model.predict, X_ref)
                shap_vals = exp(X)
                base_val = shap_vals.base_values 
                return shap_vals,base_val
            elif explainer == "Kernel SHAP":
                exp = shap.KernelExplainer(model.predict, X_ref)
                shap_vals = exp.shap_values(X)
                expected_val = exp.expected_value
                return shap_vals,expected_val
            elif explainer == "Tree SHAP":
                exp = shap.TreeExplainer(model,X_ref)
                shap_vals = exp(X)
                return shap_vals
            elif explainer == "LIME":
                X_np= X.to_numpy()
                featureNames=X.columns
                lime_exp = lime.lime_tabular.LimeTabularExplainer(
                    X_np,
                    feature_names=featureNames ,
                    class_names=['feature_names'], 
                    verbose=True, 
                    mode='regression') 
                exp_lime = []
                for x in X:
                    ex = lime_exp.explain_instance(x, 
                        model.predict, num_features=len(featureNames))
                    exp_lime.append(x)
                return exp_lime
            elif explainer == "LIME-SHAP":
                exp_lime = []
                if type(X) == np.ndarray:
                    X_np= X
                else:
                    X_np = X.to_numpy()
        #         featureNames=X.columns
                No_features = X.shape[1]
                
                lime_explainer_shap = shap.other.LimeTabular(model.predict,X_np,mode = 'regression')
                lime_attribs = lime_explainer_shap.attributions(X_np,num_features=No_features)
                return lime_attribs
            else:
                    return None
        else:
            return None
    elif task == "Classification":
        if dataSetType == "tabular":
            if explainer == "SHAP":
                exp = shap.Explainer(model.predict, X_ref)
                shap_vals = exp(X)
                base_val = shap_vals.base_values 
                return shap_vals,base_val
            elif explainer == "Kernel SHAP":
                exp = shap.KernelExplainer(model.predict, X_ref)
                shap_vals = exp.shap_values(X)
                expected_val = exp.expected_value
                return shap_vals,expected_val
            elif explainer == "Tree SHAP":
                exp = shap.TreeExplainer(model,X_ref)
                shap_vals = exp(X)
                return shap_vals
            elif explainer == "LIME":
                X_np= X.to_numpy()
                featureNames=X.columns
                lime_exp = lime.lime_tabular.LimeTabularExplainer(
                    X_np,
                    feature_names=featureNames ,
                    class_names=['feature_names'], 
                    verbose=True, 
                    mode='classification') 
                exp_lime = []
                for x in X:
                    ex = lime_exp.explain_instance(x, 
                        model.predict, num_features=len(featureNames))
                    exp_lime.append(x)
                return exp_lime
            elif explainer == "LIME-SHAP":
                exp_lime = []
                X_np= X.to_numpy()
                featureNames=X.columns
                No_features = len(featureNames)
                for x in X_np:
                    lime_explainer_shap = shap.other.LimeTabular(model.predict,x,mode = 'classification')
                    lime_attribs = lime_explainer_shap.attributions(x,num_features=No_features)
                    exp_lime.append(lime_attribs)
                return exp_lime
            else:
                return None
        elif dataSetType == "Text":
            if explainer == "SHAP":
                exp = shap.DeepExplainer(model, X_ref)
                shap_values = exp.shap_values(X)

                return exp,shap_values
            elif explainer == "LIME":
                class_names = ['negative', 'positive']
                explainer = LimeTextExplainer(class_names=class_names)
                #TODO X needs fixing
                explanation = explainer.explain_instance(X, model.predict, num_features=100)
                return explanation
        elif dataSetType == "IMAGE":
            if explainer == "SHAP":
                background = X[:100]
                e = shap.DeepExplainer(model, background)
                n_test_images = 10
                test_images = X[100:100+n_test_images]
                shap_values = e.shap_values(test_images)
                return e,shap_values
            # elif explainer == "LIME":
            #     expl = lime_image.LimeImageExplainer()
            #     explanation = expl.explain_instance(np.array(pill_transf(img)), 
            #                              batch_predict, # classification function
            #                              top_labels=5, 
            #                              hide_color=0, 
            #                              num_samples=1000)
        else:
            return None


def Explanation_cls(explainer,model,X,X_ref):
    
    if explainer == "SHAP":
        exp = shap.Explainer(model, X_ref)
        shap_vals = exp(X)
        base_val = shap_vals.base_values 
        return shap_vals,base_val
    elif explainer == "Kernel SHAP":
        exp = shap.KernelExplainer(model, X_ref)
        shap_vals = exp.shap_values(X)
        expected_val = exp.expected_value
        return shap_vals,expected_val
    elif explainer == "Tree SHAP":
        exp = shap.TreeExplainer(model,X_ref)
        shap_vals = exp(X)
        return shap_vals
    elif explainer == "LIME":
       
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            X,
            mode='classification') 
        exp_lime = []
        for x in X:
            ex = lime_exp.explain_instance(x, 
                model, num_features=x.shape[0])
            exp_lime.append(x)
        return exp_lime
    elif explainer == "LIME-SHAP":
        exp_lime = []
        
        lime_explainer_shap = shap.other.LimeTabular(model,X,mode = 'classification')
        lime_attribs = lime_explainer_shap.attributions(X,num_features=X.shape[1])
        return lime_attribs
    else:
        return None
'''lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_train_np,
    feature_names=columns ,
    class_names=['feature_names'], 
    verbose=True, 
    mode='regression')'''      
