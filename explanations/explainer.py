import numpy as np 
import lime
from lime.lime_text import LimeTextExplainer
from lime import lime_image
import shap
from keras.preprocessing.text import Tokenizer
import tensorflow as tf

def predict_func(x,model,max_length=100):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    _seq = tokenizer.texts_to_sequences(texts = x)
    _text_data = tf.keras.preprocessing.sequence.pad_sequences(_seq, maxlen=max_length)
    return model.predict(_text_data)

def Explanation(explainer,model,X,X_ref,dataSetType,task):
    """
    build local explanation for target model using SHAP and LIME explainers

    parameters:
    explainer(string): defines the type of explainer to be used, either SHAP or LIME
    model: target model
    X: instance for which explantion is generated
    X_ref: is used to generate random distribution from which the explainers can 
    datasetType: defines the type of underlying datasest
    task: defines the task of the model, either classification or regression

    returns
    explanation(SHAP value or LIME explanation)
    """
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
                
                
                lime_explainer_shap = shap.other.LimeTabular(model, X, mode = 'classification')
                lime_attribs = lime_explainer_shap.attributions(X,num_features=X.shape[1])
                exp_lime.append(lime_attribs)
                return exp_lime
            else:
                return None
        elif dataSetType == "Text":
            if explainer == "SHAP":
                exp = shap.DeepExplainer(model, X_ref)
                shap_val = exp.shap_values(X)

                return shap_val
            elif explainer == "LIME":
                class_names = ['negative', 'positive']
                explainer = LimeTextExplainer(class_names=class_names)
                pred_f = lambda x : predict_func(x,model)
                explanation = explainer.explain_instance(X, pred_f, num_features=100)
                return explanation
        elif dataSetType == "IMAGE":
            if explainer == "SHAP":
                background = X[np.random.choice(X.shape[0], 100, replace=False)]
                e = shap.DeepExplainer(model, background)
                shap_values = e.shap_values(X[1:5])
                # shap.image_plot(shap_values, X[1:5])

                return np.array(shap_values)
            elif explainer == "LIME":
                pred_fn1 = lambda images: model.predict(images)
                explainer = lime_image.LimeImageExplainer(random_state=42)

                explanation_val = []
                explanation=[]
                for i in range(4):
                    e = explainer.explain_instance(
                            X[10], 
                            pred_fn1)
                    explanation_val.append(e.segments)
                    explanation.append(e)

                return np.array(explanation_val)
        else:
            return None


   
def visualize1(X, shap_value,type_of_Vis,local_global):
    #TODO
    if local_global == "local":
        shap.initjs()
        if type_of_Vis == "force":
            shap.plots.force(shap_value)
        
    elif local_global == "global":
        if type_of_Vis == "plot bar":
            shap.plots.bar(shap_value)
        elif type_of_Vis == "heatmap":
            shap.plots.heatmap(shap_value)
        elif type_of_Vis == "summary plot":
            shap.summary_plot(shap_value, X)
        

