from explanations.explainer import Explanation,Explanation_cls
from target_models.model import loadDataset,train_model,train_test_split,get_dataset,train
from evaluation.metrics import metrics_cls,metrics_reg,fai_cls_forText,monotonicity_metric_txt,pred_func,faithfulness_metrics_cls,monotonicity
import shap
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
import tensorflow as tf
from tensorflow.keras.datasets import imdb
tf.compat.v1.disable_v2_behavior()
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries


def Main_reg(dataset):
    X,y = loadDataset(dataset)
    sc = StandardScaler()
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    X_scaled = sc.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    if dataset == "diabetes":
        X_trainS,X_testS,y_trainS,y_testS = X_train.to_numpy(),X_test.to_numpy(),y_train.to_numpy(),y_test.to_numpy()
    else:
        X_trainS,X_testS,y_trainS,y_testS = train_test_split(X_scaled,y)

    print("--------------------------------------------------------------")
    print("Training Logistic regression, Random Forest classifier and SVM classifier ...")
    LR_model = train_model("Linear Regression",X_train,y_train)
    RF_model = train_model("Random Forest Regressor",X_train,y_train)
    SVR_model = train_model("SVR",X_trainS,y_trainS)
    print("Done Training")
    print("--------------------------------------------------------------")
    print("Model Evaluation")
    print("")
    print("Linear Regression")
    print("R2 for Train", LR_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', LR_model.score(X_test, y_test))
   
    print("Random Forest Regression")
    print('R2 for Train', RF_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', RF_model.score(X_test, y_test))
   
    print("Random Forest Regression")
    print('R2 for Train', SVR_model.score( X_trainS, y_trainS))
    # print('R2 for Test (cross validation)', r2_score(y_testS, sc.inverse_transform(SVC_model.predict(X_testS))))

    X100 = shap.maskers.Independent(X, max_samples=100)
    X100_ = shap.utils.sample(X, 100)
    X100S = shap.maskers.Independent(X_trainS, max_samples=100)
    X100_S = shap.utils.sample(X_trainS, 100)
    idx = 10
    print("--------------------------------------------------------------")
    print("Building Explanation ...")
    LR_shap,LR_baseVal = Explanation("SHAP",LR_model,X_test.iloc[:idx,],X100,"tabular","Regression")
    LR_shap_k,RF_expected_val_k = Explanation("Kernel SHAP",LR_model,X_test.iloc[:idx,],X100_,"tabular","Regression")
    # LR_lime1 = Explanation("LIME",LR_model,X_test,X100)
    LR_lime = Explanation("LIME-SHAP",LR_model,X_test.iloc[:idx,],X100,"tabular","Regression")

    RF_shap, RF_baseVal = Explanation("SHAP",RF_model,X_test.iloc[:idx,],X100,"tabular","Regression")
    RF_shap_k,RF_expected_val_k = Explanation("Kernel SHAP",RF_model,X_test.iloc[:idx,],X100_,"tabular","Regression")
    # RF_lime1 = Explanation("LIME",RF_model,X_test,X100)
    RF_lime = Explanation("LIME-SHAP",RF_model,X_test.iloc[:idx],X100,"tabular","Regression")

    X100S = shap.maskers.Independent(X_trainS, max_samples=100)
    X100_S = shap.utils.sample(X_trainS, 100)
    print("SVC-shap")
    SVR_shap, SVR_baseVal = Explanation("SHAP",SVR_model,X_testS[:idx,],X100S,"tabular","Regression")
    print("SVC kernel shap")
    SVR_shap_k,SVR_expected_val_k = Explanation("Kernel SHAP",SVR_model,X_testS[:idx,],X100_S,"tabular","Regression")
    print("svc lime")
    SVR_lime = Explanation("LIME-SHAP",SVR_model,X_testS[:idx,],X100S,"tabular","Regression")
    print("Done building Explanation")


    faithfulness_LR_shap = metrics_reg(model=LR_model,X=X_test.iloc[:idx,],shap_val=LR_shap,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for shap Linear reg:",np.mean(np.array(faithfulness_LR_shap)))
    faithfulness_LR_shap_k = metrics_reg(model=LR_model,X=X_test.iloc[:idx,],shap_val=LR_shap_k,explainer_type="kernel shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for kernel shap Linear reg:",np.mean(np.array(faithfulness_LR_shap_k)))
    faithfulnes_LR_lime = metrics_reg(model=LR_model,X=X_test.iloc[:idx,],shap_val=LR_lime[:idx,],explainer_type="lime",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for shap RF reg:",np.mean(np.array(faithfulnes_LR_lime)))
    faithfulness_RF_shap = metrics_reg(model=RF_model,X=X_test.iloc[:idx,],shap_val=RF_shap,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for shap RF reg:",np.mean(np.array(faithfulness_RF_shap)))
    faithfulness_RF_shap_k = metrics_reg(model=RF_model,X=X_test.iloc[:idx,],shap_val=RF_shap_k,explainer_type="kernel shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for kernel shap RF reg:",np.mean(np.array(faithfulness_RF_shap_k)))
    faithfulnes_RF_lime = metrics_reg(model=RF_model,X=X_test.iloc[:idx,],shap_val=RF_lime[:idx,],explainer_type="lime",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for shap RF reg:",np.mean(np.array(faithfulnes_RF_lime)))
    
    xS = pd.DataFrame(X_testS,columns = X.columns)

    faithfulness_SVR_shap = metrics_reg(model=SVR_model,X=xS.iloc[:idx,],shap_val=SVR_shap,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for shap SVR reg:",np.mean(np.array(faithfulness_SVR_shap)))
    faithfulness_SVR_shap_k = metrics_reg(model=SVR_model,X=xS.iloc[:idx,],shap_val=SVR_shap_k,explainer_type="kernel shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for kernel shap SVR reg:",np.mean(np.array(faithfulness_SVR_shap_k)))
    faithfulnes_SVR_lime = metrics_reg(model=SVR_model,X=xS.iloc[:idx,],shap_val=SVR_lime[:idx,],explainer_type="lime",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for shap SVR reg:",np.mean(np.array(faithfulnes_SVR_lime)))
    monotonicity_LR_shap = metrics_reg(model=LR_model,X=X_test.iloc[:idx,],shap_val=LR_shap,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    print(all(monotonicity_LR_shap))
    monotonicity_LR_shap_k = metrics_reg(model=LR_model,X=X_test.iloc[:idx,],shap_val=LR_shap_k,explainer_type="kernel shap",metrics_type="monotonicity",dataset=dataset)
    print(all(monotonicity_LR_shap_k))
    monotonicity_LR_lime = metrics_reg(model=LR_model,X=X_test.iloc[:idx,],shap_val=LR_lime,explainer_type="lime",metrics_type="monotonicity",dataset=dataset)
    print(all(monotonicity_LR_lime))
    monotonicity_RF_shap = metrics_reg(model=RF_model,X=X_test.iloc[:idx,],shap_val=RF_shap,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    print(all(monotonicity_RF_shap))
    monotonicity_RF_shap_k = metrics_reg(model=RF_model,X=X_test.iloc[:idx,],shap_val=RF_shap_k,explainer_type="kernel shap",metrics_type="monotonicity",dataset=dataset)
    print(all(monotonicity_RF_shap_k))
    monotonicity_RF_lime = metrics_reg(model=RF_model,X=X_test.iloc[:idx,],shap_val=RF_lime,explainer_type="lime",metrics_type="monotonicity",dataset=dataset)
    print(all(monotonicity_RF_lime))
    monotonicity_SVR_shap = metrics_reg(model=SVR_model,X=X_test.iloc[:idx,],shap_val=SVR_shap,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    print(all(monotonicity_SVR_shap))
    monotonicity_SVR_shap_k = metrics_reg(model=SVR_model,X=X_test.iloc[:idx,],shap_val=SVR_shap_k,explainer_type="kernel shap",metrics_type="monotonicity",dataset=dataset)
    print(all(monotonicity_SVR_shap_k))
    monotonicity_SVR_lime = metrics_reg(model=SVR_model,X=X_test.iloc[:idx,],shap_val=SVR_lime,explainer_type="lime",metrics_type="monotonicity",dataset=dataset)
    print(all(monotonicity_SVR_lime))
    return faithfulness_RF_shap,monotonicity_LR_shap,monotonicity_RF_shap,faithfulness_LR_shap


def Main_cls(dataset):
    print(dataset)
    if dataset == "wine":
        X,y,target_names,feature_names = loadDataset(dataset)
    else:
        X,y = loadDataset(dataset)
    
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    X_trainS,X_testS,y_trainS,y_testS = train_test_split(X_scaled,y)

    print("--------------------------------------------------------------")
    print("Training Logistic regression, Random Forest classifier and SVM classifier ...")
    LR_model = train_model("Logistic Regression",X_train,y_train)
    RF_model = train_model("Random Forest Classifier",X_train,y_train)
    SVC_model = train_model("SVC",X_trainS,y_trainS)
    print("Done Training")
    print("--------------------------------------------------------------")
    print("Model Evaluation")
    print("Logistic regression")
    print('R2 for Train', RF_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', RF_model.score(X_test, y_test))
    print("Random forest classifier")
    print('R2 for Train: ', LR_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', LR_model.score(X_test, y_test))
    print("SVM classifier")
    print('R2 for Train)', SVC_model.score( X_trainS, y_trainS))
    
    predict_fnLR = lambda x:LR_model.predict_proba(x)[:,1]
    predict_fnRF = lambda x:RF_model.predict_proba(x)[:,1]
    predict_fnSVC = SVC_model.decision_function 

    X100 = shap.maskers.Independent(X, max_samples=100)
    X100_ = shap.utils.sample(X, 100)
    
    print("--------------------------------------------------------------")
    print("Building Explanation ...")
    LR_shap,LR_baseVal = Explanation("SHAP",predict_fnLR,X_test[:10,],X100,"tabular","Classification")
    LR_shap_k,LR_expected_val_k = Explanation("Kernel SHAP",predict_fnLR,X_test[:10,],X100_,"tabular","Classification")
    LR_lime1 = Explanation("LIME",LR_model.predict_proba,X_test[:10],X100,"tabular","Classification")
    
    RF_shap, RF_baseVal = Explanation("SHAP",predict_fnRF,X_test[:10,],X100,"tabular","Classification")
    RF_shap_k,RF_expected_val_k = Explanation("Kernel SHAP",predict_fnRF,X_test[:10,],X100_,"tabular","Classification")
    RF_lime1 = Explanation("LIME",RF_model.predict_proba,X_test[:10,],X100,"tabular","Classification")
    print("SVC-shap")
    SVC_shap, SVC_baseVal = Explanation("SHAP",predict_fnSVC,X_testS[:10,],X100,"tabular","Classification")
    print("SVC-shap_k")
    SVC_shap_k,SVC_expected_val_k = Explanation("Kernel SHAP",predict_fnSVC,X_testS[:10,],X100_,"tabular","Classification")
    print("SVC-shap")
    SVC_lime1 = Explanation("LIME",SVC_model.predict_proba,X_testS[:10,],X100,"tabular","Classification")
    
    print("Done building Explanation")
    ################### evaluation#####################
    
    #faithfulness
    faithfulness_LR_shap= metrics_cls(model=LR_model,X=X_test[:10,],shap_val=LR_shap,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for shap Logistic reg:",np.mean(np.array(faithfulness_LR_shap)))
    faithfulness_LR_shap_k = metrics_cls(model=LR_model,X=X_test[:10,],shap_val=LR_shap_k,explainer_type="kernel shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for kernel shap Logistic reg:",np.mean(np.array(faithfulness_LR_shap_k)))
    faithfulness_LR_lime = metrics_cls(model=LR_model,X=X_test[:10,],shap_val=LR_lime1,explainer_type="lime",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for lime Logistic Reg:",np.mean(np.array(faithfulness_LR_lime)))
    faithfulness_RF_shap = metrics_cls(model=RF_model,X=X_test[:10,],shap_val=RF_shap,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for shap RF Classification:",np.mean(np.array(faithfulness_RF_shap)))
    faithfulness_RF_shap_k = metrics_cls(model=RF_model,X=X_test[:10,],shap_val=RF_shap_k,explainer_type="kernel shap",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for kernel shap RF Classification:",np.mean(np.array(faithfulness_RF_shap_k)))
    faithfulness_RF_lime = metrics_cls(model=LR_model,X=X_test[:10,],shap_val=RF_lime1,explainer_type="lime",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for lime RF Classification:",np.mean(np.array(faithfulness_RF_lime)))
    #TODO svc DONE
    # bit of modification to pass attribution val of the predicted class for wine dataset
    if dataset == "wine":
        pred_class1 = np.argmax(SVC_model.predict_proba(X_testS[:10,]), axis=1)
        x = X_testS[:10,]
        s=SVC_shap.values[:10,]
        o = []
        # for i in range(x.shape[0]):
        #     p = pred_class1[i]
        #     o.append(s[i,:,p])
        # o_ = np.array(o)
        # faithfulness_SVC_shap= metrics_cls(model=SVC_model,X=X_testS[:10,],shap_val=o_,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
        # print("Mean Faithfulness for SHAP SVM Classification:",np.mean(np.array(faithfulness_SVC_shap)))
        #kernel shap
        # pred_class = np.argmax(SVC_model.predict_proba(X_testS[:10,]), axis=1)
        # x = X_testS[:10,]
        # sk=np.array(SVC_shap_k)[:10,]
        # ok = []
        # for i in range(x.shape[0]):
        #     p = pred_class[i]
        #     ok.append(sk[p,i,:])
        # o_k= np.array(ok)
        # faithfulness_SVC_k_shap= metrics_cls(model=SVC_model,X=X_testS[:10,],shap_val=o_k,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
        # print("Mean Faithfulness for kernel SHAP SVM Classification:",np.mean(np.array(faithfulness_SVC_k_shap)))
    else:
        faithfulness_SVC_shap = metrics_cls(model=SVC_model,X=X_test[:10,],shap_val=SVC_shap,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
        print("Mean Faithfulness for shap SVC Classification:",np.mean(np.array(faithfulness_SVC_shap)))
        faithfulness_SVC_shap_k = metrics_cls(model=SVC_model,X=X_test[:10,],shap_val=SVC_shap_k,explainer_type="kernel shap",metrics_type="faithfulness",dataset=dataset)
        print("Mean Faithfulness for kernel shap SVC Classification:",np.mean(np.array(faithfulness_SVC_shap_k)))
    
    #LIME
    faithfulness_SVC_lime= metrics_cls(model=SVC_model,X=X_testS[:10,],shap_val=np.array(SVC_lime1),explainer_type="lime",metrics_type="faithfulness",dataset=dataset)
    print("Mean Faithfulness for lime SVM Classification:",np.mean(np.array(faithfulness_SVC_lime)))
    monotonicity_LR_shap = metrics_cls(model=LR_model,X=X_test[:10,],shap_val=LR_shap,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    monotonicity_LR_shap_k = metrics_cls(model=LR_model,X=X_test[:10,],shap_val=LR_shap_k ,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)

    print(type(LR_lime1))
    monotonicity_LR_lime = metrics_cls(model=LR_model,X=X_test[:10,],shap_val=np.array(LR_lime1),explainer_type="lime",metrics_type="monotonicity",dataset=dataset)
    monotonicity_RF_shap = metrics_cls(model=RF_model,X=X_test[:10,],shap_val=RF_shap,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    monotonicity_RF_shap_k = metrics_cls(model=RF_model,X=X_test[:10,],shap_val=RF_shap_k,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    monotonicity_RF_lime = metrics_cls(model=RF_model,X=X_test[:10,],shap_val=np.array(RF_lime1),explainer_type="lime",metrics_type="monotonicity",dataset=dataset)

    monotonicity_SVC_lime = metrics_cls(model=SVC_model,X=X_testS[:10,],shap_val=np.array(SVC_lime1),explainer_type="lime",metrics_type="monotonicity",dataset=dataset)
    pred_class1 = np.argmax(SVC_model.predict_proba(X_testS[:10,]), axis=1)
    x = X_testS[:10,]
    s=SVC_shap.values[:10,]
    # o = []
    # for i in range(x.shape[0]):
    #     p = pred_class1[i]
    #     o.append(s[i,:,p])
    # o_ = np.array(o)



    pred_class = np.argmax(SVC_model.predict_proba(X_testS[:10,]), axis=1)
    # x = X_testS[:10,]
    # sk=np.array(SVC_shap_k)[:10,]
    # ok = []
    # for i in range(x.shape[0]):
    #     p = pred_class[i]
    #     ok.append(sk[p,i,:])
    # o_k= np.array(ok)


    # monotonicity_SVC_shap = metrics_cls(model=SVC_model,X=X_testS[:10,],shap_val=o_,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    # monotonicity_SVC_shap_k = metrics_cls(model=SVC_model,X=X_testS[:10,],shap_val=o_k,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    print("monotonicity in form of boolean for LR_shap:",monotonicity_LR_shap)
    print("monotonicity in form of boolean for LR_shap_k:",monotonicity_LR_shap_k)
    print("monotonicity in form of boolean for LR_lime:",monotonicity_LR_lime)
    print("monotonicity in form of boolean for RF_shap:",monotonicity_RF_shap)
    print("monotonicity in form of boolean for RF_shap_k:",monotonicity_RF_shap_k)
    print("monotonicity in form of boolean for RF_lime:",monotonicity_RF_lime)
    # print("monotonicity in form of boolean for SVC_shap:",monotonicity_SVC_shap)
    # print("monotonicity in form of boolean for SVC_shap_k:",monotonicity_SVC_shap_k)
    print("monotonicity in form of boolean for SVC_lime:",monotonicity_SVC_lime)

def Main_text():
    X,y = loadDataset("imdb")
    x_train,y_train,x_test,y_test = train_test_split(X,y)   
    imdb_model = train_model(model= "RNN",X =  x_train,y = y_train)
    
    # we use the first 100 training examples as our background dataset to integrate over
    print("------------ buildin Explanation with Deepshap explainer----------------------")
    from tensorflow.keras.datasets import imdb
    tf.compat.v1.disable_v2_behavior()
    explainer = shap.DeepExplainer(imdb_model, x_train[:100])

    # explain the first 10 predictions
    # explaining each prediction requires 2 * background dataset size runs
    shap_values = explainer.shap_values(x_test[:20])

    # init the JS visualization code
    shap.initjs()

    # transform the indexes to words
    
    words = imdb.get_word_index()
    num2word = {}
    for w in words.keys():
        num2word[words[w]] = w
    x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), x_test[i]))) for i in range(10)])

    # plot the explanation of the first prediction
    # Note the model is "multi-output" because it is rank-2 but only has one column
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test_words[0])
    print("---Done building Explanation---")
    print("Faithfulness metrics")
    model = imdb_model
    base = np.zeros(shape=(100))
    f = []
    for i in range(20):
        x = x_test[i]
        coefs = shap_values[0][i]
        f.append(fai_cls_forText(model,x,coefs,base))

    pp.plot(f,'-o')
    pp.show()
    print("average faithfulness of shap text explainer for RNN based text classifier:", np.mean(np.array(f)))
    model = imdb_model
    base = np.zeros(shape=(100))
    m = []
    for i in range(10):
        x = x_test[i]
        coefs = shap_values[0][i]
        m.append(monotonicity_metric_txt(model,x,coefs,base))
    print(any(np.array(m)))

def main_image():
    x_train1,y_train1, x_test1,y_test1 = get_dataset(rgb = True)#rgb for lime
    x_train2,y_train2, x_test2,y_test2 = get_dataset(rgb = False)#grayscale for shap

    model1 = train(rgb = True)
    model1.compile(
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=keras.optimizers.Adam(),
      metrics=['accuracy']
    )

    print(model1.summary())

    model1.fit(
            x_train1, 
            y_train1, 
            epochs=2, 
            batch_size=32, 
            validation_data = (x_test1, y_test1))

    model2 = train(rgb = False)
    model2.compile(
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=keras.optimizers.Adam(),
      metrics=['accuracy']
    )

    print(model2.summary())

    model2.fit(
            x_train2, 
            y_train2, 
            epochs=2, 
            batch_size=32, 
            validation_data = (x_test2, y_test2))


    pred_fn1 = lambda images: model1.predict(images)
    #============================LIME Explainer ==========================================
    explainer = lime_image.LimeImageExplainer(random_state=42)

    explanation_val = []
    explanation=[]
    for i in range(4):
      e = explainer.explain_instance(
             x_test1[10], 
             pred_fn1)
      explanation_val.append(e.segments)
      explanation.append(e)

    pp.imshow(x_test1[10])
    image, mask = explanation[0].get_image_and_mask(
             model1.predict(
                  x_test1[10].reshape((1,28,28,3))
             ).argmax(axis=1)[0],
             positive_only=True, 
             hide_rest=False)
    pp.imshow(mark_boundaries(image, mask))

    explanation_val_np = np.array(explanation_val)

    #============================SHAP Explainer ==========================================

    background = x_test2[np.random.choice(x_test2.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model2, background)
    shap_values = e.shap_values(x_test2[1:5])
    shap.image_plot(shap_values, x_test2[1:5])#,matplotlib=True)
    sv = np.array(shap_values)
    
    #===========================================Evaluation========================================
    idx1=[]
    idx2=[]
    
    for i in range(5):
      x_grayscale = x_train2[i]
      x_rgbb = x_train1[i]
      pred_grayscale = pred_func(x_grayscale,model2)[0]
      pred_rgb = pred_func(x_rgbb,model1)[0]
      max_grayscale = np.amax(pred_grayscale)
      max_rgb = np.amax(pred_rgb)
      _idx1 = np.where(pred_rgb == max_rgb)[0]
      idx1.append(_idx1[0])
      _idx2 = np.where(pred_grayscale == max_grayscale)[0]
      idx2.append(_idx2[0])
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)


    base = np.zeros(x_test2[0].shape)
    fidelity_shap = []
    for i in range(4):
      __idx = idx2[i]
      coefs = sv[__idx,i,:,:,0]
      X = x_test2[i]
      fidelity_shap.append(faithfulness_metrics_cls(model2,X,coefs,base))

    base = np.zeros(x_test1[0].shape)
    # print(x_test1[0].shape)
    fidelity_lime = []
    for i in range(4):
      __idx = idx1[i]
      coefs = explanation_val_np[i]
      X = x_test1[i]
      # print(faithfulness_metrics_cls(model,X,coefs,base))
      fidelity_lime.append(faithfulness_metrics_cls(model1,X,coefs,base))
    print("Average Fidelity for SHAP",fidelity_shap)
    print("Average Fidelity for LIME",fidelity_lime)
    #Monotonicity
    mono_LIME = []
    for i in range(4):
      __idx = idx1[i]
      coefs = explanation_val_np[i]

      print("coef shape",coefs.shape)
      X = x_test1[i]
      print("X shape",X.shape)
      # print(faithfulness_metrics_cls(model,X,coefs,base))
      mono_LIME.append(monotonicity(model1,X,coefs,base))
    mono_SHAP = []
    base = np.zeros(x_test2[0].shape)
    for i in range(4):
      __idx = idx2[i]
      coefs = sv[__idx,i,:,:,0]

      print("coef shape",coefs.shape)
      X = x_test2[i]
      print("X shape",X.shape)
      # print(faithfulness_metrics_cls(model,X,coefs,base))
      mono_SHAP.append(monotonicity(model2,X,coefs,base))
    print("Monotonicity for SHAP",mono_SHAP)
    print("Monotonicity Fidelity for LIME",mono_LIME)
dataset1_cls = "wine"
dataset2_cls = "breast cancer"

Main_cls(dataset1_cls)
# Main_cls(dataset2_cls) 

# dataset1_reg = "boston"
# dataset2_reg = "superconductivity"
# dataset3_reg = "diabetes"


# Main_reg(dataset1_reg)
# Main_reg(dataset2_reg)
# Main_reg(dataset3_reg)

