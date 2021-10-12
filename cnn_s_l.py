import tensorflow as tf
tf.compat.v1.disable_v2_behavior() 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random
import shap
# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

def plot(x):
  f = x[:,:,0]

  fig = plt.figure
  plt.imshow(f, cmap='gray_r')
  plt.show()
   

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
# pred_fn2 = lambda image: model2.predict(image.reshape((1,28,28,1)))#np.expand_dims(image, axis=0))
def pred_func(images,model):
  if images.shape[2] == 1:
    return model.predict(images.reshape((1,28,28,1)))
  elif images.shape[2] == 3:
    return model.predict(images.reshape(1,28,28,3))

# pred_fn2(x_test2)[0]


def faithfulness_metrics_cls(model,X,coefs,base):
    
    pred_class = np.amax(pred_func(X,model))
    # ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
    pred_probs = []
    shap_list = []
    # x = X[0,:,:] 
    # print("original")
    plot(X)
    for i in range(1,5):
      for j in range(1,5):
        #compute average shap_val for superpixel of size (4x4)
        avg_shap = np.average(coefs[7*(i-1) :7*i ,7*(j-1) :7*j])
        # print(avg_shap)
        shap_list.append(avg_shap)                      
        x_copy = X.copy()
        # print(x_copy[0,7*(i-1) :7*i ,7*(j-1) :7*j].shape)
        # print(base[0,7*(i-1) :7*i ,7*(j-1) :7*j].shape)
        x_copy[7*(i-1) :7*i ,7*(j-1) :7*j,0] = base[7*(i-1) :7*i ,7*(j-1) :7*j,0]
        # print("ablated")
        plot(x_copy)
        # print("Avg shap",avg_shap)
        x_copy_pr = np.amax(pred_func(x_copy,model))
        # print(x_copy_pr)
        # print(pred_class)
        pred_probs.append(x_copy_pr-pred_class)
      
    #   print(i)
    # print(shap_list)
    # print(pred_probs)

    return -np.corrcoef(np.array(shap_list), np.array(pred_probs))[0,1]


def mainCNN():
    
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

    plt.imshow(x_test1[10])
    image, mask = explanation[0].get_image_and_mask(
             model1.predict(
                  x_test1[10].reshape((1,28,28,3))
             ).argmax(axis=1)[0],
             positive_only=True, 
             hide_rest=False)
    plt.imshow(mark_boundaries(image, mask))

    explanation_val_np = np.array(explanation_val)

    #============================SHAP Explainer ==========================================


    # masker = shap.maskers.Image("inpaint_telea", x_test[0].shape)
    background = x_test2[np.random.choice(x_test2.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model2, background)


    shap_values = e.shap_values(x_test2[1:5])
    #shap_values_neg = e.shap_values(-x_test2[1:5])

    shap.image_plot(shap_values, x_test2[1:5])

    #shap.image_plot(shap_values_neg, x_test2[1:5])
    sv = np.array(shap_values)#.shape
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
      # print(faithfulness_metrics_cls(model,X,coefs,base))
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
mainCNN()