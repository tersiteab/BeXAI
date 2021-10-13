# BeXAI
This is a benchmark suite for selected explainable AI (XAI).
<p>The benchmark suite consists of Target machine learning models that are to be explained. These models are trained across text, image and tabular datasets. Two prominent model-agnostic, post-hoc local explainers are used to generate interpretations of predictions made by the target models: LIME(Local Interpretable Model-agnostic Explanation) and SHAP(SHapley Additive exPlanation). To evaluate the performance of these explainers, fidelity and monotonicity metrics are used. Fidelity measures how well the exlanations match with the model prediction. Monotonicity(consistency) is desirabe characterstic that feature importance based explainers like SHAP and LIME to have. Monotonicity is...</p>

## Target Models

**Regression Models**:

- Linear Regression

- Random Forest Regression

- Support Vector Machine Regression


**Classification Models**:

- Logistic Regression

- Random Forest Classification

- Support Vector Machine Classification

- CNN Image Classification

- RNN Text Classification

## Datasets:

**Tabular Datasets**:

For Regression Task

- Boston Housing Dataset-available on Scikit Learn Library

- Superconductivity-available on UCI ML dataset repository [Superconductivty Data Data Set](https://archive.ics.uci.edu/ml/datasets/superconductivty+data)

- Diabetes Dataset- available on Scikit Learn Library

For Classification Task

- Wine Dataset-available on Scikit Learn Library

- Breast Cancer Dataset-available on Scikit Learn Library


**Image Dataset**:

- MNist handwritten digit dataset-available on Keras Library

**Text Dataset**:

- IMDB -available on Scikit Learn Library

## Explainers

**SHAP**: proposed by Lundberg et al, SHAP explainer uses coalition game theory to calculate the contributions of each feature to the prediction.

Paper: [Lundberg, Scott M., and Su-In Lee. “A unified approach to interpreting model predictions” Advances in Neural Information Processing Systems. 2017](https://arxiv.org/abs/1705.07874)

**LIME**:  creates explanation by building proxy linear model(like Linear reg and Decision tree) that approximates the target model around the instance of interest and assigning importance of features according to the coefficients of the proxy model 

Paper: [Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. “Why should I trust you?: Explaining the predictions of any classifier.” Proceedings of the 22nd](https://arxiv.org/abs/1602.04938) 
