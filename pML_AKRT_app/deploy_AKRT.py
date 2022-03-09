#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###web application: Simplify ML for AKRT##

#import essential library
import streamlit as st
import pandas as pd 
import numpy as np
import pickle
# import sklearn
import xgboost as xgb

#load model
##xgb
xgbmodel = xgb.XGBClassifier()
xgbmodel.load_model('XGBoost_AKRT.json')

##RF
filename = 'RandomForests_AKRT.sav'
rfmodel = pickle.load(open(filename, 'rb'))

##DT
filename = 'DecisionTree_AKRT.sav'
dtmodel = pickle.load(open(filename, 'rb'))

##MinMaxScaler
filename = 'MinMaxScaler_AKRT.sav'
min_max_scaler = pickle.load(open(filename, 'rb'))

##SVM
filename = 'SVM_AKRT.sav'
svmmodel = pickle.load(open(filename, 'rb'))

## MLP
filename = 'MLP_AKRT.sav'
mlpmodel = pickle.load(open(filename, 'rb'))

##LR
filename = 'LogisticRegression_AKRT.sav'
lrmodel = pickle.load(open(filename, 'rb'))


#function
@st.cache
def fit_xgb(info):
    res = xgbmodel.predict(info)
    return res
#     if res ==1:
#       return 'Yes'
#     else:
#       return 'No'
@st.cache
def fit_xgb_prob(info):
    res = xgbmodel.predict_proba(info)
    return res[0,1]

@st.cache
def fit_rf(info):
    res = rfmodel.predict(info)
    return res
#     if res ==1:
#       return 'Yes'
#     else:
#       return 'No'
@st.cache
def fit_rf_prob(info):
    res = rfmodel.predict_proba(info)
    return res[0,1]

@st.cache
def transform_scaler(info):
    res = min_max_scaler.transform(info)
    return res
#     if res ==1:
#       return 'Yes'
#     else:
#       return 'No'

@st.cache
def fit_mlp(info_scaled):
    res = mlpmodel.predict(info_scaled)
    return res
#     if res ==1:
#       return 'Yes'
#     else:
#       return 'No'
@st.cache
def fit_mlp_prob(info_scaled):
    res = mlpmodel.predict_proba(info_scaled)
    return res[0,1]

@st.cache
def fit_svm(info_scaled):
    res = svmmodel.predict(info_scaled)
    return res
#     if res ==1:
#       return 'Yes'
#     else:
#       return 'No'
@st.cache
def fit_svm_prob(info_scaled):
    res = svmmodel.predict_proba(info_scaled)
    return res[0,1]

@st.cache
def fit_dt(info):
    res = dtmodel.predict(info)
    return res
#     if res ==1:
#       return 'Yes'
#     else:
#       return 'No'
@st.cache
def fit_dt_prob(info):
    res = dtmodel.predict_proba(info)
    return res[0,1]

@st.cache
def fit_lr(info):
    res = lrmodel.predict(info)
    return res
#     if res ==1:
#       return 'Yes'
#     else:
#       return 'No'
@st.cache
def fit_lr_prob(info):
    res = lrmodel.predict_proba(info)
    return res[0,1]



# """model
# ['gender', 'age', 'bun_1', 'creatinine_1', 'diuretic_1', 'nonrenal_sofa',
#        'ht', 'dm', 'ckd', 'sepsis', 'akistage1_1', 'akistage1_2',
#        'akistage1_3', 'serum_potassium_-2', 'serum_potassium_-1',
#        'serum_potassium_0', 'serum_potassium_1', 'serum_potassium_3',
#        'serum_potassium_4'],"""

#start application
st.title('The Pragmatic Machine Learning Algorithm to Predict Acute Kidney Replacement Therapy in Critically-ill Patient (*pML-AKRT*)')
txt1 = '''To predict AKRT within 7 days of critically-ill patient.\n 
We purpose our best two models: XGBoost and Multilayer Perceptron.'''
st.write(txt1)

####Input###
st.header('Please input values')

gender = st.radio('Gender', ['male','female']) #male = 1, female =2
if gender == 'male':
    gender = 1
else:
    gender = 2

age = st.number_input('Age',min_value = 18,max_value = 100, value =50, step = 1)

#basic diseases
ht = st.radio('Hypertension?', ['No','Yes'])
if ht == 'No':
    ht = 0
else:
    ht = 1

dm = st.radio('Diabetes mellitus?', ['No','Yes'])
if dm == 'No':
    dm = 0
else:
    dm = 1

ckd = st.radio('Chronic kidney disease?', ['No','Yes'])
if ckd == 'No':
    ckd = 0
else:
    ckd = 1

sepsis = st.radio('Have sepsis?', ['No','Yes'])
if sepsis == 'No':
    sepsis = 0
else:
    sepsis = 1

stage = st.radio('AKI staging - maximun staging: urine output or serum creatinine criteria', 
                 ['no AKI','I','II','III'])
if stage == 'I':
    akistage1 = 1
    akistage2 = 0
    akistage3 = 0
elif stage == 'II':
    akistage1 = 0
    akistage2 = 1
    akistage3 = 0  
elif stage == 'III':
    akistage1 = 0
    akistage2 = 0
    akistage3 = 1
else: 
    akistage1 = 0
    akistage2 = 0
    akistage3 = 0
st.write('[AKI staging-definition](https://drive.google.com/file/d/1tUY2F1jbCcoxtR4lqGxJueFWyNP_FPNb/view?usp=sharing)')

diuretic = st.radio('Received Diuretic (furosemide or other)?', ['No','Yes'])
if diuretic == 'No':
    diuretic = 0
else:
    diuretic = 1

bun = st.number_input('Blood Urea Nitrogen (mg/dL)', min_value=5,max_value=200,value=20,step=1)

cr = st.number_input('Serum Creatinine (mg/dL)', min_value=0.1,max_value=20.0,value=1.0,step=0.1)

potassium = st.number_input('Serum Potassium', min_value=0.2,max_value=10.0,value=3.5,step=0.1)
if potassium < 2.5:
    serum_potassium_m2 = 0
    serum_potassium_m1 = 0
    serum_potassium_0 = 0
    serum_potassium_1 = 0
    serum_potassium_3 = 0
    serum_potassium_4 = 0
if 2.5 <=  potassium < 3.0:
    serum_potassium_m2 = 1
    serum_potassium_m1 = 0
    serum_potassium_0 = 0
    serum_potassium_1 = 0
    serum_potassium_3 = 0
    serum_potassium_4 = 0
if 3.0 <=  potassium < 3.5:
    serum_potassium_m2 = 0
    serum_potassium_m1 = 1
    serum_potassium_0 = 0
    serum_potassium_1 = 0
    serum_potassium_3 = 0
    serum_potassium_4 = 0
if 3.5 <=  potassium < 5.5:
    serum_potassium_m2 = 0
    serum_potassium_m1 = 0
    serum_potassium_0 = 1
    serum_potassium_1 = 0
    serum_potassium_3 = 0
    serum_potassium_4 = 0
if 5.5 <=  potassium < 6.0:
    serum_potassium_m2 = 0
    serum_potassium_m1 = 0
    serum_potassium_0 = 0
    serum_potassium_1 = 1
    serum_potassium_3 = 0
    serum_potassium_4 = 0
if 6.0 <=  potassium < 7:
    serum_potassium_m2 = 0
    serum_potassium_m1 = 0
    serum_potassium_0 = 0
    serum_potassium_1 = 0
    serum_potassium_3 = 1
    serum_potassium_4 = 0
if 7.0 <=  potassium:
    serum_potassium_m2 = 0
    serum_potassium_m1 = 0
    serum_potassium_0 = 0
    serum_potassium_1 = 0
    serum_potassium_3 = 0
    serum_potassium_4 = 1
    
    
    
st.write('For calculate non-renal SOFA score')
#pfratio
pfratio = st.radio('PaO2/FiO2', ['≥ 400 mmHg',
                                '300 - 399 mmHg',
                                '200-299 mmHg',
                                '≤ 199 mmHg and NOT mechanically ventilated',
                                '100 - 199 mmHg and mechanically ventilated',
                                '<100 mmHg and mechanically ventilated'])
if pfratio == '≥ 400 mmHg':
    pfratio_num = 0
elif pfratio == '300 - 399 mmHg':
    pfratio_num = 1
elif pfratio == '200-299 mmHg':
    pfratio_num = 2
elif pfratio == '≤ 199 mmHg and NOT mechanically ventilated':
    pfratio_num = 2
elif pfratio == '100 - 199 mmHg and mechanically ventilated':
    pfratio_num = 3
else:
    pfratio_num = 4
    
#platelet
platelet = st.radio('Platelets', ['≥ 150,000',
                                '100,000 - 149,000',
                                '50,000 - 99,000',
                                '20,000 - 49,000',
                                '<20,000'])
if platelet == '≥ 150,000':
    platelet_num = 0
elif platelet == '100,000 - 149,000':
    platelet_num = 1
elif platelet == '50,000 - 99,000':
    platelet_num = 2
elif platelet == '20,000 - 49,000':
    platelet_num = 3
else:
    platelet_num = 4

#Bilirubin
bilirubin = st.number_input('Bilirubin (mg/dL)',min_value = 0.0,max_value = 40.0, value =1.0, step = 0.2)
if bilirubin < 1.2:
    bilirubin_num = 0
elif 1.2 <= bilirubin <= 1.9:
    bilirubin_num = 1
elif 2.0 <= bilirubin <= 5.9:
    bilirubin_num = 2
elif 6.0 <= bilirubin <= 11.9:
    bilirubin_num = 3
else:
    bilirubin_num = 4

    #GCS
gcs = st.number_input('Glasgow Coma Scale (0 - 15 points)',min_value = 0,max_value = 15, value =15, step = 1)
if gcs == 15:
    gcs_num = 0
elif 13 <= gcs <= 14:
    gcs_num = 1
elif 10 <= gcs <= 12:
    gcs_num = 2
elif 6 <= gcs <= 9:
    gcs_num = 3
else:
    gcs_num = 4
    
#Mean arterial pressure
MAP = st.radio('Mean arterial pressure', ['No hypotension',
                                         'MAP < 70 mmHg',
                                         'Dopamine ≤ 5 mcg/kg/min or Dobutamine (any dose)',
                                         'Dopamine > 5, Epinephrine ≤ 0.1, or Norepinephrine ≤ 0.1',
                                         'Dopamine > 15, Epinephrine > 0.1, or Norepinephrine > 0.1'])
if MAP == 'No hypotension':
    MAP_num = 0
elif MAP == 'MAP < 70 mmHg':
    MAP_num = 1
elif MAP == 'Dopamine ≤ 5 mcg/kg/min od Dobutamine (any dose)':
    MAP_num = 2
elif MAP == 'Dopamine > 5, Epinephrine ≤ 0.1, or Norepinephrine ≤ 0.1':
    MAP_num = 3
elif MAP == 'Dopamine > 15, Epinephrine > 0.1, or Norepinephrine > 0.1':
    MAP_num = 4
    
#non-renal SOFA
non_renal_SOFA = pfratio_num + platelet_num + gcs_num + bilirubin_num+ MAP_num

##wrap up data
patient = np.array([[gender, age, bun, cr, diuretic, non_renal_SOFA,
                        ht, dm, ckd, sepsis, akistage1, akistage2, akistage3,
                        serum_potassium_m2, serum_potassium_m1, serum_potassium_0,
                        serum_potassium_1, serum_potassium_3, serum_potassium_4]])

# st.write(patient)


#predict
# ans_xgb = fit_xgb(patient)
ans_xgb_prob = fit_xgb_prob(patient)
patient_scaled = transform_scaler(patient)
ans_mlp_prob = fit_mlp_prob(patient_scaled)
ans_svm_prob = fit_svm_prob(patient_scaled)
# ans_rf = fit_rf(patient)
ans_rf_prob = fit_rf_prob(patient)
ans_dt_prob = fit_dt_prob(patient)
ans_lr_prob = fit_lr_prob(patient)


#result
if st.button('Click to Calculated', key = 'predict button'):
    st.header('Results')
    st.subheader('XGBoost model')
#     st.write('Predict AKRT: ', ans_xgb)
    st.write('The probability of AKRT within 7 days is', round(ans_xgb_prob*100,2), '%')
    st.subheader('Multilayer-perceptron model')
#     st.write('Predict AKRT: ', ans_rf)
    st.write('The probability of AKRT within 7 days is', round(ans_mlp_prob*100,2), '%')
    
    st.write('...............................................................................................................')
    st.write('We strongly recommend the first two models. However, we also constructed other models that are shown below.')

    st.subheader('Random forests model')
    st.write('The probability of AKRT within 7 days is', round(ans_rf_prob*100,2), '%')
    st.subheader('Logistic regression model')
    st.write('The probability of AKRT within 7 days is', round(ans_lr_prob*100,2), '%')
    st.subheader('Decision tree model')
    st.write('The probability of AKRT within 7 days is', round(ans_dt_prob*100,2), '%')
    
st.write('----------------------------------------------------------------------------') 

# sidebar
st.sidebar.header('pML-AKRT')
txt2 = '''The models were trained and tested on data from [SEA-AKI study](https://www.sciencedirect.com/science/article/abs/pii/S088394412100085X). This data was provided by Dr. Nattachai Srisawat. All models were externally validated with Vajira Acute Kidney Injury Dataset.
The complete experiment method and results have been prepared and will be published soon. However, The performance of models is partially reported below. The model construction, experiment, and web development have been done by Dr. Wanjak Pongsittisak and [VBaM4H](https://vbam4h.com). 
For more information or any suggestion, don't hesitate to get in touch with [us](mailto:wanjak@nmu.ac.th).''' 
st.sidebar.write(txt2)
st.sidebar.write('Version 0.91')
st.sidebar.markdown('&copy; 2021 VBaM4H All Rights Reserved')


#table below
#performance
performance = {'AUC DevDataset': [0.942,0.944], 
        'AUC ValDataset': [0.772, 0.7444],
        'Accuracy DevDataset': [0.90,0.91],
        'Accuracy ValDataset': [0.78,0.73],
        'Area under Precision-Recall curve DevDataset': [0.70,0.68],
        'Area under Precision-Recall curve ValDatase': [0.55,0.54]}
tab = pd.DataFrame(performance, index=['XGBoost',' Multilayer Perceptron'])
st.header('Explaination')
st.header('Performance of algorithm')
st.dataframe(tab.T)

#threshold explained
st.header('What is the cut-point?')
txt4 = '''The cut-point for the machine learning algorithm for the classification task is 0.5. However, the models tend to classify the majority class due to the imbalanced dataset. Hence we use the technique of moving the threshold from the AUROC or Precision-Recall curve to get the best cut-point for each model. The cut-points are shown below.'''
st.write(txt4)

#table threshold
threshold = {
    'Optimized by AUROC': ['12%','15%'],
    'Optimized by Precision-Recall curve':['16%','31%'],
}
tab_threshold = pd.DataFrame(threshold, index=['XGBoost',' Multilayer Perceptron'])
st.header('Cut-point')
st.dataframe(tab_threshold.T)

#thank you
st.header('Credit')
txt3 = '''All code is written by [Python 3.7] (https://www.python.org/). We thank everyone who contributes to all libraries and packages that we used: [Pandas](https://pandas.pydata.org/), [Numpy](https://numpy.org/), [Scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.readthedocs.io/en/latest/), [Matplotlib](https://matplotlib.org/), and [Streamlit](https://streamlit.io/)'''
st.write(txt3)

