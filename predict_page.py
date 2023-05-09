# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:14:55 2023

@author: Henry Kha
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import preprocessing

model_path = "best_xgb.sav"
def loading_model(model_path=model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
model = loading_model()

datapath = "noddpairname.csv"

df = pd.read_csv(datapath)


ddp_encoder = preprocessing.LabelEncoder()
ddp_encoder.fit(df['DDPairs'])
df['DDPairs']= ddp_encoder.transform(df['DDPairs'])  

interpretation = "classes_of_ddis.csv"
pred_interpretation = pd.read_csv(interpretation)
classes_encoder = preprocessing.LabelEncoder()
classes_encoder.fit(pred_interpretation['DDI'])

pred_interpretation['DDI_encoder']= classes_encoder.transform(pred_interpretation['DDI'])


def print_output(interpretation=interpretation):
    pred_interpretation = pd.read_csv(interpretation)
    printed_output = pred_interpretation[pred_interpretation["class"]==y_pred.astype(int)[0]]["DDI"].iloc[0].replace("drug_a",DRUG).replace("drug_b",INTERACTING_DRUG)
    return printed_output


b = tuple(pd.read_csv("drugs.csv")["interacting"].values.tolist())

def show_predict_page():
    st.title("Drug-Drug-Interaction Recommendation")
    st.write("""Diabetes Oral Drugs""")
    DRUGs = (
        'Acarbose',
         'Acetohexamide',
         'Alogliptin',
         'Benfluorex',
         'Buformin',
         'Canagliflozin',
         'Carbutamide',
         'Chlorpropamide',
         'Dapagliflozin',
         'Empagliflozin',
         'Ertugliflozin',
         'Evogliptin',
         'Gemigliptin',
         'Glibornuride',
         'Gliclazide',
         'Glipizide',
         'Gliquidone',
         'Glisoxepide',
         'Glyburide',
         'Glymidine',
         'Ipragliflozin',
         'Linagliptin',
         'Lobeglitazone',
         'Luseogliflozin',
         'Metahexamide',
         'Metformin',
         'Miglitol',
         'Mitiglinide',
         'Phenformin',
         'Pioglitazone',
         'Repaglinide',
         'Rosiglitazone',
         'Rosuvastatin',
         'Saxagliptin',
         'Sitagliptin',
         'Sotagliflozin',
         'Teneligliptin',
         'Tolazamide',
         'Tolbutamide',
         'Troglitazone',
         'Vildagliptin',
         'Voglibose'
        )
    
    INTERACTING_DRUGs = b
    
    DRUG = st.selectbox("Query_Drugs", DRUGs)
    
    INTERACTING_DRUG = st.selectbox("Interacting_Drugs",INTERACTING_DRUGs)
    
    ok = st.button("Predict the Interaction of two Drugs")
    
    if ok:
        try:
            DRUG, INTERACTING_DRUG = np.array([[DRUG,INTERACTING_DRUG]])[0]
            com = DRUG+"_"+INTERACTING_DRUG
            g = ddp_encoder.transform([com]).tolist()[0]
            g = df[df["DDPairs"]==g]
            g = g.drop("DDPairs",axis=1)
            y_pred = model.predict(g)
            output = classes_encoder.inverse_transform([y_pred]).tolist()[0]
            output = output.replace("drug_a",DRUG).replace("drug_b",INTERACTING_DRUG)
            #idx = np.where(df==com)
            st.subheader(f'Recommendation: \n {output}')
        except ValueError:
            st.subheader("There are no interactions found")
        
        
    
        
