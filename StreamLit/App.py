import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model('HiggsBoson.hdf5')

def preprocess_test(arr):
    arr=np.array(arr)
    arr=arr.reshape(-1, 14)
    return arr

def main():
  
    st.header('Higgs Boson Event Detection')
    st.write('This is a simple demo of the Streamlit framework')
    st.write('It demonstrates how to load a model, make predictions, and display the results')
    st.write('The model was trained on the Higgs Boson dataset')
    st.subheader('Input the Data')
    st.write('Please input the data below')
    colnames=['DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h','DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt','DER_pt_ratio_lep_tau', 'PRI_tau_pt', 'PRI_lep_pt', 'PRI_lep_phi','PRI_met', 'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_all_pt']
    
    v1=st.number_input(colnames[0],)
    v2=st.number_input(colnames[1],)
    v3=st.number_input(colnames[2],)
    v4=st.number_input(colnames[3],)
    v5=st.number_input(colnames[4],)
    v6=st.number_input(colnames[5],)
    v7=st.number_input(colnames[6],)
    v8=st.number_input(colnames[7],)
    v9=st.number_input(colnames[8],)
    v10=st.number_input(colnames[9],)
    v11=st.number_input(colnames[10],)
    v12=st.number_input(colnames[11],)
    v13=st.number_input(colnames[12],)
    v14=st.number_input(colnames[13],)
    
    
    vals=[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14]
    
    
    if st.button('Detect Event'): 
      vals= preprocess_test(vals)
      pred=(np.argmax(model.predict(vals)))
      st.write("Predicted val")
      if pred==0:
        st.success("Predicted val"+" b")
      else:
        st.success("Predicted val"+" s")
      
    
if __name__ == '__main__':
    main()  
    
    
