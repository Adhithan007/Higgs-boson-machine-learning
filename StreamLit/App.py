import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

model = load_model('StreamLit/HiggsBoson.hdf5')

def preprocess_test(arr):
    arr=np.array(arr)
    arr=arr.reshape(-1, 14)
    return arr

def main():
  
    st.header('Higgs Boson Event Detection')
    st.write('This is a simple demo of the Streamlit framework')
    st.write('It demonstrates how to load a model, make predictions, and display the results')
    
    
    if (st.button("Upload dataset")):
        file = st.file_uploader('Dataset')
    
    if(file):
        df = pd.read_csv(file)
        st.write(df.head())
        df=df.drop(columns=['EventId','Weight'])
        st.subheader('About Data:')
        st.write(df.describe().T)
#         st.bar_chart(df['Label'].value_counts())
#         fig,axes=plt.subplots(figsize=(10,8))
#         sns.barplot(x = df['Label'].value_counts().index, y = df['Label'].value_counts().values)
#         plt.title('Label counts')
         
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(7, 4))
        ax = sns.barplot(x = df['Label'].value_counts().index, y = df['Label'].value_counts().values)
        st.pyplot(fig)
        
        st.subheader("Columns Dropped for the value of -999 count > 100000")
        
        for col in (df.columns):
            if -999 in df[col].value_counts().index:
                if(int(df[col].value_counts()[-999])>=90000):
                    df=df.drop(columns=[col])
                    st.write("         ",col)
        
        with st.echo():
            for col in (df.columns):
                if -999 in df[col].value_counts().index:
                    if(int(df[col].value_counts()[-999])>=90000):
                        df=df.drop(columns=[col])
                        print("   ",col)
   

        st.subheader("Columns Dropped for the value of Mean < 0")
    
        ind=(df.mean()).index
        ke =(df.mean()).values
        for i,col in enumerate(ind):
            if(ke[i]<0):
                df=df.drop(columns=[col])
                st.write("          ",col)
        
        with st.echo():
            ind=(df.mean()).index
            ke =(df.mean()).values
            for i,col in enumerate(ind):
                if(ke[i]<0):
                    df=df.drop(columns=[col])
                    print("   ",col)
        
        df[df==-999.000] = np.NaN
        df.fillna(df.mean(), inplace = True)
        X_train = df.drop(columns=['Label'])
        
        
        
        st.subheader("Finally preprocessed X_train data")
        f, ax = plt.subplots(figsize=(14, 14))
        corr = df.corr()
        hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                         linewidths=.05)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );

        f.subplots_adjust(top=0.93)
        t= f.suptitle(' Correlation Heatmap', fontsize=14)
        st.pyplot(f)
        
        
st.subheader("The basic DL model for the data")
with st.echo():
    # baseline model
    def create_baseline():
        # create model
        model = Sequential()
        model.add(Dense(60, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        # Compile model
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
        
        
    if st.button('Wannna test Model with Your DATA'):    
        st.write('Model used is trained using the above model')
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
    
    
