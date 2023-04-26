import os
import shutil
import streamlit as st
import pandas as pd

diretorio = st.selectbox('Selecione a pasta:', ['001/', '002/', '003/'])

if st.button('Copiar'):
    os.mkdir(diretorio)
    shutil.copyfile('brasil_estados.csv', diretorio+"brasil_estados.csv")
    print(diretorio+'brasil_estados.csv')
    data = pd.read_csv(diretorio+'brasil_estados.csv')    
    st.dataframe(data)    
