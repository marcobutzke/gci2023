import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px

st.set_page_config(layout="wide")

@st.cache_data
def load_database():
    return pd.read_csv('brasil_estados.csv'), \
        json.load(open('brazil-states.geojson.txt'))

def highlight_class_lc(s):
    if s.outlier_max == 1:
        return ['background-color: #00cec9']*len(s)
    elif s.class_lc == 'acima':
        return ['background-color: #fab1a0']*len(s)
    elif s.class_lc == 'media':
        return ['background-color: #ffeaa7']*len(s)
    else:
        return ['background-color: #74b9ff']*len(s)

st.title('Meu primeiro App - GCI')

estados, fronteiras = load_database()

dados, estatistica, outlier, zvalues = st.tabs(['Dados', 'Estatística Descritiva', 'Outliers', 'Valores Padronizados'])

variaveis = ['area', 'populacao', 'densidade', 'matricula', 'idh', 'receitas', 'despesas', 'rendimento', 'veiculos']

with dados:
    if st.checkbox('Região'):
        regiao = st.selectbox('Selecione a Região:', estados['regiao_nome'].unique())
        st.dataframe(estados[estados['regiao_nome'] == regiao])
    else:
        st.table(estados)
    with st.expander('Mapa'):
        variavel = st.selectbox('Variável:', variaveis)
        minimo = estados[variavel].min()
        maximo = estados[variavel].max()
        mapa_px = px.choropleth_mapbox(
            data_frame = estados, 
            geojson = fronteiras, 
            locations='sigla', 
            featureidkey='properties.sigla',
            color=variavel,
            color_continuous_scale= 'reds',
            hover_name = 'sigla', 
            hover_data =['uf', variavel, 'regiao_nome'],    
            range_color=(minimo, maximo),
            mapbox_style='carto-positron',
            zoom=3.5, 
            center = {"lat": -15.76, "lon": -47.88},
            opacity=1,
            labels={'sigla' : 'Sigla',
                    'uf': 'Estado',
                    'regiao_nome': 'Região'
            },
            width = 1200,
            height = 800,
            title = 'Mapa do Brasil'
        )
        mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
        mapa_px.update_traces(marker_line_width=1)
        st.plotly_chart(mapa_px)

with estatistica:
    variavel = st.selectbox('Selecione a variavel', variaveis)
    col1, col2, col3, col4 = st.columns([3,1,2,1])
    col1.altair_chart(alt.Chart(estados).mark_bar().encode(x="uf:O", y=variavel+':Q').properties(height=500))
    col2.dataframe(round(estados[variavel].describe(),2))
    base = alt.Chart(estados)
    bar = base.mark_bar().encode(x=alt.X(variavel+':Q', bin=True), y='count()')
    rule = base.mark_rule(color='red').encode(x='mean('+variavel+'):Q', size=alt.value(5))
    rule2 = base.mark_rule(color='green').encode(x='median('+variavel+'):Q', size=alt.value(5))
    col3.altair_chart(bar + rule + rule2)
    col4.altair_chart(alt.Chart(estados).mark_boxplot().encode(y=variavel+':Q').properties(width=200))
with outlier:
    variavel = st.selectbox('Selecione a variavel para outliers', variaveis)
    estados_var = estados[['uf', variavel]].copy()
    iqr = estados_var[variavel].quantile(0.75) - estados_var[variavel].quantile(0.25)
    out_min = estados_var[variavel].quantile(0.25) - (1.5 * iqr)
    out_max = estados_var[variavel].quantile(0.75) + (1.5 * iqr)
    limite_inferior = estados_var[variavel].mean() - (1.96 * estados_var[variavel].std() / np.sqrt(len(estados_var)))
    limite_superior = estados_var[variavel].mean() + (1.96 * estados_var[variavel].std() / np.sqrt(len(estados_var)))
    estados_var['outlier_min'] = estados_var[variavel].apply(lambda x : 1 if x < out_min else 0)
    estados_var['outlier_max'] = estados_var[variavel].apply(lambda x : 1 if x > out_max else 0)
    estados_var['class_lc'] = estados_var[variavel].apply(
        lambda x : 'abaixo' 
        if x < limite_inferior 
        else (
            'acima' 
            if x > limite_superior 
            else 'media'
        ) 
    )
    st.dataframe(estados_var.style.apply(highlight_class_lc, axis=1))
    with st.expander('Média - Intervalo de Confiança'):
        col1, col2, col3 = st.columns(3)
        col1.metric('Média', round(estados_var[variavel].mean(),2), "0")
        col2.metric('Limite Inferior', round(limite_inferior,2), round(limite_inferior - estados_var[variavel].mean(),2))
        col3.metric('Limite Superior', round(limite_superior,2), round(limite_superior - estados_var[variavel].mean(),2))
        st.altair_chart(alt.Chart(estados_var).mark_bar().encode(x="uf:O", y=variavel+':Q', color='class_lc:N').properties(height=400))        
    with st.expander('Outlier - Amplitude'):
        col1, col2, col3 = st.columns(3)
        col1.metric('Média', round(estados_var[variavel].mean(),2), "0")
        col2.metric('Limite Inferior', round(out_min,2), round(out_min - estados_var[variavel].mean(),2))
        col3.metric('Limite Superiorr', round(out_max,2), round(out_max - estados_var[variavel].mean(),2))
        st.altair_chart(alt.Chart(estados_var).mark_bar().encode(x="uf:O", y=variavel+':Q', color='outlier_max:N').properties(height=400))
    with st.expander('Sem Outlier - Nova Média'):
        estados_var_out = estados_var[(estados_var['outlier_max'] == 0) & (estados_var['outlier_min'] == 0)].copy()
        limite_inferior = estados_var_out[variavel].mean() - (1.96 * estados_var_out[variavel].std() / np.sqrt(len(estados_var_out)))
        limite_superior = estados_var_out[variavel].mean() + (1.96 * estados_var_out[variavel].std() / np.sqrt(len(estados_var_out)))
        estados_var_out['class_lc'] = estados_var_out[variavel].apply(
            lambda x : 'abaixo' 
            if x < limite_inferior 
            else (
                'acima' 
                if x > limite_superior 
                else 'media'
            ) 
        )
        col1, col2, col3 = st.columns(3)
        col1.metric('Média', round(estados_var_out[variavel].mean(),2), "0")
        col2.metric('Limite Inferior', round(limite_inferior,2), round(limite_inferior - estados_var_out[variavel].mean(),2))
        col3.metric('Limite Superiorr', round(limite_superior,2), round(limite_superior - estados_var_out[variavel].mean(),2))
        st.altair_chart(alt.Chart(estados_var_out).mark_bar().encode(x="uf:O", y=variavel+':Q', color='class_lc:N').properties(height=400))        
with zvalues:
    colunas = st.multiselect('colunas', variaveis)
    if len(colunas) > 0:
        sel = colunas
        sel.insert(0, "uf")
        estadosz = estados[sel].copy()
        listaz = []
        for col in estadosz.columns:
            if col != 'uf':
                media = estadosz[col].mean()
                dp = estadosz[col].std()
                estadosz['z_'+col] = estadosz[col].apply(lambda x : (x - media) / dp)
                listaz.append('z_'+col)
        listaz.insert(0, "uf")
        with st.expander('Dados'):
            st.dataframe(estadosz.style.hide_index().background_gradient(cmap='Blues'))
        with st.expander('Gráfico'):
            graphz = pd.DataFrame()
            for zvalue in listaz:
                if zvalue != 'uf':
                    for index, row in estadosz.iterrows():
                        graphz = graphz.append({'uf': row['uf'], 'variable': zvalue, 'valor': row[zvalue]}, ignore_index=True)
            st.altair_chart(alt.Chart(graphz).mark_bar(opacity=0.5).encode(x='uf:O', y='valor:Q', color='variable:N').properties(height=400))    
        with st.expander('Ranking'):
            if len(colunas) > 0:
                data = estados[colunas]
                print(data)
                dataz = pd.DataFrame()
                for col in data.columns:
                    if col != 'uf':
                        media = estados[col].mean()
                        dp = estados[col].std()
                        dataz[col] = estados[col].apply(lambda x: (x - media) / dp)
                dataz['total'] = dataz.sum(
                    axis=1,
                    skipna=True
                )
                dataz['ranking'] = dataz['total'].rank(ascending=False)
                iqr = dataz['total'].quantile(0.75) - dataz['total'].quantile(0.25)
                out_min = dataz['total'].quantile(0.25) - (1.5 * iqr)
                out_max = dataz['total'].quantile(0.75) + (1.5 * iqr)
                erro = 1.96 * dataz['total'].std() / np.sqrt(len(data))
                li = dataz['total'].mean() - erro
                ls = dataz['total'].mean() + erro
                dataz['zscore'] = (dataz['total'] - dataz['total'].mean()) / dataz['total'].std()
                dataz['stars'] = round(dataz['zscore'], 0) + 3
                dataz['outlier_min'] = dataz['total'].apply(
                    lambda x: 1 if x < out_min
                    else 0
                )
                dataz['outlier_max'] = dataz['total'].apply(
                    lambda x: 1 if x > out_max
                    else 0
                )
                media = dataz['total'].mean()
                dataz['class_media'] = dataz['total'].apply(
                    lambda x: 'abaixo' if x < media
                    else 'acima'
                )
                dataz['class_lc'] = dataz['total'].apply(
                    lambda x: 'abaixo' if x < li
                    else (
                        'acima' if x > ls
                        else 'media'
                    )
                )
                datac = estados[['regiao_nome','uf']].copy()
                datac = datac.merge(dataz, left_index=True, right_index=True)
                data_sort = datac.sort_values(by='ranking')
                st.table(data_sort.style.hide_index().background_gradient(cmap='Blues'))


