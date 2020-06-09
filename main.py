import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.simplefilter(action = 'ignore')

def main():
    #st.image('codenationTD.png',format='PNG')
    st.title('AceleraDev Data Science 2020')
    st.subheader('**Recommend leads**')
    #Leitura dos dados
    df1 = pd.read_csv('port1.csv')
    df2 = pd.read_csv('port2.csv')
    m1 = pd.read_csv('./data\market1.csv')
    m2 = pd.read_csv('./data\market2.csv')
    m3 = pd.read_csv('./data\market3.csv')
    m4 = pd.read_csv('./data\market4.csv')
    df = m1.append(m2).append(m3).append(m4)
    df1 = pd.read_csv('./data\port1.csv')
    df2 = pd.read_csv('./data\port2.csv')
    df3 = pd.read_csv('./data\port3.csv')
    
    # Filtrando df
    base = ['id','idade_empresa_anos','qt_filiais','fl_me','fl_sa','fl_epp','fl_mei',
            'fl_ltda','de_natureza_juridica','sg_uf','de_ramo','setor','nm_divisao',
            'nm_segmento','de_nivel_atividade','nm_meso_regiao','nm_micro_regiao',
            'de_faixa_faturamento_estimado','de_faixa_faturamento_estimado_grupo']
    df_nao_nulos = df[base]
    # Retirando as observações com nulos
    df_nao_nulos.dropna(inplace=True)
    
    # Transformando as colunas com o LabelEncoder
    colunas_transform = list(df_nao_nulos.select_dtypes(include=['object','bool']).columns)
    colunas_transform.remove('id')
    encoder = LabelEncoder()
    for label in colunas_transform:
        label_coluna = 'cod_' + label
        df_nao_nulos[label_coluna] = encoder.fit_transform(df_nao_nulos[label])
        
    # Adicionando identificação dos portifólios
    df1['portfolio'] = 1
    df2['portfolio'] = 2
    df3['portfolio'] = 3
    # Juntando os clientes
    df_clientes = df1.append(df2).append(df3)
    # Identificando os clientes na base de mercado e na base de não nulos
    df = df.join(df_clientes.set_index('id'), on='id')
    df_nao_nulos = df_nao_nulos.join(df_clientes.set_index('id'), on='id')
    # Preenchendo os demais portifolios do mercado como 0
    df['portfolio'].fillna(0, inplace=True)
    df_nao_nulos['portfolio'].fillna(0, inplace=True)
    
    # Selecionando dados de treino
    train = ['cod_de_natureza_juridica','cod_sg_uf','cod_de_ramo','cod_setor','cod_nm_divisao',
             'cod_nm_segmento','cod_de_nivel_atividade','cod_nm_meso_regiao',
             'cod_de_faixa_faturamento_estimado','cod_de_faixa_faturamento_estimado_grupo']
    X = df_nao_nulos[train]
    # Treinando modelo
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    # Adicionando as classe no df
    labels = kmeans.labels_
    df_nao_nulos['kmeans'] = labels
    
    # Classe mais intensa em cada portfolio
    class_port1 = df_nao_nulos.query('portfolio == 1')['kmeans'].value_counts().index[0]
    class_port2 = df_nao_nulos.query('portfolio == 2')['kmeans'].value_counts().index[0]
    class_port3 = df_nao_nulos.query('portfolio == 3')['kmeans'].value_counts().index[0]
    # Fazendo seleção do exemplo a ser explorado
    st.markdown('**Seleção do Portfólio Exemplo**')
    select_analise = st.radio('Escolha um portfólio abaixo :', ('Portfólio 1', 'Portfólio 2', 'Portfólio 3'))
    if select_analise == 'Portfólio 1':
        df_port = df_nao_nulos.query('kmeans == @class_port1 and portfolio not in ("1")').iloc[:,0:18]
    if select_analise == 'Portfólio 2':
        df_port = df_nao_nulos.query('kmeans == @class_port2 and portfolio not in ("2")').iloc[:,0:18]
    if select_analise == 'Portfólio 3':
        df_port = df_nao_nulos.query('kmeans == @class_port3 and portfolio not in ("3")').iloc[:,0:18]
    # Inicio exploração Portfólio Exemplo
    st.markdown('**Resumo dos Leads e variáveis disponívies**')
    st.dataframe(df_port.head())
    st.markdown('**Analise Gráfica dos Leds**')
    st.markdown('**Seleção de Leads por Filtro**')

if __name__ == '__main__':
    main()
