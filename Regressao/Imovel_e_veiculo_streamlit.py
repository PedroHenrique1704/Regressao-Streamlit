import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

import os
import io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# Configuração da página

st.set_page_config(
     page_title="Prob imovel/veiculo",
     page_icon=".\input\icon\icon.png"
)

# CSV

renda = pd.read_csv('./input/csv/renda_ajustado.csv')
renda.set_index('id_cliente', inplace=True)
renda.sort_index(inplace=True)

# Configuração CSS

     # Arquivo css

def carregar_css(arquivo_css):
    with open(arquivo_css) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

carregar_css("./input/css/estilo.css")

     # Selectbox sem rótulo

def css_rotulo():
    st.markdown(
        """
        <style>
        /* Ocultar o rótulo da selectbox */
        .stSelectbox label {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

css_rotulo()


# Funções

     # Estatisticas Sidebar
def estatisticas_colunas(coluna):
    estatisticas = renda[[coluna]].value_counts()
    return estatisticas
    
def estatisticas_colunas_variaveis(coluna):
    estatisticas_min = round(renda[[coluna]].min(),2)
    estatisticas_max = round(renda[[coluna]].max(),2)
    estatisticas_mean = round(renda[[coluna]].mean(),2)
    return(estatisticas_min,estatisticas_max,estatisticas_mean)

  

     

# Sidebar
st.sidebar.write('# Informações *Dataframe*')
     # Informação Colunas
opcao_selecionada = st.sidebar.selectbox(
    "",
    ["Selecione uma coluna", "id_cliente", "data_ref","renda","tipo_renda",
     "tempo_emprego","sexo","idade","educacao","estado_civil","qtd_filhos",
     "genitor_solteiro","posse_de_imovel","posse_de_veiculo","posse_imovel_e_veiculo","tipo_residencia","qt_pessoas_residencia"]
)

     #Id cliente
if opcao_selecionada == "id_cliente":
     st.sidebar.write("Funciona como índice para o DataFrame, uma vez que o ID é único para cada cliente.")
     st.sidebar.markdown("---")
     st.sidebar.markdown("Alguns **IDs** foram cortados no processo de *pré e pós poda*, por estarem com irrelugaridades como *dados faltantes* ou serem tratados como **outliers**")
     st.sidebar.markdown("---")

     linha_df, coluna_df = renda.shape

     st.sidebar.write("Quantidade de linhas no dataframe: ",linha_df)
     st.sidebar.write("Quantidade de colunas no dataframe: ",coluna_df)
     
     
     #data_ref
elif opcao_selecionada == "data_ref":
     st.sidebar.write("O dia que foi feito a analise, indo de janeiro 2015 à março 2016")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Datetime")

     #Renda
elif opcao_selecionada == "renda":
     st.sidebar.write("Representa a renda mensal dos clientes.")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Float")
     st.sidebar.markdown("---")
 
     minima, maxima, media = estatisticas_colunas_variaveis('renda')
    
     minima_str = f'{minima}'
     maxima_str = f'{maxima}'
     media_str = f'{media}'

     st.sidebar.write("Renda mínima: ",minima_str[6:-14])
     st.sidebar.write("Renda máxima: ",maxima_str[6:-14])
     st.sidebar.write("Renda média: ",media_str[6:-14])

     #Tipo_renda
elif opcao_selecionada == "tipo_renda":
     st.sidebar.write ("Representa o meio que o cliente utiliza para receber sua renda mensal")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: String")
     st.sidebar.markdown("---")
     st.sidebar.write(estatisticas_colunas('tipo_renda'))

     #Tempo_emprego
elif opcao_selecionada == "tempo_emprego":
    
     st.sidebar.write ("Representa a quanto tempo o cliente não se encontra desempregado (em meses)")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Float")
     st.sidebar.markdown("---")

     minima, maxima, media = estatisticas_colunas_variaveis('tempo_emprego')
    
     minima_str = f'{minima}'
     maxima_str = f'{maxima}'
     media_str = f'{media}'

     
     st.sidebar.write("Tempo mínimo: ",minima_str[6:-14])
     st.sidebar.write("Tempo máximo: ",maxima_str[6:-14])
     st.sidebar.write("Tempo médio: ",media_str[6:-14])
     
     #Sexo
elif opcao_selecionada == "sexo":
     st.sidebar.write("Representa o gênero dos clientes.")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: String")
     st.sidebar.markdown("---")
     st.sidebar.write(estatisticas_colunas('sexo'))

     #Idade
elif opcao_selecionada == "idade":
     st.sidebar.write ("Representa a Idade do cliente")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Int")
     st.sidebar.markdown("---")

     minima, maxima, media = estatisticas_colunas_variaveis('idade')
    
     minima_str = f'{minima}'
     maxima_str = f'{maxima}'
     media_str = f'{media}'

     st.sidebar.write("Idade mínima: ",minima_str[5:-13])
     st.sidebar.write("Idade máxima: ",maxima_str[6:-13])
     st.sidebar.write("Idade média:  ",media_str[6:-14])

     #educacao
elif opcao_selecionada == "educacao":
     st.sidebar.write("Representa o nível de escolaridade dos clientes")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: String")
     st.sidebar.markdown("---")
     st.sidebar.write(estatisticas_colunas('educacao'))


  #estado_civil
elif opcao_selecionada == "estado_civil":
     st.sidebar.write("Representa o estado civil do cliente")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: String")
     st.sidebar.markdown("---")
     st.sidebar.write(estatisticas_colunas('estado_civil'))


  #qtd_filhos
elif opcao_selecionada == "qtd_filhos":
     st.sidebar.write("Representa a quantidade de filhos que o cliente possui")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Int")
     st.sidebar.markdown("---")
     minima, maxima, media = estatisticas_colunas_variaveis('qtd_filhos')
    
     minima_str = f'{minima}'
     maxima_str = f'{maxima}'
     media_str = f'{media}'

     st.sidebar.write("Quantidade mínima de filhos: ",minima_str[10:-13])
     st.sidebar.write("Quantidade máxima de filhos: ",maxima_str[10:-13])
     st.sidebar.write("Quantidade média de filhos:",media_str[10:-14])

elif opcao_selecionada == "genitor_solteiro":
     st.sidebar.write("Caso o cliente esteja solteiro e possua filhos")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Bool")
     st.sidebar.markdown("---")
     st.sidebar.write(estatisticas_colunas('genitor_solteiro'))

     #posse_imovel
elif opcao_selecionada == "posse_de_imovel":
     st.sidebar.write("Caso o cliente possua um ou mais imóveis")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Bool")
     st.sidebar.markdown("---")
     st.sidebar.write(estatisticas_colunas('posse_de_imovel'))


     #Posse_veiculo
elif opcao_selecionada == "posse_de_veiculo":
     st.sidebar.write("Caso o cliente possua um ou mais  veículos")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Bool")
     st.sidebar.markdown("---")
     st.sidebar.write(estatisticas_colunas('posse_de_veiculo'))

  #Posse_imovel_e_veiculo
elif opcao_selecionada == "posse_imovel_e_veiculo":
     st.sidebar.write("Caso o cliente possua um ou mais imóveis e também possua um ou mais veículos")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Bool")
     st.sidebar.markdown("---")
     st.sidebar.write(estatisticas_colunas('posse_imovel_e_veiculo'))

     #tipo_residencia
elif opcao_selecionada == "tipo_residencia":
     st.sidebar.write("O tipo de residencia que o cliente vive")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: String")
     st.sidebar.markdown("---")
     st.sidebar.write(estatisticas_colunas('tipo_residencia'))

     #qt_pessoas_residencia
elif opcao_selecionada == "qt_pessoas_residencia":
     st.sidebar.write("Quantidade de pessoas que vivem na mesma residencia do cliente")
     st.sidebar.markdown("---")
     st.sidebar.write("Tipo: Float")
     st.sidebar.markdown("---")
     minima, maxima, media = estatisticas_colunas_variaveis('qt_pessoas_residencia')
    
     minima_str = f'{minima}'
     maxima_str = f'{maxima}'
     media_str = f'{media}'

     st.sidebar.write("Quantidade mínima de residentes: ",minima_str[21:-14])
     st.sidebar.write("Quantidade máxima de residentes: ",maxima_str[21:-14])
     st.sidebar.write("Quantidade média de residentes:",media_str[21:-14])

    


# Titulo
st.write('# Probabilidade: Regressão Logística')

# Criar tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Streamlit", "Dataframe","Problema preposto","Solução","Testes","Downloads"])

with tab1:
    st.markdown("## Funções:")
    st.write("### **Tabelas** \nUtilize as tabelas aqui presentes para navegar pelo streamlit e compreender tudo que esta disponível.")
    st.markdown("### **Barra Lateral** \nUtilize a barra lateral para entender as colunas que compõem o *dataframe*.")
    st.markdown("---")
    st.markdown("*Esse streamlit foi criado a partir do notebook **imovel_e_veiculo.ipynb** que se encontra na pasta, o dataframe que será utilizado ja passou pelo processo de pré e pós poda.*")

with tab2:
   
    # Pegar os valores de index mínimo e máximo
    min_index, max_index = st.slider(
    'Filtrar por intervalo entre índices:',
    min_value=int(renda.index.min()),
    max_value=int(renda.index.max()),
    value=(int(renda.index.min()), int(renda.index.max()))
)

    # Filtrar o dataframe com base no intervalo de índices selecionado
    df_filtrado = renda.loc[min_index:max_index]

    # Exibir o dataframe filtrado
    st.write(f"Mostrando dados de índice {min_index} a {max_index}:")
    st.dataframe(df_filtrado)

    multi_tab2 = '''*Para melhor entendimento das colunas utilize a barra lateral.*  
    *Download do dataframe disponível na aba **Downloads.***'''
    st.markdown(multi_tab2)


with tab3:
     st.markdown("### Quais variáveis possuem mais influência para que uma pessoa possua tanto um carro como um imóvel")
     st.markdown("---")
     st.markdown("Essa análise permitirá entender quais **fatores (variáveis)** têm maior impacto para que uma pessoa possua **ao menos um imóvel e ao menos um veículo**. Com essa informação, uma empresa pode tomar decisões informadas sobre **marketing, ofertas de produtos e segmentação de clientes.**")







with tab4:
     st.write("Foi feito uma regressão utilizando **20%** do dataframe como teste, e gerado um gráfico para saber a importância das variáveis, abaixo temos a *Matriz de confusão* e 2 botões de download (gráfico e avaliações).")
     st.write("*Random-state utilizado: **27***")
     
     
# Criando variáveis dummies para colunas categóricas
     renda_dummies = pd.get_dummies(renda, columns=['sexo', 'tipo_renda', 'estado_civil', 'educacao', 'tipo_residencia'], drop_first=True)

    # Separando as variáveis de features e target
     X = renda_dummies.drop(['data_ref', 'posse_imovel_e_veiculo'], axis=1)  # Features
     y = renda_dummies['posse_imovel_e_veiculo']  # Target

    # Dividir os dados em treino e teste
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

    # Criando e treinando o modelo de regressão logística
     model = LogisticRegression(max_iter=1000)
     model.fit(X_train, y_train)

    # Fazendo previsões
     y_pred = model.predict(X_test)
     
     
    # Coeficientes do modelo
     coefficients = pd.DataFrame(model.coef_[0], index=X.columns, columns=['coeficiente'])
     coefficients_sorted = coefficients.sort_values(by='coeficiente', ascending=False)
     #st.write("### Coeficientes do Modelo de Regressão Logística:")
     #st.dataframe(coefficients_sorted)
     


    # Visualização dos coeficientes
     coeficientes = coefficients.sort_values(by='coeficiente')
     plt.figure(figsize=(10, 6))
     sns.barplot(x=coeficientes['coeficiente'], y=coeficientes.index)
     plt.title('Impacto das variaveis para possuir imóvel e veículo')
     plt.xlabel('Coeficiente')
     plt.ylabel('Variáveis')
     plt.axvline(0, color='grey', linestyle='--')  # Linha vertical em 0



     
     col10, col20 = st.columns([1,1])

     with col10:
          st.markdown("**Matriz de confusão**")
          # Avaliando o modelo
          st.write(confusion_matrix(y_test, y_pred))
          conf_matrix = confusion_matrix(y_test, y_pred)
          

     with col20:
         
          st.markdown("")
          st.markdown("")
          st.markdown("")

          # Botão para download do gráfico
          buffer = io.BytesIO()
          plt.savefig(buffer, format='png')
          buffer.seek(0)

          st.download_button(
              label="Baixar gráfico",
              data=buffer,
              file_name='grafico_coeficientes.png',
              mime='image/png'
         )
          
          #Download Txt


          avaliacoes_texto = f"""Matriz de Confusão:\n{conf_matrix}\n\nRelatório de Classificação:\n{classification_report(y_test, y_pred)}\n\nCoeficientes:\n{coefficients.to_string()}"""

          st.download_button(
               label="Baixar avaliações como .txt",
               data=avaliacoes_texto,
               file_name='avaliacoes_modelo.txt',
               mime='text/plain'
          )






    # Exibir o gráfico no Streamlit
     st.pyplot(plt)





    






with tab5:
     st.markdown("Aqui você pode utilizar a mesma regressão, mas com um certo nível de personalização, podendo *remover variáveis da regressão* e trocar o *random-state*")

     renda_dummies = pd.get_dummies(renda, columns=['sexo', 'tipo_renda', 'estado_civil', 'educacao', 'tipo_residencia'], drop_first=True)
# Lista de colunas para dropar, sem incluir 'data_ref' e 'posse_imovel_e_veiculo'
     colunas_para_dropar = renda_dummies.columns.difference(['data_ref', 'posse_imovel_e_veiculo'])

# Multiselect para escolher colunas a serem dropadas
     colunas_selecionadas = st.multiselect(
    'Escolha as colunas para remover da regressão:',
    colunas_para_dropar
)
     
# Random-estate
     random_state = st.slider('Escolha o valor de random_state:', min_value=1, max_value=99, value=27)


# Remover sempre 'data_ref' e 'posse_imovel_e_veiculo' e as colunas selecionadas pelo usuário
     X = renda_dummies.drop(['data_ref', 'posse_imovel_e_veiculo'] + colunas_selecionadas, axis=1)
     y = renda_dummies['posse_imovel_e_veiculo']

# Dividir os dados em treino e teste
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Criar e treinar o modelo de regressão logística
     model = LogisticRegression(max_iter=1000)
     model.fit(X_train, y_train)

# Fazer previsões
     y_pred = model.predict(X_test)

# Avaliar o modelo
     
     

# Exibir coeficientes do modelo
     coefficients = pd.DataFrame(model.coef_[0], index=X.columns, columns=['coeficiente']).sort_values(by='coeficiente')

# Visualizar os coeficientes em um gráfico
   
     plt.figure(figsize=(10, 6))
     sns.barplot(x=coefficients['coeficiente'], y=coefficients.index)
     plt.title('Impacto das variaveis para possuir imóvel e veículo')
     plt.xlabel('Coeficiente')
     plt.ylabel('Variáveis')
     plt.axvline(0, color='grey', linestyle='--')


     col1, col2 = st.columns([1, 1])
     with col1:
          st.markdown("**Matriz de confusão**")
          st.write(confusion_matrix(y_test, y_pred))
          conf_matrix2 = confusion_matrix(y_test, y_pred)

     with col2:
          st.markdown("")
          st.markdown("")
          st.markdown("")
          # Botão para download do gráfico
          buffer = io.BytesIO()
          plt.savefig(buffer, format='png')
          buffer.seek(0)

          st.download_button(
               label="Baixar gráfico",
               data=buffer,
               file_name='grafico_coeficientes_ajustado.png',
               mime='image/png'
          )
          
          #Download Txt

# Preparar dados para baixar avaliações como .txt
          avaliacoes_texto = f"""Matriz de Confusão:\n{conf_matrix2}\n\nRelatório de Classificação:\n{classification_report(y_test, y_pred)}\n\nCoeficientes:\n{coefficients.to_string()}"""

# Botão para download das avaliações como arquivo de texto
          st.download_button(
               label="Baixar avaliações como .txt",
               data=avaliacoes_texto,
               file_name='avaliacoes_modelo_ajustado.txt',
               mime='text/plain'
          )








     st.pyplot(plt)


     #Download gráfico





with tab6:
    # Criar 4 colunas
    col1, col2, col3, col4 = st.columns(4)

    # Cabeçalhos na primeira linha
    with col1:
        st.markdown("<h3 style='text-align: center;'>CSV:</h3>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h3 style='text-align: center;'>Notebook:</h3>", unsafe_allow_html=True)
    with col3:
        st.markdown("<h3 style='text-align: center;'>Streamlit:</h3>", unsafe_allow_html=True)
    with col4:
        st.markdown("<h3 style='text-align: center;'>HTML:</h3>", unsafe_allow_html=True)

    # Adicionar botões na segunda linha
    with col1:
        st.download_button(
            label="Original",
            data="./input/csv/previsao_de_renda.csv",
            file_name='previsao_de_renda.csv',
            mime='text/csv'
        )
    with col2:
        st.download_button(
            label="Notebook",
            data="imovel_e_veiculo.ipynb",
            file_name='imovel_e_veiculo.ipynb',
            mime='text/x-python'
        )
    with col3:
        st.download_button(
            label="Streamlit",
            data="Imovel_e_veiculo_streamlit.py",
            file_name='imovel_e_veiculo.py',
            mime='text/x-python'
        )
    with col4:
        st.download_button(
            label="Original",
            data="./output/renda_analisys.html",
            file_name='Analise_de_renda.html',
            mime='text/html'
        )

    # Terceira linha para os próximos botões
    with col1:
        st.download_button(
            label="Pós poda",
            data="./input/csv/renda_ajustado.csv",
            file_name='previsao_de_renda_ajustado.csv',
            mime='text/csv'
        )
    with col2:
       st.markdown("")
    with col3:
       st.download_button(
            label="CSS",
            data=".\input\css\estilo.css",
            file_name='css.css',
            mime='text/css'
        )

    with col4:
          
       st.download_button(
            label="Ajustado.html",
            data="./output/renda_analisys_ajustada.html",
            file_name='Analise_de_renda_ajustada.html',
            mime='text/html'
        )


    