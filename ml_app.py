#IMPORTANDO BIBLIOTECAS
#manipulação de dados
import pandas as pd

#visualização de dados dinâmicos
import streamlit as st

#machine learn
from sklearn.metrics import accuracy_score

#função para separar data_set em treinamento e validação
from sklearn.model_selection import train_test_split

#função para chamar o modelo de Arvore de Decisões como MachineLearn 
from sklearn.tree import DecisionTreeClassifier

#Título do aplicativo
st.write("""
**Prevendo Diabetes**\n
Utilização de Machine Learning para Previsão de Diabetes em Pacientes.\n
Fonte: PIMA - INDIA (Kaggle)         
         """)

#dataset utilizado para treinamento do modelo
df = pd.read_csv('data_base/diabetes.csv')

#CABEÇALHO
st.subheader('Informações do Paciente')

#Nome do usuário
user_input = st.sidebar.text_input('Digite Nome do Paciente:')
st.write('Paciente: ', user_input)

#Dados de entrada para aplicar ao modelo
x = df.drop(labels='Outcome',axis=1)
y = df['Outcome']

#Separando dados em treinamento e teste

#Set_Train = 42 força o sistema sempre 
#escolher os mesmos dados para treinamento
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,random_state=42)

#dados dos usuários com a função

#Todas as entradas, são características presentes na base de dados
#A qual aplicamos nossa Árvore de Decisão para classificar Diabetes

def get_User_Data():
    #Definindo barra de ajustes das informações do paciente
    #que irá ser testado para diabetes

    #Se o paciente está em período de gravidez 
    pregnancies = st.sidebar.slider('Gravidez',0,15,1)
    
    #Indicação do nível de glicose no sangue do paciente
    glucose = st.sidebar.slider('Glicose',0,200,110)

    #Pressão sanguínea do paciente 
    blood_pressure = st.sidebar.slider('Pressão Sanguínea',0,122,72)

    #Espressura da pele do paciente
    skin_thickness = st.sidebar.slider('Espressura da pele',0,99,20)

    #Níveis de Insulina
    insulin = st.sidebar.slider('Insulina',0,900,30)

    #Índice de Massa Corporal do Paciente
    bmi = st.sidebar.slider('Índice de Massa Corporal',0.0,70.0,15.0)
    
    #Histórico Familiar de Diabetes do Paciente
    dpf = st.sidebar.slider('Histórico Familiar de Diabetes',0.0,3.0,0.0)

    #Idade do paciente, a princípio os dados treinados possuem parâmetros
    #para pacientes acima de 15 anos
    age = st.sidebar.slider('Idade',15,100,21)

    #salvando as informações do usuario em uma variável que 
    #será carregada no nosso modelo de machine learn 
    #para classificar diabetes

    user_data = {'Pregnancies':pregnancies,
                 'Glucose':glucose,
                 'BloodPressure':blood_pressure,
                 'SkinThickness':skin_thickness,
                 'Insulin':insulin,
                 'BMI':bmi,
                 'DiabetesPedigreeFunction':dpf,
                 'Age':age
                 }


    features = pd.DataFrame(user_data,index=[0])

    return features

requisicao_Paciente = get_User_Data()

#Construção do Gráfico
graph_Bar = st.bar_chart(requisicao_Paciente)

st.subheader('Dados do Usuário')
st.write(requisicao_Paciente)

#realizando a classificação dos nossos dados
#Aplicamos uma arvore de decisão por critério de Entropia, com até 3 níveis
dtc = DecisionTreeClassifier(criterion='entropy',max_depth=3)

#passando conjunto de dados para treinar o modelo
dtc.fit(x_train,y_train)

#Calculando a acurácia do modelo
st.subheader('Acurácia do Modelo:')

#estamos escrevendo na tela a acurácia do modelo 
st.write(str(round(accuracy_score(y_test,dtc.predict(x_test))*100,2))+'%')

#Previsão
#guardando uma variável o valor 0 ou 1
#de acordo com o resutado da arvore de decisão
#onde ela aplica os dados do paciente em nosso modelo 
#e retorna 0 para sem diabetes e 1 para com diabetes
prediction = dtc.predict(requisicao_Paciente)

st.subheader('Previsão: ')
if prediction==1:
    st.write('Diabetes Confirmada')
else:
    st.write('Sem Diabetes')
