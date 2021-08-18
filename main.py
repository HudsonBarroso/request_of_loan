import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Criando um título para nosso projeto
st.title("Loan Simulation")
st.write('''
### Fill the fields to simulate if you can guarantee a Loan!
''')

st.sidebar.header('Fill to simulate')

gender_mapping = {
                'Male': 1,
                'Female':0,
}

married_mapping = {
                 'Yes':1,
                 'No':0,
}

dependents_mapping = {
                '0':0,
                '1':1,
                '2':1,
                '3+':1,
}

education_mapping = {
                'Graduate':1,
                'Not Graduate':0,
}

self_employed_mapping = {
                'Yes':1,
                'No':0,
}

property_mapping = {
                'Urban': 1,
                'Semi Urban': 1,
                'Rural': 0,
}


# Criando o menu lateral para inserir os dados do funcionário
def user_input_features():
    gender_feature = st.sidebar.selectbox("Gender", ("Male", "Female"))
    gender = gender_mapping[gender_feature]

    married_feature = st.sidebar.selectbox("Is Married?", ("Yes", "No"))
    married = married_mapping[married_feature]

    dependents_features = st.sidebar.selectbox("Number of Dependents", ("0", "1", "2", "3+"))
    dependents = dependents_mapping[dependents_features]

    education_feature = st.sidebar.selectbox("Education Level", ("Graduate", "Not Graduate"))
    education = education_mapping[education_feature]

    self_employed_feature = st.sidebar.selectbox("Is Self Employed?", ("Yes", "No"))
    self_employed = self_employed_mapping[self_employed_feature]

    property_feature = st.sidebar.selectbox("Property Area?", ("Urban", "Semi Urban", "Rural"))
    property_area = property_mapping[property_feature]

    applicantIncome = st.sidebar.slider("Applicant Income?", 5000, 10000, 8000)
    applicantIncome = applicantIncome / 1000
    coApplicantIncome = st.sidebar.slider("Co Applicant Income?", 0, 10000, 4000)
    coApplicantIncome = coApplicantIncome / 1000
    loanAmount = st.sidebar.slider("Loan Amount", 10, 400, 200)
    loanAmountTerm = st.sidebar.slider("Loan Amount Term", 12, 480, 300)
    creditHistory = st.sidebar.slider("Credit History", 0, 100, 80)
    creditHistoryFinal = creditHistory / 100


    data = {
        'gender': gender,
        'married': married,
        'dependents': dependents,
        'education': education,
        'self_employed': self_employed,
        'applicantIncome': applicantIncome,
        'coApplicantIncome': coApplicantIncome,
        'loanAmount': loanAmount,
        'loanAmountTerm': loanAmountTerm,
        'creditHistory': creditHistoryFinal,
        'property_area': property_area,
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# lendo o dataset de teste
promotion_test = pd.read_csv('./input/test.csv')
# concatenando os dados do usuário com os dados do dataset de teste
df = pd.concat([input_df, promotion_test], axis=0)

# selecionando a primeira linha (o valor inserido pelo usuário)
df = df[:1]

# realizando a leitura do modelo salvo
load_randomForest = pickle.load(open('loan_prediction.pkl', 'rb'))

# aplicando o modelo para realizar a previsão
prediction = load_randomForest.predict(input_df)
prediction_probability = load_randomForest.predict_proba(input_df)

st.subheader('Result')
result = np.array(['We cannot offer a loan, sorry!.','Congratulations, your request of loan was approved!'])
st.write(result[prediction][0])

st.subheader('Prevision probability')
st.write('Based on the selected data,\nyou have {0:.2f}% of chances to get a loan.'.format(
    prediction_probability[0][1] * 100))

if prediction == 1:
    st.image('./images/cash_ok.jpg', use_column_width=True)
else:
    st.image('./images/no_money.jpg', use_column_width=True)

