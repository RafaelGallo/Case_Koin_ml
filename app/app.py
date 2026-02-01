import streamlit as st
import pandas as pd
import joblib

# Carrega o modelo (caminho relativo)
model = joblib.load(r"models/modelo_turing/modelo_tuned_lightgbm_kfold.pkl")

st.title("Predição de Inadimplência")

# Inputs
age = st.number_input("Idade", 18, 100, 30)
income = st.number_input("Renda", 0.0, 100000.0, 3000.0)
loan_amount = st.number_input("Valor do empréstimo", 0.0, 50000.0, 5000.0)
credit_score = st.number_input("Score crédito", 0, 1000, 600)
num_dependents = st.number_input("Dependentes", 0, 10, 1)
employment_time = st.number_input("Tempo de emprego", 0, 40, 5)
debt_ratio = st.number_input("Debt ratio", 0.0, 1.0, 0.3)
default_history = st.selectbox("Histórico inadimplência", [0,1])
balance = st.number_input("Saldo", 0.0, 100000.0, 2000.0)
transactions = st.number_input("Transações", 0, 500, 20)
region = st.selectbox("Região", [0,1,2])
gender = st.selectbox("Sexo", [0,1])
marital_status = st.selectbox("Estado civil", [0,1,2])
education = st.selectbox("Educação", [0,1,2])

input_df = pd.DataFrame([{
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "num_dependents": num_dependents,
    "employment_time": employment_time,
    "debt_ratio": debt_ratio,
    "default_history": default_history,
    "balance": balance,
    "transactions": transactions,
    "region": region,
    "gender": gender,
    "marital_status": marital_status,
    "education": education
}])

if st.button("Prever"):
    proba = model.predict_proba(input_df)[0][1]
    st.metric("Probabilidade de inadimplência", f"{proba:.2%}")
