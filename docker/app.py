import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Predição de Inadimplência",
                   page_icon="📊",
                   layout="centered")

st.title("Predição de Inadimplência")
st.write("Aplicação para estimar a probabilidade de inadimplência de um cliente.")

# Carregar modelo
MODEL_PATH = "models/modelo_turing/modelo_tuned_lightgbm_kfold.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Modelo não encontrado em: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# Layout dos inputs (colunas)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Idade", 18, 100, 30)
    income = st.number_input("Renda", 0.0, 100000.0, 3000.0)
    loan_amount = st.number_input("Valor do empréstimo", 0.0, 50000.0, 5000.0)
    credit_score = st.number_input("Score crédito", 0, 1000, 600)
    num_dependents = st.number_input("Dependentes", 0, 10, 1)
    employment_time = st.number_input("Tempo de emprego (anos)", 0, 40, 5)

with col2:
    debt_ratio = st.number_input("Debt ratio", 0.0, 1.0, 0.3)
    default_history = st.selectbox("Histórico de inadimplência", [0, 1])
    balance = st.number_input("Saldo", 0.0, 100000.0, 2000.0)
    transactions = st.number_input("Transações", 0, 500, 20)
    region = st.selectbox("Região", [0, 1, 2])
    gender = st.selectbox("Sexo", [0, 1])
    marital_status = st.selectbox("Estado civil", [0, 1, 2])
    education = st.selectbox("Educação", [0, 1, 2])

# DataFrame de entrada
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

st.subheader("Dados informados")
st.dataframe(input_df)

# Botão de previsão
if st.button("Prever risco de inadimplência"):

    # Validação simples
    if income <= 0:
        st.warning("A renda deve ser maior que zero.")
    else:
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("Resultado")

        # Barra de progresso
        st.progress(float(proba))

        st.metric("Probabilidade de inadimplência", f"{proba:.2%}")

        # Interpretação do risco
        if proba < 0.30:
            st.success("Risco baixo de inadimplência")
        elif proba < 0.60:
            st.warning("Risco médio de inadimplência")
        else:
            st.error("Risco alto de inadimplência")

        # Salvar histórico (log)
        log_df = input_df.copy()
        log_df["probabilidade"] = proba
        log_df["data_previsao"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_file = "logs_previsoes.csv"

        if not os.path.exists(log_file):
            log_df.to_csv(log_file, index=False)
        else:
            log_df.to_csv(log_file, mode="a", header=False, index=False)

        st.info("Previsão registrada no histórico.")
