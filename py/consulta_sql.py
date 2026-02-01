import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração visual
sns.set(style="whitegrid")

# Caminho do banco SQL
db_path = r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\Case_tecnico_Koin\sql\case_koin.db"

# Conecta ao banco
conn = sqlite3.connect(db_path)

print("Conectado ao banco com sucesso!")

# Conexão com o banco - analise de dados
db_path = r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\Case_tecnico_Koin\sql\case_koin.db"
conn = sqlite3.connect(db_path)

# Lê a tabela do banco para um DataFrame
query = "SELECT * FROM dados_tratados"
df = pd.read_sql(query, conn)

conn.close()
print("Dados carregados do banco SQL com sucesso!")

# 1. Visualizar tabelas
query_tables = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(query_tables, conn)
print("\nTabelas no banco:")
print(tables)

# Visualizar estrutura da tabela
query_schema = "PRAGMA table_info(dados_tratados);"
schema = pd.read_sql(query_schema, conn)
print("\nEstrutura da tabela dados_tratados:")
print(schema)

# Total de registros
query_count = "SELECT COUNT(*) AS total_registros FROM dados_tratados;"
df_count = pd.read_sql(query_count, conn)
print("\nTotal de registros:")
print(df_count)

# Distribuição da variável alvo
query_target = """
SELECT over30_mob3, COUNT(*) AS quantidade
FROM dados_tratados
GROUP BY over30_mob3;
"""
df_target = pd.read_sql(query_target, conn)
print("\nDistribuição da variável alvo:")
print(df_target)

# Média de valor de compra por classe
query_valor = """
SELECT over30_mob3,
       AVG(valor_compra) AS media_valor_compra
FROM dados_tratados
GROUP BY over30_mob3;
"""
df_valor = pd.read_sql(query_valor, conn)
print("\nMédia do valor de compra por classe:")
print(df_valor)

# Inadimplência por tipo de cliente
query_tipo_cliente = """
SELECT tipo_de_cliente,
       over30_mob3,
       COUNT(*) AS total
FROM dados_tratados
GROUP BY tipo_de_cliente, over30_mob3
ORDER BY tipo_de_cliente;
"""
df_tipo = pd.read_sql(query_tipo_cliente, conn)
print("\nInadimplência por tipo de cliente:")
print(df_tipo)

# Inadimplência por UF
query_uf = """
SELECT uf,
       over30_mob3,
       COUNT(*) AS total
FROM dados_tratados
GROUP BY uf, over30_mob3
ORDER BY uf;
"""
df_uf = pd.read_sql(query_uf, conn)
print("\nInadimplência por UF:")
df_uf

# Média de idade por classe
query_idade = """
SELECT over30_mob3,
       AVG(idade_cliente) AS media_idade
FROM dados_tratados
GROUP BY over30_mob3;
"""
df_idade = pd.read_sql(query_idade, conn)
print("\nMédia de idade por classe:")
df_idade

# Média de renda por classe
query_renda = """
SELECT over30_mob3,
       AVG(renda) AS media_renda
FROM dados_tratados
GROUP BY over30_mob3;
"""
df_renda = pd.read_sql(query_renda, conn)
print("\nMédia de renda por classe:")
df_renda

# Distribuição da variável alvo
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="over30_mob3", palette="Blues")
plt.title("Distribuição da variável alvo (Inadimplência)")
plt.xlabel("Classe (0 = Adimplente, 1 = Inadimplente)")
plt.ylabel("Quantidade")
plt.show()

# Valor da compra por classe
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="over30_mob3", y="valor_compra", palette="Set2")
plt.title("Valor da compra por classe")
plt.xlabel("Classe")
plt.ylabel("Valor da compra")
plt.show()

# Idade por classe
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="over30_mob3", y="idade_cliente", palette="Set3")
plt.title("Idade do cliente por classe")
plt.xlabel("Classe")
plt.ylabel("Idade")
plt.show()

# Inadimplência por tipo de cliente
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="tipo_de_cliente", hue="over30_mob3")
plt.title("Inadimplência por tipo de cliente")
plt.xlabel("Tipo de cliente")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.legend(title="Classe")
plt.show()

# Inadimplência por UF
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="uf", hue="over30_mob3")
plt.title("Inadimplência por UF")
plt.xlabel("UF")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.legend(title="Classe")
plt.show()

# Renda por classe
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="over30_mob3", y="renda", palette="coolwarm")
plt.title("Renda por classe")
plt.xlabel("Classe")
plt.ylabel("Renda")
plt.show()

# Compras por dia da semana
plt.figure(figsize=(8,4))
sns.countplot(data=df, x="dia_semana", order=df["dia_semana"].value_counts().index)
plt.title("Quantidade de compras por dia da semana")
plt.xlabel("Dia da semana")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.show()


print("Análise com gráficos finalizada!")

# Encerrar conexão

conn.close()
print("\nConexão encerrada.")