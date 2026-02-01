import pandas as pd
import os
from tqdm import tqdm
import sqlite3

tqdm.pandas()

print("Iniciando ETL para SQL...")

# EXTRACT (Extração)
file_path = r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\Case_tecnico_Koin\input\Cargos_salarios_CPNU2_ (1).xlsx"

print("Carregando arquivo Excel...")
df = pd.read_excel(file_path, sheet_name="Base_Dados")
print("Base carregada com sucesso!")


# TRANSFORM (Transformação)
print("Padronizando nomes das colunas...")

df.columns = (
    df.columns.str.lower()
    .str.replace(" ", "_")
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
)

print("Removendo registros duplicados...")
df = df.drop_duplicates(subset="id_trx")

print("Convertendo colunas de data e hora...")
df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")
df["hora_da_compra"] = pd.to_datetime(df["hora_da_compra"], errors="coerce").dt.hour

# Lista de colunas numéricas
num_cols = [
    "valor_compra",
    "tempo_ate_utilizacao",
    "idade_cliente",
    "renda",
    "score_email",
    "score_pessoa"
]

print("Convertendo colunas numéricas...")
for col in tqdm(num_cols, desc="Convertendo colunas numéricas"):
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Convertendo variável alvo...")
df["over30_mob3"] = df["over30_mob3"].astype("Int64")

print("Preenchendo valores ausentes (numéricos)...")
for col in tqdm(num_cols, desc="Preenchendo nulos numéricos"):
    df[col] = df[col].fillna(df[col].median())

# Lista de colunas categóricas
cat_cols = [
    "tipo_de_cliente",
    "risco_validador",
    "provedor_email",
    "produto_1",
    "produto_2",
    "produto_3",
    "uf"
]

print("Preenchendo valores ausentes (categóricos)...")
for col in tqdm(cat_cols, desc="Preenchendo nulos categóricos"):
    df[col] = df[col].fillna("Desconhecido")

print("Filtrando variável alvo (0 e 1)...")
df = df[df["over30_mob3"].isin([0, 1])]

print("Criando coluna dia da semana...")
df["dia_semana"] = df["data_compra"].dt.day_name()

print("Removendo colunas desnecessárias...")
df = df.drop(columns=["id_trx", "risco_validador"])

print("Transformações concluídas!")

# LOAD (Carga para SQL)
# Pasta onde ficará o banco SQL
sql_folder = r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\Case_tecnico_Koin\sql"
os.makedirs(sql_folder, exist_ok=True)

# Caminho do banco SQLite
db_path = os.path.join(sql_folder, "case_koin.db")

print("Conectando ao banco SQLite...")
conn = sqlite3.connect(db_path)

# Nome da tabela
table_name = "dados_tratados"

print("Salvando dados na tabela SQL...")
df.to_sql(table_name, conn, if_exists="replace", index=False)

conn.close()

print("Banco SQL criado com sucesso!")
print(f"Local do banco: {db_path}")
print(f"Tabela criada: {table_name}")

print("ETL finalizado com sucesso!")