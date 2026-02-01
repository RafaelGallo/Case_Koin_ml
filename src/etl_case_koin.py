import pandas as pd
import os
from tqdm import tqdm

tqdm.pandas()  # habilita progresso no pandas

# EXTRACT (Extração)
print("Iniciando ETL...")

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

for col in tqdm(num_cols, desc="Preenchendo valores nulos numéricos"):
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

for col in tqdm(cat_cols, desc="Preenchendo valores nulos categóricos"):
    df[col] = df[col].fillna("Desconhecido")


print("Filtrando variável alvo (0 e 1)...")
df = df[df["over30_mob3"].isin([0, 1])]

print("Criando coluna dia da semana...")
df["dia_semana"] = df["data_compra"].dt.day_name()

print("Removendo colunas desnecessárias...")
df = df.drop(columns=["id_trx", "risco_validador"])

print("Transformações concluídas!")


# LOAD (Carga)
output_path = r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\Case_tecnico_Koin\data\dados_tratados.csv"

print("Salvando arquivo CSV final...")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8")

print("Arquivo salvo com sucesso!")
print(f"Local do arquivo: {output_path}")

print("ETL finalizado com sucesso!")