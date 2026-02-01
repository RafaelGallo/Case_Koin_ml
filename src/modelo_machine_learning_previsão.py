import os
import joblib
from tqdm import tqdm

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Base dados
file_path = r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\Case_tecnico_Koin\input\Cargos_salarios_CPNU2_ (1).xlsx"

#
df = pd.read_excel(file_path, sheet_name="Base_Dados")

# Padronização dos nomes das colunas (minúsculas, sem espaços e sem acentos)
df.columns = (
    df.columns.str.lower()
    .str.replace(" ", "_")
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
)

# Remoção de registros duplicados com base no id da transação
df = df.drop_duplicates(subset="id_trx")

# Conversão da coluna de data para formato datetime
df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")

# Conversão da hora da compra para valor numérico (hora)
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

# Conversão das colunas numéricas para tipo numérico
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Conversão da variável alvo para inteiro
df["over30_mob3"] = df["over30_mob3"].astype("Int64")

# Tratamento de valores ausentes nas colunas numéricas usando a mediana
for col in num_cols:
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

# Tratamento de valores ausentes nas colunas categóricas
for col in cat_cols:
    df[col] = df[col].fillna("Desconhecido")

# Garantir que a variável alvo contenha apenas valores 0 ou 1
df = df[df["over30_mob3"].isin([0, 1])]

# Criação da variável dia da semana a partir da data da compra
df["dia_semana"] = df["data_compra"].dt.day_name()

# Limpeza de dados
df = df.drop(columns=["id_trx"])
df = df.drop(columns=["risco_validador"])

# Ver quantidade de valores nulos por coluna
null_counts = df.isnull().sum()

# Criar tabela com quantidade e percentual de nulos
null_table = pd.DataFrame({
    "Qtd_Nulos": null_counts,
    "Percentual_%": (null_counts / len(df)) * 100
}).sort_values(by="Percentual_%", ascending=False)

# Remover a coluna que possui 100% de valores nulos
df = df.drop(columns=["tempo_ate_utilizacao"])

# Lista de colunas numéricas
num_cols = [
    "valor_compra",
    "idade_cliente",
    "renda",
    "score_email",
    "score_pessoa"
]

# Imputação dos valores nulos usando a média
for col in num_cols:
    media = df[col].mean()
    df[col] = df[col].fillna(media)

# Verificar se ainda existem valores nulos
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Tratar com método do Z-Score
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

#
num_cols = num_cols.drop("over30_mob3")

# Calcular Z-score
z_scores = np.abs(stats.zscore(df[num_cols]))

# dataframe indicando outliers
outliers_mask = (z_scores > 3)

outliers_count = outliers_mask.sum(axis=0)

outliers_df = pd.DataFrame({
    "Variavel": num_cols,
    "Qtd_Outliers": outliers_count,
    "Percentual_Outliers": (outliers_count / len(df)) * 100
})

outliers_df.sort_values("Qtd_Outliers", ascending=False)

# Identificar colunas categóricas
cat_cols = df.select_dtypes(include=["object", "bool"]).columns

#
le = LabelEncoder()

#
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

#
target = "over30_mob3"

#
cols_drop = ["over30_mob3", "id_trx", "data_compra"]  
X = df.drop(columns=cols_drop, errors="ignore")
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Antes do SMOTE:")
print(y_train.value_counts())

print("Depois do SMOTE:")
print(pd.Series(y_train_res).value_counts())

# Transformação legítima
df["ano"] = df["data_compra"].dt.year
df["mes"] = df["data_compra"].dt.month
df["dia"] = df["data_compra"].dt.day
df["dia_semana"] = df["data_compra"].dt.weekday

df = df.drop(columns=["data_compra"])

# Função para encontrar melhor n_neighbors (KNN)
def find_best_k_knn(X, y, k_range=range(3, 21), cv=5):
    results = []
    
    for k in tqdm(k_range, desc="Testando valores de k (KNN)"):
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        
        results.append({
            "k": k,
            "mean_accuracy": scores.mean()
        })
    
    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df["mean_accuracy"].idxmax()]
    
    return results_df, best_k

# Função para encontrar melhor max_depth (Decision Tree)
def find_best_depth_decision_tree(X, y, depth_range=range(2, 21), cv=5):
    results = []
    
    for depth in tqdm(depth_range, desc="Testando max_depth (Decision Tree)"):
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42,
            class_weight="balanced"
        )
        
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        
        results.append({
            "max_depth": depth,
            "mean_accuracy": scores.mean()
        })
    
    results_df = pd.DataFrame(results)
    best_depth = results_df.loc[results_df["mean_accuracy"].idxmax()]
    
    return results_df, best_depth

# Função para Random Forest (n_estimators)
def find_best_n_estimators_rf(X, y, n_range=[50,100,200,300,500], cv=5):
    results = []
    
    for n in tqdm(n_range, desc="Testando n_estimators (Random Forest)"):
        model = RandomForestClassifier(
            n_estimators=n,
            random_state=42,
            class_weight="balanced"
        )
        
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        
        results.append({
            "n_estimators": n,
            "mean_accuracy": scores.mean()
        })
    
    results_df = pd.DataFrame(results)
    best_n = results_df.loc[results_df["mean_accuracy"].idxmax()]
    
    return results_df, best_n

knn_results, best_knn = find_best_k_knn(X_train_res, y_train_res)
dt_results, best_dt = find_best_depth_decision_tree(X_train_res, y_train_res)
rf_results, best_rf = find_best_n_estimators_rf(X_train_res, y_train_res)

best_k = int(best_knn["k"])
best_depth = int(best_dt["max_depth"])
best_rf_n = int(best_rf["n_estimators"])

print("Melhor k:", best_k)
print("Melhor max_depth:", best_depth)
print("Melhor n_estimators RF:", best_rf_n)

from collections import Counter

counter = Counter(y_train_res)

scale_pos_weight = counter[0] / counter[1]

%%time

models = {"Naive Bayes": GaussianNB(),
          
          "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
          
          "Random Forest": RandomForestClassifier(n_estimators=best_rf_n,
                                            random_state=42,
                                            class_weight="balanced"),
          
          "Decision Tree": DecisionTreeClassifier(random_state=42,
                                            max_depth=best_depth,
                                            class_weight="balanced"),
          
          "Gradient Boosting": GradientBoostingClassifier(random_state=42),
          
          "KNN": KNeighborsClassifier(n_neighbors=best_k),
          
          "XGBoost": XGBClassifier(n_estimators=300,
                                   learning_rate=0.05,
                                   max_depth=6,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   eval_metric="logloss",
                                   random_state=42,
                                   scale_pos_weight=scale_pos_weight,
                                   use_label_encoder=False),
    
            "LightGBM": LGBMClassifier(n_estimators=300,
                               learning_rate=0.05,
                               num_leaves=31,
                               random_state=42,
                               class_weight="balanced"),
    
            "CatBoost": CatBoostClassifier(iterations=300,
                                   learning_rate=0.05,
                                   depth=6,
                                   random_state=42,
                                   verbose=0,
                                   auto_class_weights="Balanced")}

#
results = []

#
for name, model in tqdm(models.items(), desc="Treinando modelos"):
    
    # Treinar
    model.fit(X_train_res, y_train_res)
    
    # Prever
    y_pred = model.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    #
    results.append({"Modelo": name,
                    "Accuracy": acc,
                    "Recall": recall,
                    "F1-score": f1})
    
    print(f"\nModelo: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall (Inadimplente): {recall:.4f}")
    print(f"F1-score (Inadimplente): {f1:.4f}")

# DataFrame final
results_df = pd.DataFrame(results).sort_values("Recall", ascending=False)

# Feature Importance modelos
ig, axes = plt.subplots(2, 3, figsize=(18,10))
axes = axes.flatten()

i = 0

for name, model in models.items():
    
    if hasattr(model, "feature_importances_"):
        
        importances = model.feature_importances_
        features = X_train_res.columns
        
        df_imp = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)
        
        sns.barplot(
            data=df_imp,
            x="Importance",
            y="Feature",
            ax=axes[i],
            palette="Blues"
        )
        
        axes[i].set_title(f"Feature Importance - {name}")
        axes[i].set_xlabel("Importância")
        axes[i].set_ylabel("Variável")
        
        i += 1

# Remove gráficos vazios
for j in range(i, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 4, figsize=(15,10))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )
    
    ax.set_title(name)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")

# Remove eixos vazios caso sobrem subplots
for i in range(len(models), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# Classification Report (um por modelo)
for name, model in models.items():
    
    y_pred = model.predict(X_test)
    
    print("="*60)
    print(f"Classification Report - {name}")
    print("="*60)
    
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Adimplente (0)", "Inadimplente (1)"],
        zero_division=0
    ))

fig, axes = plt.subplots(3, 4, figsize=(15,10))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    
    # Probabilidades
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_proba = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1], [0,1], "k--")
    
    ax.set_title(name)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

# Remove subplots vazios
for i in range(len(models), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()    

# Curva ROC em um único gráfico (todas juntas)
plt.figure(figsize=(8,6))

for name, model in models.items():
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_proba = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curvas ROC - Comparação dos Modelos")
plt.legend()
plt.show()

# Salvando os modelos para produção
save_path = r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\Case_tecnico_Koin\models"
os.makedirs(save_path, exist_ok=True)

trained_models = {}

for name, model in tqdm(models.items(), desc="Treinando e salvando modelos"):
       
    # Guardar em dicionário
    trained_models[name] = model
    
    # Nome do arquivo
    filename = name.replace(" ", "_").lower() + ".pkl"
    filepath = os.path.join(save_path, filename)
    
    # Salvar modelo
    joblib.dump(model, filepath)
    
    print(f"Modelo salvo: {filepath}")

# Garantir formato correto
X_test = pd.DataFrame(X_test, columns=X_train_res.columns)
y_test = np.array(y_test).ravel()

trained_models = {}
metrics = []

# Treinar modelos
for name, model in models.items():
    print(f"Treinando {name}...")
    model.fit(X_train_res, y_train_res)
    trained_models[name] = model

# Avaliar modelos
for name, model in trained_models.items():
    
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    metrics.append({
        "Modelo": name,
        "Accuracy": acc,
        "Recall": recall,
        "F1-score": f1
    })

# DataFrame
metrics_df = pd.DataFrame(metrics).sort_values("Recall", ascending=False)

# Destacar melhor modelo
best_recall = metrics_df["Recall"].max()

def highlight_best(row):
    if row["Recall"] == best_recall:
        return ["background-color: yellow"] * len(row)
    else:
        return [""] * len(row)

metrics_df.style.apply(highlight_best, axis=1)
metrics_df    

%%time

# Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline
pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", LGBMClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

# Hyperparameter space
param_dist = {
    "model__n_estimators": [300, 500, 800],
    "model__learning_rate": [0.01, 0.03, 0.05],
    "model__num_leaves": [31, 63, 127],
    "model__max_depth": [-1, 10, 20],
    "model__min_child_samples": [20, 50, 100],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0]
}

# RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=40,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

# tqdm progress bar
total_fits = 40 * cv.get_n_splits()

with tqdm_joblib(tqdm(desc="Hyperparameter Tuning (LightGBM)", total=total_fits)):
    search.fit(X, y)

# Resultados
print("="*60)
print("Melhores parâmetros:")
print(search.best_params_)
print(f"Melhor ROC AUC (CV): {search.best_score_:.4f}")

# Salvar modelo
joblib.dump(
    best_model,
    r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\Case_tecnico_Koin\models\modelo_turing\modelo_tuned_lightgbm_kfold.pkl"
)

print("Modelo salvo com sucesso.")

# Resultados
print("="*60)
print("MELHOR MODELO (LightGBM + K-Fold)")
print("="*60)
print("Melhores parâmetros:")
print(search.best_params_)
print(f"Melhor ROC AUC (CV): {search.best_score_:.4f}")

#
best_model = search.best_estimator_

# Avaliação com threshold
threshold = 0.30

y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# pegar o melhor modelo do tuning
best_lgb = search.best_estimator_

threshold = 0.30

# Probabilidades
y_proba = best_lgb.predict_proba(X_test)[:, 1]

# Predição com threshold
y_pred = (y_proba >= threshold).astype(int)

#
print("ROC AUC Teste:", roc_auc_score(y_test, y_proba))

# Teste de thresholds
print("\nAvaliação por threshold:")
thresholds = np.arange(0.1, 0.9, 0.1)

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    acc = accuracy_score(y_test, y_pred_t)
    recall = recall_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t)
    
    print(f"Threshold {t:.2f} | Accuracy={acc:.3f} | Recall={recall:.3f} | F1={f1:.3f}")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))    

#
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

#
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"ROC AUC: {auc:.3f}")

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:,1]

cm = confusion_matrix(y_test, y_pred)

labels = ["Adimplente (0)", "Inadimplente (1)"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.title("Matriz de Confusão - LightGBM")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# Melhor modelo
best_lgb = search.best_estimator_

# Probabilidades
y_proba = best_lgb.predict_proba(X_test)[:, 1]

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

# Threshold escolhido
threshold = 0.30
idx = np.argmin(np.abs(thresholds - threshold))

# Subplots
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Gráfico 1: ROC padrão
axes[0].plot(fpr, tpr, color="orange", label=f"AUC = {roc_auc:.3f}")
axes[0].plot([0,1],[0,1],"--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Curva ROC - LightGBM")
axes[0].legend()
axes[0].grid()

# Gráfico 2: ROC com threshold
axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
axes[1].scatter(fpr[idx], tpr[idx], color="red", label=f"Threshold = {threshold}")
axes[1].plot([0,1],[0,1],'--', color="gray")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("Curva ROC - LightGBM (Threshold)")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()