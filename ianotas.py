

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


dataframe = pd.read_excel('Analíse_relaçao_economiasEgastos.xlsx')


features = ['Água', 'Celular', 'Luz', 'Internet', 'Aluguel', 'Cartão', 'Lazer', 'Apostas', 'Emprego Fixo', 'Bicos']
dataframe = dataframe[features]


min_values = dataframe.min()
max_values = dataframe.max()


scaler = MinMaxScaler()
dataframe_normalizadoGeral = pd.DataFrame(scaler.fit_transform(dataframe), columns=features)


k_optimal = 4


kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
dataframe_normalizadoGeral['Cluster'] = kmeans.fit_predict(dataframe_normalizadoGeral)


centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
print("Centroides dos clusters:")
print(centroids)


descricao_clusters = {
    0: "Econômico: Gasta pouco e economiza bem, mas falta investir.",
    1: "Equilibrado: Gasta de forma controlada e mantém uma boa organização financeira.",
    2: "Gastador: Gasta muito e pode ter desperdícios, precisa de mais controle financeiro.",
    3: "Corrido: Gasta muito com finanças necessárias, não desperdiça e tem pouco dinheiro sobrando."
}


knn = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(
    dataframe_normalizadoGeral[features], dataframe_normalizadoGeral['Cluster'], test_size=0.2, random_state=42
)
knn.fit(X_train, y_train)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

def gerar_usuario_aleatorio():
    return {feature: np.random.randint(min_values[feature], max_values[feature] + 1) for feature in features}

def classificar_novo_usuario(novo_usuario):
    novo_usuario_df = pd.DataFrame([novo_usuario], columns=features)
    novo_usuario_normalizado = scaler.transform(novo_usuario_df)
    cluster_predito = knn.predict(novo_usuario_normalizado)[0]


    orcamento_total = novo_usuario['Emprego Fixo'] + novo_usuario['Bicos']
    despesas = {k: v for k, v in novo_usuario.items() if k not in ['Emprego Fixo', 'Bicos']}
    despesas_percentuais = {k: (v / orcamento_total) * 100 for k, v in despesas.items()}


    plt.figure(figsize=(10, 5))
    plt.bar(despesas.keys(), despesas.values(), color='skyblue')
    plt.xlabel('Categoria')
    plt.ylabel('Valor em R$')
    plt.title('Gastos Absolutos do Novo Usuário')
    plt.xticks(rotation=45)
    for i, (categoria, valor) in enumerate(despesas.items()):
        plt.text(i, valor + 50, f'R$ {valor:.2f}', ha='center', fontsize=10, fontweight='bold')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.pie(despesas_percentuais.values(), labels=despesas_percentuais.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Distribuição da Renda do Novo Usuário')
    plt.show()

    renda = {'Emprego Fixo': novo_usuario['Emprego Fixo'], 'Bicos': novo_usuario['Bicos']}
    plt.figure(figsize=(8, 8))
    plt.pie(renda.values(), labels=renda.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Distribuição da Renda Total do Novo Usuário')
    plt.show()

    print(f"O novo usuário pertence ao cluster {cluster_predito}: {descricao_clusters.get(cluster_predito, 'Cluster desconhecido')}")
    return cluster_predito, descricao_clusters.get(cluster_predito, 'Cluster desconhecido')


novo_usuario = gerar_usuario_aleatorio()
print("Novo usuário gerado:", novo_usuario)
cluster_resultado, descricao = classificar_novo_usuario(novo_usuario)