from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  # Habilita acesso de outras origens (React Native)

# 📌 Carregando e processando os dados
dataframe = pd.read_excel('Analíse_relaçao_economiasEgastos.xlsx')
features = ['Água', 'Celular', 'Luz', 'Internet', 'Aluguel', 'Cartão', 'Lazer', 'Apostas', 'Emprego Fixo', 'Bicos']
dataframe = dataframe[features]

scaler = MinMaxScaler()
dataframe_normalizado = pd.DataFrame(scaler.fit_transform(dataframe), columns=features)

# 📌 Treinando o modelo de agrupamento (K-Means)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
dataframe_normalizado['Cluster'] = kmeans.fit_predict(dataframe_normalizado)

# 📌 Descrição dos clusters
descricao_clusters = {
    0: "Econômico: Gasta pouco e economiza bem, mas falta investir.",
    1: "Equilibrado: Gasta de forma controlada e mantém uma boa organização financeira.",
    2: "Gastador: Gasta muito e pode ter desperdícios, precisa de mais controle financeiro.",
    3: "Corrido: Gasta muito com finanças necessárias, não desperdiça e tem pouco dinheiro sobrando."
}

# 📌 Treinando o modelo de classificação (KNN)
knn = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(
    dataframe_normalizado[features], dataframe_normalizado['Cluster'], test_size=0.2, random_state=42
)
knn.fit(X_train, y_train)

# 📌 Rota de teste (home)
@app.route('/')
def home():
    return jsonify({"mensagem": "API de Classificação Financeira funcionando!"})

# 📌 Rota para classificar um novo usuário
@app.route('/classificar', methods=['POST'])
def classificar():
    try:
        data = request.json  # Recebe os dados do React Native no formato JSON
        if not data or 'userId' not in data:
            return jsonify({"erro": "Nenhum dado válido recebido"}), 400

        resultado = classificar_novo_usuario(data)
        return jsonify(resultado)

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# 📌 Função para classificar um novo usuário
def classificar_novo_usuario(novo_usuario):
    user_id = novo_usuario.get("userId", None)

    # Separando os dados do JSON para manter o formato desejado
    ganhos = novo_usuario.get("ganhos", {})
    despesas = novo_usuario.get("despesas", {})
    investimentos = novo_usuario.get("investimentos", {})

    # Convertendo para DataFrame com as features que a IA usa
    novo_usuario_df = pd.DataFrame([{
        "Água": despesas.get("água", 0),
        "Celular": despesas.get("celular", 0),
        "Luz": despesas.get("luz", 0),
        "Internet": despesas.get("internet", 0),
        "Aluguel": despesas.get("aluguel", 0),
        "Cartão": despesas.get("cartão", 0),
        "Lazer": despesas.get("lazer", 0),
        "Apostas": despesas.get("apostas", 0),
        "Emprego Fixo": ganhos.get("salario", 0),
        "Bicos": ganhos.get("freelas", 0)
    }])

    novo_usuario_normalizado = scaler.transform(novo_usuario_df)
    cluster_predito = knn.predict(novo_usuario_normalizado)[0]

    return {
        "id": np.random.randint(1, 100),  # Gerando um ID aleatório
        "userId": user_id,
        "ganhos": ganhos,
        "despesas": despesas,
        "investimentos": investimentos,
        "classificacao": {
            "cluster": int(cluster_predito),
            "descricao": descricao_clusters.get(cluster_predito, 'Cluster desconhecido')
        }
    }

# 📌 Executando a API
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
