from flask import Flask, request, jsonify  
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Carregar os dados do Excel
sheet_names = ['Ganhos', 'Despesas', 'Investimentos']

ganhos = pd.read_excel('usuarios_500.xlsx', sheet_name="Ganhos")
despesas = pd.read_excel('usuarios_500.xlsx', sheet_name="Despesas")
investimentos = pd.read_excel('usuarios_500.xlsx', sheet_name="Investimentos")

# Garantir que todos tenham a coluna "Usuario"
for df in [ganhos, despesas, investimentos]:
    if 'Usuario' not in df.columns:
        df['Usuario'] = range(1, len(df) + 1)

# Unir os dados das três tabelas
base_dados = ganhos.merge(despesas, on="Usuario").merge(investimentos, on="Usuario")

# Definir as colunas de interesse
features = ganhos.columns.tolist()[1:] + despesas.columns.tolist()[1:] + investimentos.columns.tolist()[1:]

dataframe = base_dados[features]

# Normalizar os dados
scaler = MinMaxScaler()
dataframe_normalizado = pd.DataFrame(scaler.fit_transform(dataframe), columns=features)

# Aplicar KMeans para clusterização
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
dataframe_normalizado['Cluster'] = kmeans.fit_predict(dataframe_normalizado)

descricao_clusters = {
    0: "Econômico: Gasta pouco e economiza bem, mas falta investir.",
    1: "Equilibrado: Gasta de forma controlada e mantém uma boa organização financeira.",
    2: "Gastador: Gasta muito e pode ter desperdícios, precisa de mais controle financeiro.",
    3: "Corrido: Gasta muito com finanças necessárias, não desperdiça e tem pouco dinheiro sobrando."
}

# Treinar modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(
    dataframe_normalizado[features], dataframe_normalizado['Cluster'], test_size=0.2, random_state=42
)
knn.fit(X_train, y_train)

@app.route('/')
def home():
    return jsonify({"mensagem": "API de Classificação Financeira funcionando!"})

@app.route('/classificar', methods=['POST'])
def classificar():
    try:
        data = request.json  # Recebe os dados JSON
        if not data:
            return jsonify({"erro": "Nenhum dado recebido"}), 400

        # Processar os dados recebidos para o formato correto
        novo_usuario = processar_dados(data)
        
        # Classificar com IA
        resultado = classificar_novo_usuario(novo_usuario)

        return jsonify(resultado)

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# Processar os dados recebidos no formato correto
def processar_dados(data):
    ganhos = data.get("ganhos", {})
    despesas = data.get("despesas", {})
    investimentos = data.get("investimentos", {})

    return {
        "salario": ganhos.get("salario", 0),
        "bonus": ganhos.get("bonus", 0),
        "outros": ganhos.get("outros", 0),
        "rendimentosPassivos": ganhos.get("rendimentosPassivos", 0),
        "freelas": ganhos.get("freelas", 0),
        "dividendos": ganhos.get("dividendos", 0),
        "aluguel": despesas.get("aluguel", 0),
        "contas": despesas.get("contas", 0),
        "alimentacao": despesas.get("alimentacao", 0),
        "transporte": despesas.get("transporte", 0),
        "educacao": despesas.get("educacao", 0),
        "saude": despesas.get("saude", 0),
        "lazer": despesas.get("lazer", 0),
        "acoes": investimentos.get("acoes", 0),
        "fundos": investimentos.get("fundos", 0),
        "criptomoedas": investimentos.get("criptomoedas", 0),
        "imoveis": investimentos.get("imoveis", 0),
        "rendafixa": investimentos.get("rendafixa", 0),
        "negocios": investimentos.get("negocios", 0)
    }

# Classificar um novo usuário
def classificar_novo_usuario(novo_usuario):
    novo_usuario_df = pd.DataFrame([novo_usuario], columns=features)
    novo_usuario_normalizado = scaler.transform(novo_usuario_df)
    cluster_predito = knn.predict(novo_usuario_normalizado)[0]

    return {
        "cluster": int(cluster_predito),
        "descricao": descricao_clusters.get(cluster_predito, 'Cluster desconhecido')
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
