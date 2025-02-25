from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import io
import base64

app = Flask(__name__)


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


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classificação de Usuários</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .form-container { margin-bottom: 20px; }
        .result-container { margin-top: 20px; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>Classificação de Usuários</h1>
    <div class="form-container">
        <h2>Insira os dados do usuário:</h2>
        <form action="/classificar" method="post">
            {% for feature in features %}
                <label for="{{ feature }}">{{ feature }}:</label><br>
                <input type="number" id="{{ feature }}" name="{{ feature }}" required><br><br>
            {% endfor %}
            <button type="submit">Classificar</button>
        </form>
    </div>
    {% if resultado %}
    <div class="result-container">
        <h2>Resultado:</h2>
        <p><strong>Cluster:</strong> {{ resultado.cluster }} - {{ resultado.descricao }}</p>
        <h3>Gráficos:</h3>
        <img src="data:image/png;base64,{{ resultado.gastos_absolutos }}" alt="Gastos Absolutos">
        <img src="data:image/png;base64,{{ resultado.distribuicao_renda }}" alt="Distribuição da Renda">
        <img src="data:image/png;base64,{{ resultado.distribuicao_renda_total }}" alt="Distribuição da Renda Total">
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template, features=features, resultado=None)

@app.route('/classificar', methods=['POST'])
def classificar():
    
    data = {feature: float(request.form[feature]) for feature in features}
    
    
    resultado = classificar_novo_usuario(data)
    
   
    return render_template_string(html_template, features=features, resultado=resultado)

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
    
   
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    gastos_absolutos = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.pie(despesas_percentuais.values(), labels=despesas_percentuais.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Distribuição da Renda do Novo Usuário')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    distribuicao_renda = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    renda = {'Emprego Fixo': novo_usuario['Emprego Fixo'], 'Bicos': novo_usuario['Bicos']}
    plt.figure(figsize=(8, 8))
    plt.pie(renda.values(), labels=renda.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Distribuição da Renda Total do Novo Usuário')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    distribuicao_renda_total = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return {
        "cluster": int(cluster_predito),
        "descricao": descricao_clusters.get(cluster_predito, 'Cluster desconhecido'),
        "gastos_absolutos": gastos_absolutos,
        "distribuicao_renda": distribuicao_renda,
        "distribuicao_renda_total": distribuicao_renda_total
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
