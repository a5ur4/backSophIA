import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

app = Flask(__name__)

# Carregar o arquivo .pkl
try:
    with open('data.pkl', 'rb') as f:
        data_pkl = pickle.load(f)
        modelo = data_pkl['model']
        colunas_esperadas = data_pkl['attributes']
        dtypes_esperados = data_pkl['dtypes']
        value_col = data_pkl['value_col']
        label_encoders_armazenados = data_pkl.get('label_encoders', {})
        scalers_armazenados = data_pkl.get('scalers', {})
except FileNotFoundError:
    print("Erro: Arquivo 'data.pkl' não encontrado.")
    exit(1)
except KeyError as e:
    print(f"Erro: A chave esperada '{e}' não foi encontrada no arquivo .pkl.")
    exit(1)

def converter_para_tipo_correto(df, dtypes):
    """Converte as colunas do DataFrame para os tipos de dados esperados."""
    for col, dtype_str in dtypes.items():
        if col in df.columns:
            try:
                if dtype_str == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype_str == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif dtype_str == 'bool':
                    df[col] = df[col].apply(lambda x: bool(str(x).lower() == 'true'))
            except ValueError as e:
                raise ValueError(f"Erro ao converter a coluna '{col}' para o tipo '{dtype_str}': {str(e)}")
    return df

def validar_valores(data, value_col):
    """Valida se os valores recebidos estão dentro dos esperados."""
    erros = {}
    for col, valores_esperados in value_col.items():
        if col in data:
            valor_recebido = data[col]
            dtype_esperado = None
            if isinstance(valores_esperados, dict) and 'dtype' in valores_esperados:
                dtype_esperado = valores_esperados['dtype']

            if dtype_esperado == 'float64' or dtype_esperado == 'int64':
                valor_recebido = pd.to_numeric(valor_recebido, errors='coerce')
                if pd.isna(valor_recebido):
                    erros[col] = f"Valor inválido para '{col}'. Esperado numérico."
                    continue
            elif isinstance(valores_esperados, list):
                if valor_recebido not in valores_esperados:
                    erros[col] = f"Valor '{valor_recebido}' inválido para '{col}'. Valores esperados: {valores_esperados}"
            elif dtype_esperado == 'bool':
                if str(valor_recebido).lower() not in ['true', 'false']:
                    erros[col] = f"Valor '{valor_recebido}' inválido para '{col}'. Esperado booleano (True/False)."

    if erros:
        return erros
    return None

def calibrate_probability(probability):
    """Calibra a probabilidade para valores mais baixos, centrados em torno de 17%."""
    # Ajuste linear: reduz a probabilidade para um intervalo mais próximo de 17%
    # Fórmula: calibrated = (probability - 50) * (17 / 50) + 17
    calibrated = (probability - 50) * (17 / 50) + 17
    return max(0, round(calibrated * 100) / 100)  # Garante que não seja negativo e arredonda para 2 casas

@app.route("/prever", methods=["POST"])
def prever():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Nenhum dado fornecido."}), 400

        # Verificar se todas as colunas esperadas estão presentes
        missing_cols = [col for col in colunas_esperadas if col not in data]
        if missing_cols:
            return jsonify({"error": f"Colunas ausentes: {missing_cols}."}), 400

        # Validar os valores recebidos
        erros_validacao = validar_valores(data, value_col)
        if erros_validacao:
            return jsonify({"error": "Erros de validação nos dados.", "details": erros_validacao}), 400

        # Criar DataFrame com os dados na ordem correta
        df = pd.DataFrame([data])[colunas_esperadas]

        # Converter para os tipos de dados esperados
        df = converter_para_tipo_correto(df, dtypes_esperados)

        # Aplicar Label Encoding (se houver)
        for col, encoder in label_encoders_armazenados.items():
            if col in df.columns and df[col].dtype == 'object':
                df[col] = encoder.transform(df[col])

        # Aplicar Escalonamento (se houver)
        for col, scaler in scalers_armazenados.items():
            if col in df.columns and df[col].dtype in ['float64', 'Int64']:
                df[[col]] = scaler.transform(df[[col]])

        # Realizar a predição
        pred = modelo.predict(df)[0]
        prob = modelo.predict_proba(df)[0][1] * 100  # Probabilidade em porcentagem
        prob_calibrated = calibrate_probability(prob)  # Calibra a probabilidade
        limiar_risco_alto = 0.5  # Limiar ainda em 0.5 para a classificação
        risco_obito = 1 if prob_calibrated >= limiar_risco_alto else 0

        return jsonify({
            "risco_obito": int(risco_obito),
            "probabilidade": prob_calibrated
        })
    except Exception as e:
        return jsonify({"error": f"Erro ao processar a requisição: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)