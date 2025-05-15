import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask_cors import CORS
import numpy as np
import os
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), '../logs/app.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)
CORS(app)  # Permite solicitações de diferentes origens (importante para frontend)

# Caminho para o modelo
data_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'data.pkl')


# Carregar o arquivo .pkl
try:
    with open(data_path, 'rb') as f:
        data_pkl = pickle.load(f)
        modelo = data_pkl['model']
        colunas_esperadas = data_pkl['attributes']
        dtypes_esperados = data_pkl['dtypes']
        value_col = data_pkl['value_col']
        label_encoders_armazenados = data_pkl.get('label_encoders', {})
        scalers_armazenados = data_pkl.get('scalers', {})
    logging.info("Modelo carregado com sucesso")
except FileNotFoundError:
    logging.error(f"Erro: Arquivo '{data_path}' não encontrado")
    print(f"Erro: Arquivo '{data_path}' não encontrado. Verifique se o arquivo existe no diretório correto.")
    exit(1)
except KeyError as e:
    logging.error(f"Erro: A chave esperada '{e}' não foi encontrada no arquivo .pkl")
    print(f"Erro: A chave esperada '{e}' não foi encontrada no arquivo .pkl.")
    exit(1)
except Exception as e:
    logging.error(f"Erro inesperado ao carregar o modelo: {str(e)}")
    print(f"Erro inesperado ao carregar o modelo: {str(e)}")
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
    return max(0, min(100, round(calibrated * 100) / 100))  # Garante valor entre 0-100 e arredonda para 2 casas

@app.route("/", methods=["GET"])
def index():
    """Página inicial com documentação da API."""
    return jsonify({
        "status": "online",
        "api": "SophIA Risk API",
        "version": "1.0",
        "endpoints": {
            "/prever": {
                "method": "POST",
                "description": "Endpoint para prever risco de óbito materno",
                "required_fields": colunas_esperadas
            },
            "/health": {
                "method": "GET",
                "description": "Verifica o status da API"
            }
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint para verificar a saúde da API."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route("/prever", methods=["POST"])
def prever():
    """Endpoint para previsão de risco baseado nos dados enviados."""
    try:
        data = request.get_json()
        if not data:
            logging.warning("Requisição recebida sem dados")
            return jsonify({"error": "Nenhum dado fornecido."}), 400

        # Log da requisição (sem dados sensíveis)
        logging.info(f"Requisição recebida - campos: {list(data.keys())}")

        # Verificar se todas as colunas esperadas estão presentes
        missing_cols = [col for col in colunas_esperadas if col not in data]
        if missing_cols:
            logging.warning(f"Colunas ausentes na requisição: {missing_cols}")
            return jsonify({
                "error": "Colunas ausentes",
                "missing_columns": missing_cols,
                "required_columns": colunas_esperadas
            }), 400

        # Validar os valores recebidos
        erros_validacao = validar_valores(data, value_col)
        if erros_validacao:
            logging.warning(f"Erros de validação: {erros_validacao}")
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
        limiar_risco_alto = 17.0  # Limiar em 17% para classificação
        risco_obito = 1 if prob_calibrated >= limiar_risco_alto else 0

        # Determinar nível de risco para exibição frontend
        nivel_risco = "ALTO" if risco_obito == 1 else "BAIXO"
        
        resultado = {
            "risco_obito": int(risco_obito),
            "probabilidade": prob_calibrated,
            "nivel_risco": nivel_risco,
            "timestamp": datetime.now().isoformat()
        }
        
        logging.info(f"Predição realizada com sucesso: {nivel_risco} ({prob_calibrated:.2f}%)")
        return jsonify(resultado)
    
    except Exception as e:
        logging.error(f"Erro ao processar a requisição: {str(e)}", exc_info=True)
        return jsonify({"error": f"Erro ao processar a requisição: {str(e)}"}), 500

if __name__ == "__main__":
    # Criar diretório de logs se não existir
    os.makedirs(os.path.join(os.path.dirname(__file__), '../logs'), exist_ok=True)
    
    # Criar diretório de models se não existir
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    
    # Avisar sobre localização do arquivo data.pkl
    print(f"\n[INFO] Buscando modelo em: {data_path}")
    print("[INFO] Se o modelo não for encontrado, coloque-o na pasta 'models' no diretório raiz.\n")
    
    # Iniciar a aplicação
    app.run(debug=False, host='0.0.0.0', port=5000)