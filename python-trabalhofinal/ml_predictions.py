import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model(classifier_type='random_forest', params=None):
    """
    Retorna o modelo de classificador baseado no tipo e parâmetros
    
    Parâmetros:
    - classifier_type: Tipo do classificador ('random_forest', 'gradient_boost', 'svm')
    - params: Dicionário com os parâmetros do modelo
    """
    if params is None:
        params = {}
    
    if classifier_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            random_state=42
        )
    elif classifier_type == 'gradient_boost':
        return GradientBoostingRegressor(
            learning_rate=params.get('learning_rate', 0.1),
            n_estimators=100,
            random_state=42
        )
    elif classifier_type == 'svm':
        return SVR(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0)
        )
    else:
        return RandomForestRegressor(n_estimators=100, random_state=42)

def train_and_predict_next_year(data, target_col, country, classifier_type='random_forest', classifier_params=None):
    """
    Treinar modelo e fazer previsões para o próximo ano
    
    Parâmetros:
    - data: DataFrame com os dados
    - target_col: Coluna alvo para predição
    - country: País para análise
    """
    try:
        # Converter coluna alvo para numérico
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        
        # Pegar dados do país
        country_data = data[data['Country'] == country].copy()
        country_data = country_data.sort_values('Year')
        
        # Pegar último ano e valor atual
        last_year = country_data['Year'].max()
        current_value = country_data[target_col].iloc[-1]
        
        if pd.isnull(current_value):
            valid_values = country_data[target_col].dropna()
            if valid_values.empty:
                return None, None, None, None
            current_value = valid_values.iloc[-1]
        
        # Preparar features
        numeric_cols = ['Year'] + [col for col in data.select_dtypes(include=[np.number]).columns 
                                 if col != target_col and col != 'Year']
        
        # Preparar dados para treino
        X = country_data[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        y = country_data[target_col].fillna(method='ffill').fillna(method='bfill')
        
        if len(y) < 5:  # Requer pelo menos 5 anos de dados
            return None, None, None, None
        
        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Criar e treinar modelo com o classificador configurado
        model = get_model(classifier_type, classifier_params)
        model.fit(X_train, y_train)
        
        # Calcular score
        score = model.score(X_test, y_test)
        
        # Preparar dados para previsão
        next_year = last_year + 1
        next_year_data = X.iloc[-1:].copy()
        next_year_data['Year'] = next_year
        
        # Fazer previsão
        prediction = model.predict(next_year_data)[0]
        
        return float(current_value), float(prediction), int(next_year), float(score)
    except Exception as e:
        logger.error(f"Erro ao processar {country} para {target_col}: {str(e)}")
        return None, None, None, None

def generate_predictions(df, classifier_type='random_forest', classifier_params=None):
    """
    Gerar predições para todos os países
    
    Parâmetros:
    - df: DataFrame com os dados
    - classifier_type: Tipo do classificador a ser usado
    - classifier_params: Parâmetros do classificador
    """
    columns = {
        "Crescimento Populacional": "Population growth (annual %)",
        "Acesso à Água": "People using at least basic drinking water services (% of population)",
        "Uso da Internet": "Individuals using the Internet (% of population)",
        "Matrícula Escolar": "School enrollment, primary (% gross)",
        "Área Florestal": "Forest area (% of land area)"
    }
    
    df = df.replace(['..', '', ' '], np.nan)
    countries = df['Country'].unique()
    results = {}
    
    for country in countries:
        country_results = {}
        
        for indicator_name, column in columns.items():
            if column in df.columns:
                current, prediction, next_year, score = train_and_predict_next_year(
                    df, column, country, classifier_type, classifier_params
                )
                if current is not None:
                    variacao = ((prediction - current) / abs(current) * 100) if current != 0 else 0
                    country_results[indicator_name] = {
                        'atual': current,
                        'previsao': prediction,
                        'ano_previsao': next_year,
                        'variacao': variacao,
                        'score': score,
                        'modelo': classifier_type.replace('_', ' ').title()
                    }
        
        if country_results:
            results[country] = country_results
    
    return results

def save_predictions_report(results):
    """Salvar relatório de predições em arquivo"""
    if not os.path.exists('output'):
        os.makedirs('output')
        
    with open('output/previsoes_sul_asia.txt', 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE PREVISÕES PARA PAÍSES DO SUL DA ÁSIA\n")
        f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for country, predictions in results.items():
            f.write(f"\n{country.upper()}\n")
            f.write("=" * 50 + "\n")
            
            for indicator, data in predictions.items():
                f.write(f"\n{indicator}\n")
                f.write(f"Modelo utilizado: {data['modelo']}\n")
                f.write(f"Score do modelo: {data['score']:.4f}\n")
                f.write(f"Atual ({data['ano_previsao']-1}): {data['atual']:.2f}%\n")
                f.write(f"Previsão ({data['ano_previsao']}): {data['previsao']:.2f}%\n")
                f.write(f"Variação Esperada: {data['variacao']:.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
