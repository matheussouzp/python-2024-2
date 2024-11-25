from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory, session, flash
import pandas as pd
import numpy as np
import os
import shutil
import logging
from ml_predictions import generate_predictions, save_predictions_report
from functools import lru_cache
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Criar a aplicação Flask
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)

# Configurações
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['SECRET_KEY'] = 'sua_chave_secreta_aqui'  # Necessário para sessions

# Configurações padrão do classificador
DEFAULT_CLASSIFIER_CONFIG = {
    'type': 'random_forest',
    'params': {
        'n_estimators': 100,
        'max_depth': 10
    }
}

# Criar diretórios necessários
for directory in [app.config['UPLOAD_FOLDER'], 
                 app.config['STATIC_FOLDER'], 
                 'output']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Definir colunas necessárias
REQUIRED_COLUMNS = [
    'Country',
    'Year',
    'Population growth (annual %)',
    'People using at least basic drinking water services (% of population)',
    'Individuals using the Internet (% of population)',
    'School enrollment, primary (% gross)',
    'Forest area (% of land area)',
    'Population, total',
    'GDP per capita (current US$)',
    'Life expectancy at birth, total (years)',
    'Urban population (% of total population)'
]

def analyze_missing_data(data):
    """Analisar dados ausentes e gerar visualizações"""
    # Calcular percentual de dados ausentes
    missing_percentages = data.isnull().sum() / len(data) * 100
    
    # Criar heatmap de dados ausentes
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Heatmap de Dados Ausentes')
    plt.tight_layout()
    plt.savefig('output/missing_data_heatmap.png')
    plt.close()
    
    # Criar gráfico de barras de percentual de dados ausentes
    plt.figure(figsize=(12, 6))
    missing_percentages.plot(kind='bar')
    plt.title('Percentual de Dados Ausentes por Coluna')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/missing_data_percentages.png')
    plt.close()
    
    # Salvar análise em texto
    with open('output/analise_dados_ausentes.txt', 'w', encoding='utf-8') as f:
        f.write("Análise de Dados Ausentes\n")
        f.write("=" * 50 + "\n\n")
        for col, pct in missing_percentages.items():
            f.write(f"{col}: {pct:.2f}% ausentes\n")

def analyze_feature_importance(data):
    """Analisar importância das features para o crescimento populacional"""
    try:
        # Preparar dados e features
        features = [
            'GDP per capita (current US$)', 
            'Life expectancy at birth, total (years)',
            'Urban population (% of total population)',
            'People using at least basic drinking water services (% of population)',
            'Individuals using the Internet (% of population)'
        ]
        
        # Preparar dados para o modelo
        X = data[features].fillna(method='ffill').fillna(method='bfill')
        y = data['Population growth (annual %)'].fillna(method='ffill').fillna(method='bfill')
        
        # Treinar modelo Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
# Calcular e ordenar importância das features
        importance = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Configurar e criar o gráfico
        plt.figure(figsize=(15, 8))
        ax = importance.plot(kind='barh', x='feature', y='importance')
        plt.grid(True, alpha=0.3)
        
        # Ajustar limites do eixo x para acomodar os valores
        plt.xlim(0, importance['importance'].max() * 1.15)
        
        # Adicionar valores nas barras com padding dinâmico
        max_importance = importance['importance'].max()
        for i, v in enumerate(importance['importance']):
            ax.text(v + max_importance * 0.02, i, f'{v:.3f}', va='center', fontsize=10, ha='left')
        
        # Melhorar labels das features
        feature_labels = {
            'GDP per capita (current US$)': 'PIB per capita',
            'Life expectancy at birth, total (years)': 'Expectativa de vida',
            'Urban population (% of total population)': 'População urbana',
            'People using at least basic drinking water services (% of population)': 'Acesso à água potável',
            'Individuals using the Internet (% of population)': 'Uso de Internet'
        }
        
        # Atualizar os labels do eixo y
        ax.set_yticklabels([feature_labels[feat] for feat in importance['feature']])
        
        # Ajustar título e layout
        plt.title('Importância das Features para o Crescimento Populacional', pad=20, fontsize=12)
        plt.xlabel('Importância', fontsize=10)
        
        # Ajustar margens
        plt.subplots_adjust(left=0.3)
        plt.tight_layout()
        
        # Salvar com maior DPI para melhor qualidade
        plt.savefig('output/crescimento_populacional_importancia.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plotar crescimento populacional por país
        plt.figure(figsize=(12, 6))
        for country in data['Country'].unique():
            country_data = data[data['Country'] == country]
            plt.plot(country_data['Year'], 
                    country_data['Population growth (annual %)'], 
                    label=country, marker='o')
        plt.title('Crescimento Populacional por País ao Longo do Tempo')
        plt.xlabel('Ano')
        plt.ylabel('Crescimento Populacional (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/crescimento_populacional.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"Erro na análise de importância das features: {str(e)}")

def generate_graphs(data):
    """Gerar gráficos interativos"""
    graphs = []
    
    # Análises adicionais
    analyze_missing_data(data)
    analyze_feature_importance(data)

    # Gráfico 1: População por País
    if 'Country' in data.columns and 'Population, total' in data.columns:
        fig = px.bar(data.sort_values('Population, total', ascending=False),
                     x='Country', y='Population, total',
                     title="População Total por País")
        graph_path = os.path.join('static', 'population_bar_chart.html')
        fig.write_html(graph_path)
        graphs.append({'title': 'População por País', 'path': 'population_bar_chart.html'})

    # Gráfico 2: PIB Per Capita x Expectativa de Vida
    if 'GDP per capita (current US$)' in data.columns and 'Life expectancy at birth, total (years)' in data.columns:
        fig = px.scatter(data, x='GDP per capita (current US$)',
                         y='Life expectancy at birth, total (years)',
                         title="PIB Per Capita x Expectativa de Vida",
                         color='Country')
        graph_path = os.path.join('static', 'gdp_life_expectancy_scatter.html')
        fig.write_html(graph_path)
        graphs.append({'title': 'PIB Per Capita x Expectativa de Vida', 'path': 'gdp_life_expectancy_scatter.html'})

    # Gráfico 3: Distribuição da População Urbana por País
    if 'Country' in data.columns and 'Year' in data.columns and 'Urban population (% of total population)' in data.columns:
        fig = px.line(data, x='Year', y='Urban population (% of total population)', 
                      color='Country', markers=True,
                      title="Distribuição da População Urbana por País ao Longo dos Anos")
        graph_path = os.path.join('static', 'urban_population_line_chart.html')
        fig.write_html(graph_path)
        graphs.append({'title': 'Distribuição da População Urbana por País', 'path': 'urban_population_line_chart.html'})

    return graphs

@lru_cache(maxsize=1)
def load_data():
    """Carregar e preparar os dados com cache"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'South_Asian_dataset.csv')
        if not os.path.exists(file_path):
            return None
        df = pd.read_csv(file_path)
        df = df.replace(['..', '', ' '], np.nan)
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        return None

def allowed_file(filename):
    """Verificar se o arquivo é permitido"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def validate_csv_structure(file_path):
    """Validar estrutura do arquivo CSV"""
    try:
        df = pd.read_csv(file_path)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Colunas ausentes: {missing_columns}")
            return False, f"Colunas ausentes: {', '.join(missing_columns)}"
        
        return True, "Arquivo válido"
    except Exception as e:
        logger.error(f"Erro ao validar arquivo: {str(e)}")
        return False, f"Erro ao ler arquivo: {str(e)}"

@app.route('/')
def upload():
    """Rota para página de upload"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Rota para processar upload de arquivo"""
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Apenas arquivos CSV são permitidos'})
    
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'South_Asian_dataset.csv')
        
        # Fazer backup se arquivo existir
        if os.path.exists(file_path):
            backup_path = file_path + '.backup'
            shutil.copy2(file_path, backup_path)
        
        file.save(file_path)
        
        # Validar estrutura do arquivo
        is_valid, message = validate_csv_structure(file_path)
        
        if not is_valid:
            if os.path.exists(file_path + '.backup'):
                shutil.copy2(file_path + '.backup', file_path)
            return jsonify({'error': message})
        
        if os.path.exists(file_path + '.backup'):
            os.remove(file_path + '.backup')
        
        # Limpar cache de dados
        load_data.cache_clear()
        
        # Carregar dados e gerar gráficos
        df = load_data()
        if df is not None:
            generate_graphs(df)
        
        return jsonify({
            'success': 'Arquivo enviado com sucesso',
            'redirect': url_for('predictions')
        })
        
    except Exception as e:
        logger.error(f"Erro no upload do arquivo: {str(e)}")
        return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'})

@app.route('/predictions')
def predictions():
    """Rota para página de predições"""
    df = load_data()
    if df is None:
        return redirect(url_for('upload'))
    
    try:
        # Obter configuração do classificador
        classifier_config = session.get('classifier_config', DEFAULT_CLASSIFIER_CONFIG)
        
        # Gerar predições com o classificador configurado
        predictions_data = generate_predictions(
            df,
            classifier_type=classifier_config['type'],
            classifier_params=classifier_config['params']
        )
        save_predictions_report(predictions_data)
        
        countries = sorted(df['Country'].unique())
        
        return render_template('predictions.html', 
                             predictions=predictions_data,
                             countries=countries)
    except Exception as e:
        logger.error(f"Erro ao gerar predições: {str(e)}")
        return render_template('predictions.html', 
                             error=f"Erro ao gerar predições: {str(e)}",
                             countries=[])

@app.route('/graphs')
def graphs():
    """Rota para página de gráficos"""
    df = load_data()
    if df is None:
        return redirect(url_for('upload'))
    
    try:
        graphs_data = generate_graphs(df)
        return render_template('graphs.html', graphs=graphs_data)
    except Exception as e:
        logger.error(f"Erro ao gerar gráficos: {str(e)}")
        return render_template('graphs.html', error=f"Erro ao gerar gráficos: {str(e)}")

@app.route('/classifier_config')
def classifier_config():
    """Rota para página de configuração do classificador"""
    current_config = session.get('classifier_config', DEFAULT_CLASSIFIER_CONFIG)
    return render_template('classifier_config.html', current_config=current_config)

@app.route('/save_classifier_config', methods=['POST'])
def save_classifier_config():
    """Rota para salvar configurações do classificador"""
    try:
        classifier_type = request.form.get('classifier_type', 'random_forest')
        params = {}
        
        if classifier_type == 'random_forest':
            params['n_estimators'] = int(request.form.get('n_estimators', 100))
            params['max_depth'] = int(request.form.get('max_depth', 10))
        elif classifier_type == 'gradient_boost':
            params['learning_rate'] = float(request.form.get('learning_rate', 0.1))
        elif classifier_type == 'svm':
            params['kernel'] = request.form.get('kernel', 'rbf')
        
        config = {
            'type': classifier_type,
            'params': params
        }
        
        session['classifier_config'] = config
        flash('Configurações do classificador salvas com sucesso!', 'success')
        return redirect(url_for('predictions'))
    except Exception as e:
        logger.error(f"Erro ao salvar configurações: {str(e)}")
        return render_template('classifier_config.html', 
                             error=f"Erro ao salvar configurações: {str(e)}",
                             current_config=DEFAULT_CLASSIFIER_CONFIG)

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Rota para servir arquivos da pasta output"""
    return send_from_directory('output', filename)

if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    app.run(debug=True)
