# README do Projeto Flask para Análise de Dados e Predições

## Descrição do Projeto

Este projeto é uma aplicação web desenvolvida com Flask para realizar análises de dados e predições baseadas em um dataset de indicadores sociais e econômicos de países. Ele oferece funcionalidades como upload de arquivos CSV, análise de dados ausentes, geração de gráficos interativos, e predições com algoritmos de machine learning, como Random Forest. 

---

## Funcionalidades Principais

1. **Upload de Arquivo CSV**  
   - Permite ao usuário carregar um arquivo contendo os dados necessários para análise.
   - Verifica a estrutura do arquivo CSV para garantir que contenha as colunas esperadas.

2. **Análise de Dados Ausentes**  
   - Gera um heatmap de dados ausentes.
   - Calcula e apresenta o percentual de dados ausentes por coluna.

3. **Geração de Gráficos Interativos**  
   - Gráficos baseados em Plotly e Matplotlib para explorar as tendências dos dados, como:
     - População total por país.
     - Relação entre PIB per capita e expectativa de vida.
     - Distribuição da população urbana ao longo do tempo.

4. **Predições**  
   - Utiliza Random Forest para estimar o crescimento populacional baseado em atributos econômicos e sociais.
   - Gera relatórios com os resultados das predições.

5. **Configuração do Classificador**  
   - Permite customizar os parâmetros do classificador utilizado nas predições.

---

## Estrutura do Código

### Diretórios e Arquivos
- `app.py`: Arquivo principal contendo toda a lógica do backend Flask.
- `templates/`: Diretório contendo os arquivos HTML para renderizar as páginas do site.
- `static/`: Diretório para arquivos estáticos, como imagens e gráficos interativos gerados.
- `uploads/`: Diretório onde os arquivos CSV enviados pelos usuários são armazenados.
- `output/`: Diretório onde os gráficos e relatórios gerados são salvos.

---

## Dependências

As principais bibliotecas utilizadas neste projeto incluem:

- **Flask**: Framework web para Python.
- **Pandas**: Manipulação e análise de dados.
- **NumPy**: Suporte a cálculos numéricos e arrays.
- **Matplotlib** e **Seaborn**: Visualização de dados.
- **Plotly**: Criação de gráficos interativos.
- **Scikit-learn**: Machine learning e Random Forest.
- **Logging**: Para logar mensagens de erro e informações.

Instale todas as dependências executando:

pip install flask
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install plotly
pip install scikit-learn


---

Execute a aplicação:
   ```bash
   python app.py
   ```
Acesse o site no navegador:
   ```
   http://127.0.0.1:5000
   ```

---

## Estrutura do Dataset

O arquivo CSV deve conter as seguintes colunas:

- `Country`: Nome do país.
- `Year`: Ano de referência.
- `Population growth (annual %)`: Crescimento populacional.
- `People using at least basic drinking water services (% of population)`: Porcentagem da população com acesso à água potável.
- `Individuals using the Internet (% of population)`: Porcentagem da população que usa a Internet.
- `School enrollment, primary (% gross)`: Taxa de matrícula escolar primária.
- `Forest area (% of land area)`: Porcentagem de área florestal.
- `Population, total`: População total.
- `GDP per capita (current US$)`: PIB per capita.
- `Life expectancy at birth, total (years)`: Expectativa de vida ao nascer.
- `Urban population (% of total population)`: Porcentagem de população urbana.

---

## Rotas

1. **Página de Upload** (`/`)  
   Página inicial para envio do arquivo CSV.

2. **Upload de Arquivo** (`/upload`)  
   Processa e valida o arquivo enviado.

3. **Predições** (`/predictions`)  
   Exibe as predições geradas com base nos dados enviados.

4. **Gráficos** (`/graphs`)  
   Exibe gráficos interativos gerados a partir dos dados.

5. **Configuração do Classificador** (`/classifier_config`)  
   Permite configurar parâmetros do modelo de machine learning.

---

## Análise e Visualização

### Dados Ausentes
- Heatmap gerado no arquivo `output/missing_data_heatmap.png`.
- Percentuais salvos em `output/missing_data_percentages.png`.

### Importância das Features
- Gráfico de importância das features salvo em `output/crescimento_populacional_importancia.png`.

### Gráficos Interativos
- Salvos no diretório `static/` com links acessíveis no site.

