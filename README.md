# House Prices — Previsão de Preços de Imóveis

Projeto de machine learning desenvolvido para a competição **[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)** do Kaggle, com o objetivo de prever o preço de venda de imóveis residenciais na cidade de Ames, Iowa (EUA).

> **Melhor resultado obtido:** score **0.16529** (RMSLE) na competição do Kaggle, utilizando Regressão Linear com pré-processamento refinado.

---

## Índice

- [Visão Geral](#visão-geral)
- [Metodologia — CRISP-DM](#metodologia--crisp-dm)
  - [1. Entendimento do Negócio](#1-entendimento-do-negócio)
  - [2. Entendimento dos Dados](#2-entendimento-dos-dados)
  - [3. Preparação dos Dados](#3-preparação-dos-dados)
  - [4. Modelagem](#4-modelagem)
  - [5. Avaliação](#5-avaliação)
  - [6. Implantação](#6-implantação)
- [Funcionalidades Principais](#funcionalidades-principais)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Guia de Início Rápido](#guia-de-início-rápido)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [Como Usar](#como-usar)
  - [Executando o notebook](#executando-o-notebook)
  - [Executando o app Streamlit](#executando-o-app-streamlit)
- [Exemplos Práticos](#exemplos-práticos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Resultados](#resultados)

---

## Visão Geral

O dataset contém **1.460 registros de treinamento** e **81 variáveis** que descrevem características físicas, de qualidade e de localização de imóveis residenciais. A tarefa é um problema de **regressão supervisionada**: dado o conjunto de atributos de um imóvel, prever seu preço de venda (`SalePrice`).

O projeto foi estruturado seguindo a metodologia **CRISP-DM** (*Cross Industry Standard Process for Data Mining*), que organiza o ciclo de vida de um projeto de ciência de dados em seis fases iterativas, garantindo rastreabilidade e justificativa para cada decisão técnica tomada.

---

## Metodologia — CRISP-DM

A escolha pelo CRISP-DM se deve ao seu amplo reconhecimento na indústria e à sua natureza **iterativa e orientada ao negócio**, o que permite revisitar fases anteriores conforme novos insights surgem durante a análise. Para um problema de competição com dados tabulares e objetivo de minimização de erro, o CRISP-DM oferece uma estrutura clara para documentar e justificar cada etapa.

### 1. Entendimento do Negócio

**Objetivo:** prever o preço final de venda de imóveis residenciais com o menor erro possível.

A métrica de avaliação da competição é o **RMSLE** (Root Mean Squared Logarithmic Error), o que penaliza mais erros em imóveis de valor mais baixo. Internamente, utilizamos **MAE** e **RMSE** em escala absoluta para comparar os modelos durante o desenvolvimento.

### 2. Entendimento dos Dados

O conjunto de dados foi fornecido pela competição do Kaggle e contém:

| Arquivo | Descrição |
|---|---|
| `train.csv` | 1.460 registros com a variável alvo `SalePrice` |
| `test.csv` | 1.459 registros sem `SalePrice` (para submissão) |
| `data_description.txt` | Dicionário completo das 81 variáveis |

A análise inicial revelou que **7 colunas** possuíam mais de 10% de valores nulos (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType`, `FireplaceQu`, `LotFrontage`) e que a variável alvo apresenta distribuição assimétrica à direita, com mediana em torno de **$163.000** e média de **$180.921**.

### 3. Preparação dos Dados

O pré-processamento foi desenvolvido de forma iterativa e resultou no seguinte pipeline:

**Remoção de colunas com alta taxa de nulos**
```python
eliminar = base.columns[(base.isnull().sum() / base.shape[0]) > 0.10]
base = base.drop(eliminar, axis=1)
```
Colunas com mais de 10% de valores ausentes foram removidas, pois qualquer estratégia de imputação introduziria viés significativo dada a proporção de dados faltantes.

**Codificação de variáveis categóricas — One-Hot Encoding**
```python
# BsmtQual: nulos preenchidos com categoria explícita 'None'
base["BsmtQual"] = base["BsmtQual"].fillna("None")
bsmt_dummies = pd.get_dummies(base["BsmtQual"], prefix="BsmtQual").astype(int)

# KitchenQual: drop_first=True para evitar multicolinearidade
kitchen_dummies = pd.get_dummies(base["KitchenQual"], prefix="KitchenQual", drop_first=True).astype(int)
```
O parâmetro `drop_first=True` em `KitchenQual` foi utilizado para evitar a **armadilha da variável dummy** (*dummy variable trap*), que gera multicolinearidade perfeita e prejudica especialmente modelos lineares.

**Imputação direcionada**
```python
# Ar-condicionado central: variável binária
base["central_air"] = base["CentralAir"].apply(lambda x: 1 if x == "Y" else 0)

# MSZoning: preenchido com a moda (preserva distribuição original)
base["MSZoning"] = base["MSZoning"].fillna(base["MSZoning"].mode()[0])

# GarageYrBlt: -1 indica ausência de garagem
# MasVnrArea: 0 indica ausência de revestimento de alvenaria
base2 = base2.fillna({"GarageYrBlt": -1, "MasVnrArea": 0})
```

**Seleção de features**

Apenas colunas numéricas foram mantidas após a codificação, eliminando variáveis textuais que não foram tratadas e que poderiam gerar erros no treinamento.

### 4. Modelagem

Foram avaliados três algoritmos de regressão da biblioteca **Scikit-learn**, com divisão treino/teste de 67%/33% (seed fixo `random_state=42` para reprodutibilidade):

| Modelo | Justificativa |
|---|---|
| **Regressão Linear** | Baseline interpretável; assume relação linear entre features e preço. Eficiente em dados tabulares bem pré-processados. |
| **Árvore de Decisão** | Captura relações não-lineares e interações entre variáveis sem necessidade de normalização. Porém, tende ao overfitting sem limitação de profundidade. |
| **KNN Regressor (k=2)** | Modelo baseado em distância; útil para identificar padrões locais. Sensível à escala das features e a outliers. |

A escolha por esses três modelos foi intencional: permitem comparar abordagens paramétricas (Regressão Linear), baseadas em regras (Árvore) e baseadas em instâncias (KNN), cobrindo diferentes hipóteses sobre a estrutura dos dados.

### 5. Avaliação

| Modelo | MAE | RMSE |
|---|---|---|
| **Regressão Linear** | **$ 22.149** | **$ 36.601** |
| Árvore de Decisão | $ 27.113 | $ 44.145 |
| KNN (k=2) | $ 33.273 | $ 52.285 |

A **Regressão Linear** obteve o melhor desempenho em ambas as métricas. Esse resultado é consistente com a literatura: após um pré-processamento de qualidade, modelos lineares tendem a superar modelos mais complexos em datasets tabulares de tamanho moderado, pois têm menor variância e são menos propensos ao overfitting.

O modelo foi então aplicado ao `test.csv` para geração do arquivo de submissão, resultando em score **0.16529** no Kaggle.

### 6. Implantação

O modelo foi disponibilizado por meio de um **app interativo em Streamlit** (`app.py`) com três seções:

- **Visão Geral** — estatísticas descritivas e primeiros registros do dataset
- **Comparativo de Modelos** — métricas de desempenho e gráficos Real vs. Previsto
- **Simulador de Previsão** — interface para inserir características de um imóvel e obter uma estimativa de preço em tempo real

---

## Funcionalidades Principais

- Análise exploratória dos dados de treinamento (estatísticas descritivas, distribuições)
- Comparativo visual entre três modelos de regressão com métricas MAE, RMSE e MSE
- Simulador interativo de previsão de preço com os parâmetros mais relevantes do imóvel
- Pipeline de pré-processamento reprodutível e documentado

---

## Tecnologias Utilizadas

| Biblioteca | Versão mínima | Uso |
|---|---|---|
| Python | 3.9+ | Linguagem principal |
| pandas | 2.0.0 | Manipulação e análise de dados |
| numpy | 1.26.0 | Operações numéricas |
| scikit-learn | 1.4.0 | Modelos de machine learning e métricas |
| streamlit | 1.32.0 | Interface web interativa |
| plotly | 5.20.0 | Visualizações interativas |

---

## Guia de Início Rápido

### Pré-requisitos

- Python 3.9 ou superior instalado
- pip atualizado (`python -m pip install --upgrade pip`)

### Instalação

**1. Clone o repositório**
```bash
git clone https://github.com/AnaClaraR12/Projeto-House-Prices.git
cd Projeto-House-Prices
```

**2. (Opcional) Crie um ambiente virtual**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

---

## Como Usar

### Executando o notebook

Abra o arquivo `House_Prices_Kaggle_Parte03.ipynb` em qualquer ambiente compatível com Jupyter (VS Code, JupyterLab, Google Colab) e execute as células em ordem.

Os arquivos `train.csv` e `test.csv` devem estar na mesma pasta do notebook.

### Executando o app Streamlit

```bash
streamlit run app.py
```

O app abrirá automaticamente no navegador em `http://localhost:8501`.

---

## Exemplos Práticos

**Previsão de um imóvel via Simulador**

Acesse a página **Simulador de Previsão** no app e ajuste os parâmetros:

| Parâmetro | Exemplo |
|---|---|
| Qualidade Geral (1–10) | 7 |
| Área Habitável (sq ft) | 1.500 |
| Ano de Construção | 2000 |
| Banheiros Completos | 2 |
| Quartos (acima do solo) | 3 |
| Área da Garagem (sq ft) | 400 |
| Qualidade do Porão | Gd (Good) |
| Qualidade da Cozinha | Gd (Good) |
| Ar-Condicionado Central | Sim |

Com esses parâmetros, o modelo de Regressão Linear estima um preço próximo a **$ 192.000**, acima da mediana do dataset ($ 163.000), o que é consistente com um imóvel de qualidade acima da média.

**Geração do arquivo de submissão (via notebook)**

Ao executar todas as células do notebook, o arquivo `resultado03.csv` é gerado automaticamente na mesma pasta com o formato exigido pela competição:

```
Id,SalePrice
1461,128000.00
1462,157000.00
...
```

---

## Estrutura do Projeto

```
Projeto-House-Prices/
├── House_Prices_Kaggle_Parte03.ipynb  # Notebook principal com todo o pipeline
├── app.py                             # App Streamlit interativo
├── train.csv                          # Dados de treinamento (Kaggle)
├── test.csv                           # Dados de teste (Kaggle)
├── resultado03.csv                    # Arquivo de submissão gerado
├── data_description.txt               # Dicionário das variáveis do dataset
├── requirements.txt                   # Dependências do projeto
├── img/
│   ├── final.PNG                      # Resultado final no Kaggle
│   └── resultado_final.PNG            # Histórico de submissões
└── README.md
```

---

## Resultados

| Submissão | Modelo | Score (RMSLE) |
|---|---|---|
| 1ª | Regressão Linear (limpeza básica) | 0.25476 |
| 2ª | Regressão Linear (imputação refinada) | 0.20211 |
| **3ª** | **Regressão Linear (One-Hot Encoding + moda)** | **0.16529** |

A melhoria progressiva do score demonstra que o ganho veio principalmente da **qualidade do pré-processamento** — especialmente da codificação adequada das variáveis categóricas e da estratégia de imputação direcionada — e não da complexidade do modelo em si.

![Resultado Final Kaggle](https://github.com/AnaClaraR12/Projeto-House-Prices/blob/main/img/final.PNG)
