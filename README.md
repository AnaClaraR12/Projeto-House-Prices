# House Prices 🏠
Projeto criado para **[Competição Kaggle com o objetivo de prever preços de imóveis usando modelos de machine learning.](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**

O histórico dos resultados é mostrado abaixo e pode ser obtido no Kaggle:
![Resultado](https://github.com/AnaClaraR12/Projeto-House-Prices/blob/main/img/resultado_final.PNG)

## **[Primeira Etapa](https://github.com/AnaClaraR12/Projeto-House-Prices/blob/main/Projeto_House_Prices_Kaggle.ipynb)**
Nesta etapa inicial do projeto, o foco foi a construção de um modelo de machine learning para prever os preços de imóveis, com uma abordagem mais direta para a limpeza e tratamento de dados.

### **Limpeza e Tratamento de Dados**

Primeiro, a base de dados de treinamento (train.csv) foi lida utilizando a biblioteca pandas. Em seguida, a análise dos dados mostrou que algumas colunas tinham uma alta porcentagem de valores nulos. Para simplificar, todas as colunas com mais de 10% de dados faltantes foram removidas. As colunas restantes com valores nulos foram preenchidas com o valor -1 para garantir que o modelo pudesse ser treinado.

### **Modelos de Machine Learning**

O conjunto de dados foi dividido em treino e teste para avaliar o desempenho dos modelos. Foram utilizados três algoritmos de regressão da biblioteca Scikit-learn:

- Regressão Linear: Um modelo simples e interpretável para prever a relação entre as variáveis.

- Árvore de Decisão para Regressão: Um modelo que pode capturar relações não-lineares nos dados.

- KNeighborsRegressor: Um modelo de aprendizado baseado em instâncias que prevê o preço com base nos vizinhos mais próximos.

O desempenho de cada modelo foi avaliado usando o Erro Médio Absoluto (MAE) e o Erro Quadrático Médio (MSE), comparando os valores previstos com os valores reais da base de teste. O modelo de Regressão Linear apresentou os melhores resultados nesta etapa, com um MAE de **23763.19** e um MSE de **1533982883.44**. Um gráfico de dispersão foi gerado para visualizar o desempenho de cada modelo, comparando os valores reais com as previsões.

A etapa final consistiu em aplicar o modelo de Regressão Linear (o que obteve o melhor desempenho) para fazer previsões na base de dados de teste do Kaggle (test.csv) resultando em uma previsão de 0,25476. O mesmo processo de limpeza e tratamento de dados, como a remoção de colunas com mais de 10% de valores nulos e o preenchimento de valores faltantes, foi aplicado na base de teste antes de gerar as previsões.

## **[Segunda Etapa](https://github.com/AnaClaraR12/Projeto-House-Prices/blob/main/Projeto_House_Prices_Kaggle_Parte02.ipynb.ipynb)**
Na segunda etapa, o projeto foi aprimorado com uma limpeza e tratamento de dados mais cuidadosos para otimizar os modelos de machine learning e melhorar a acurácia das previsões.

### **Limpeza e Tratamento de Dados Avançado**

Assim como na primeira etapa, as colunas com mais de 10% de dados nulos foram removidas. No entanto, foi realizada uma análise mais detalhada em algumas colunas de texto para identificar e tratar dados categóricos. A coluna CentralAir, por exemplo, que continha os valores 'Y' e 'N', foi convertida em uma nova coluna numérica chamada central_air, onde 'Y' foi mapeado para 1 e 'N' para 0. Essa transformação permite que o modelo interprete melhor essa característica.

Para as colunas restantes com valores nulos, foi adotada uma estratégia de preenchimento mais específica:

- A coluna GarageYrBlt (ano de construção da garagem) teve seus valores nulos preenchidos com -1.

- A coluna MasVnrArea (área de alvenaria) teve seus valores nulos preenchidos com 0.

- Outras colunas numéricas com valores faltantes foram preenchidas com -1.

Essa abordagem mais refinada de limpeza resultou em um conjunto de dados mais robusto e preparado para a modelagem.

### **Modelos de Machine Learning e Resultados**

Os mesmos três modelos de machine learning da primeira etapa foram utilizados: Regressão Linear, Árvore de Decisão para Regressão e KNeighborsRegressor. Após a nova rodada de treinamento e avaliação com o conjunto de dados tratado, o modelo de Regressão Linear novamente apresentou o melhor desempenho. Os novos valores de erro foram um MAE de **23690.97** e um MSE de **1529206168.50**. Esses resultados mostram uma pequena melhoria em relação à primeira etapa, indicando que o tratamento de dados mais detalhado foi eficaz.

A etapa final seguiu o mesmo padrão, utilizando o modelo de Regressão Linear para gerar novas previsões na base de teste do Kaggle. As previsões foram exportadas para um novo arquivo CSV, demonstrando o resultado final do projeto após as melhorias com uma previsão agora de 0,20211.

## **[Terceira Etapa](https://github.com/AnaClaraR12/Projeto-House-Prices/blob/main/House_Prices_Parte_03.ipynb)**

### **Limpeza e Tratamento de Dados Avançado**

O projeto passou por um pré-processamento de dados mais sofisticado para melhorar o desempenho do modelo. Foram implementadas as seguintes melhorias:

- **One-Hot Encoding**: As colunas de qualidade do porão (BsmtQual) e da cozinha (KitchenQual) tiveram suas categorias transformadas em colunas numéricas binárias. O parâmetro drop_first=True foi usado na coluna KitchenQual para evitar multicolinearidade, tornando o modelo mais robusto.

- **Imputação Refinada**: A estratégia de imputação para dados ausentes foi aprimorada. Os valores nulos na coluna MSZoning foram preenchidos com a moda (valor mais frequente), garantindo a preservação da distribuição da coluna.

### **Modelos de Machine Learning e Resultados**
Os mesmos três modelos de machine learning da primeira etapa foram utilizados: Regressão Linear, Árvore de Decisão para Regressão e KNeighborsRegressor. Com as melhorias no pré-processamento, o modelo de Regressão Linear manteve o melhor desempenho, resultando em uma nova previsão que alcançou um score de 0.16529 na competição do Kaggle.

![Final_Kaggle](https://github.com/AnaClaraR12/Projeto-House-Prices/blob/main/img/final.PNG)
