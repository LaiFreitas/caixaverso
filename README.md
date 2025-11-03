# Projeto: Previsão de Churn de Clientes Telco

Análise de ponta a ponta e modelagem de machine learning para prever a rotatividade de clientes. O fluxo de trabalho completo, da exploração à avaliação do modelo, está contido no notebook `caixaverso_2.ipynb`.

## 1. Problema de Negócio

Este projeto visa resolver um problema de negócio central para empresas de telecomunicações: a alta taxa de rotatividade de clientes (churn). O custo de aquisição de novos clientes é substancialmente maior do que o de retenção dos existentes.

**Objetivo:** Desenvolver um modelo de classificação que identifique com precisão os clientes com maior probabilidade de cancelar seus serviços. Isso permite que a empresa direcione proativamente os esforços de retenção, otimizando o orçamento de marketing e reduzindo o churn.

## 2. Base de Dados

* **Fonte:** Kaggle
* **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Descrição:** O dataset contém 7.043 registros de clientes e 21 atributos, incluindo:
    * **Dados Demográficos:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
    * **Serviços Contratados:** `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.
    * **Dados da Conta:** `tenure` (tempo de contrato), `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
    * **Variável Alvo:** `Churn` (Sim/Não).
* **Desafio Chave:** O dataset é **desbalanceado**, com a classe minoritária (Churn='Yes') representando aproximadamente 26.5% dos dados. Isso informa a escolha das métricas de avaliação, tornando a acurácia inadequada.

## 3. Metodologia

O notebook `caixaverso_2.ipynb` segue um fluxo de trabalho estruturado para garantir a robustez e evitar data leakage.

1.  **Análise Exploratória (EDA):** Investigação a fundo das distribuições das features e sua correlação com a variável alvo (`Churn`) para extrair insights de negócio (ex: o impacto do tipo de contrato e do `tenure` no churn).
2.  **Engenharia de Features:** Criação de novos atributos (`NumOptionalServices`) e tratamento de anomalias (ex: `TotalCharges` como `object` e sua relação com `tenure`=0).
3.  **Pré-processamento (Pipeline):** Utilização de `ColumnTransformer` e `Pipeline` do Scikit-learn para criar um fluxo de transformação reprodutível.
    * **Features Numéricas:** Imputação de nulos com `SimpleImputer` (mediana) e padronização com `StandardScaler`.
    * **Features Categóricas:** Imputação de nulos com `SimpleImputer` (moda) e codificação com `OneHotEncoder`.
4.  **Modelagem e Otimização:**
    * **Baseline:** `LogisticRegression` otimizado com `GridSearchCV` para estabelecer uma performance base.
    * **Modelo Avançado:** `XGBClassifier` otimizado com `RandomizedSearchCV` (para eficiência) e uso do parâmetro `scale_pos_weight` para lidar com o desbalanceamento dos dados.
5.  **Avaliação:**
    * **Métrica Principal:** **ROC-AUC** foi selecionada por ser a métrica mais robusta para classificação binária desbalanceada, medindo a capacidade do modelo de distinguir entre as classes.
    * **Métricas Secundárias:** Análise da Matriz de Confusão, Precision e, principalmente, o **Recall da classe 1 (Churn)**, que é vital para o objetivo de negócio (encontrar o máximo de *churners* possível).

## 4. Como Executar

Para reproduzir esta análise, siga os passos:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/LaiFreitas/caixaverso.git](https://github.com/LaiFreitas/caixaverso.git)
    cd caixaverso
    ```

2.  **Crie um ambiente virtual e ative-o:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: .\venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    Você precisará das bibliotecas principais para executar o notebook. Instale-as manualmente:
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
    ```

4.  **Baixe os dados:**
    Faça o download do dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e coloque o arquivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` dentro de uma pasta chamada `data/`.
    *(Esta pasta está no .gitignore e não deve ser "commitada" no repositório).*

5.  **Inicie o Jupyter Notebook:**
    ```bash
    jupyter notebook caixaverso_2.ipynb
    ```

## 5. Estrutura do Repositório

. ├── data/ │ 
	└── WA_Fn-UseC_-Telco-Customer-Churn.csv <-- (Baixado manualmente, ignorado pelo Git) 
  ├── .gitignore 
  ├── caixaverso_2.ipynb <-- (Notebook principal com todo o fluxo) 
  └── README.md <-- (Este arquivo)