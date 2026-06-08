# Diagnóstico de Câncer de Mama com Machine Learning

Projeto avaliativo P2 da disciplina **Machine Learning**.

## Integrantes

| Discente | RA |
| --- | --- |
| Rayssa Gomides Marconato | 2001130 |
| Samara Adorno | 2001639 |

Grupo: 5

## Descrição do problema

O projeto classifica tumores de mama como benignos ou malignos a partir de características numéricas extraídas de exames citológicos. O problema é de classificação binária, com foco acadêmico em apoio à decisão e interpretação de métricas em contexto oncológico.

## Dataset

Foi usado o Breast Cancer Wisconsin, equivalente ao dataset `uciml/breast-cancer-wisconsin-data`. A cópia utilizada no projeto está em `data/dataset.csv`, com 569 amostras, 30 variáveis preditoras, a coluna `diagnosis` e as colunas auxiliares `id` e `Unnamed: 32`.

## Metodologia

1. Leitura e verificação inicial dos dados.
2. Análise exploratória com distribuição das classes, histogramas, boxplot, correlação e pairplot.
3. Remoção de colunas sem valor preditivo.
4. Conversão da classe: maligno = 1 e benigno = 0.
5. Divisão estratificada em treino, validação e teste.
6. Treinamento com `Pipeline`, `StandardScaler` e `SelectKBest`.
7. Comparação entre SVM, RandomForest e Naive Bayes.
8. Otimização com `GridSearchCV` e `StratifiedKFold`.
9. Avaliação por acurácia, precisão, recall, F1, AUC, matriz de confusão e curva ROC.
10. Salvamento do modelo final em `model/modelo_final.joblib`.

## Aplicação Streamlit

A aplicação em `app.py` possui menu lateral, identidade visual em tons de rosa, logo fixa em `assets/logo_outubro_rosa.png`, painel de indicadores, análise exploratória, simulação de predição, comparação com casos semelhantes, análise do modelo e visualização do dataset.

Para executar:

```bash
streamlit run app.py
```

O app carrega o modelo salvo em `model/modelo_final.joblib`, monta os campos com as variáveis usadas no treinamento e executa a predição.

## Modelo final

O modelo final escolhido foi RandomForest com seleção de features no pipeline. A escolha foi feita com base no desempenho de validação, priorizando F1, recall e AUC.

Métricas finais no teste após retreino com treino + validação:

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9561 | 1.0000 | 0.8810 | 0.9367 | 0.9936 |

## Estrutura

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- .streamlit/
|   `-- config.toml
|-- assets/
|   `-- logo_outubro_rosa.png
|-- docs/
|   |-- roteiro_video_notebook.md
|   `-- roteiro_video_app.md
|-- data/
|   `-- dataset.csv
|-- model/
|   |-- modelo_final.joblib
|   `-- feature_names.joblib
|-- notebooks/
|   `-- notebook_atualizado.ipynb
|-- reports/
|   |-- relatorio_atualizado.pdf
|   |-- metricas_consolidadas.csv
|   |-- features_importantes.csv
|   `-- figures/
`-- src/
    |-- train_model.py
    |-- create_notebook.py
    `-- create_report.py
```

## Como executar o notebook

Instale as dependências:

```bash
pip install -r requirements.txt
```

Abra e execute:

```bash
jupyter notebook notebooks/notebook_atualizado.ipynb
```

## Deploy

Publicar no Streamlit Community Cloud:

1. Subir este projeto para um repositório público no GitHub.
2. Entrar em https://share.streamlit.io.
3. Selecionar o repositório.
4. Definir `app.py` como arquivo principal.
5. Conferir se `requirements.txt`, `model/modelo_final.joblib` e `data/dataset.csv` estão no repositório.

Link do app publicado: preencher após o deploy.

## Limitações

O modelo foi desenvolvido para fins acadêmicos. Ele não substitui avaliação médica, validação clínica, exames complementares ou decisão profissional. Em oncologia, falsos negativos são especialmente críticos porque podem atrasar investigação e tratamento.

## Conclusão

A versão final corrige a divisão metodológica solicitada na devolutiva da P1, adiciona tabela consolidada de métricas, amplia a EDA, inclui seleção de features e garante coerência entre notebook, modelo salvo, relatório e aplicação Streamlit.
