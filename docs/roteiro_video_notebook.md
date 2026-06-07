# Roteiro para o vídeo - Explicação do notebook

Tempo sugerido: 7 a 10 minutos.

## 1. Apresentação inicial

Meu nome é Rayssa Gomides Marconato, curso Análise e Desenvolvimento de Sistemas, grupo 5. O tema do projeto é diagnóstico de câncer de mama com Machine Learning, usando o dataset Breast Cancer Wisconsin para classificar tumores como benignos ou malignos.

Este projeto é a continuação da P1. Na P2, eu corrigi os pontos indicados na devolutiva: incluí separação em treino, validação e teste, adicionei uma tabela consolidada de métricas, ampliei a análise exploratória e incluí seleção de features dentro do pipeline.

## 2. Importação das bibliotecas

No início do notebook, importo as bibliotecas principais:

- `pandas` e `numpy` para manipulação dos dados;
- `matplotlib` e `seaborn` para gráficos;
- `train_test_split`, `StratifiedKFold` e `GridSearchCV` para divisão dos dados, validação cruzada e otimização;
- `Pipeline`, `StandardScaler` e `SelectKBest` para pré-processamento organizado;
- `SVC`, `RandomForestClassifier` e `GaussianNB` como modelos comparados;
- métricas como acurácia, precisão, recall, F1, AUC, matriz de confusão e curva ROC;
- `joblib` para salvar o modelo final.

A ideia do `Pipeline` é garantir que as etapas de padronização, seleção de features e classificação sejam aplicadas de forma consistente, evitando vazamento de dados.

## 3. Leitura e verificação dos dados

O dataset é carregado a partir de `data/dataset.csv`. Ele tem 569 amostras, 30 variáveis numéricas usadas pelo modelo e a variável-alvo `diagnosis`.

A coluna `diagnosis` tem dois valores:

- `M`, para tumor maligno;
- `B`, para tumor benigno.

Nesta etapa, uso `head()`, `info()`, `isna().sum()`, `duplicated()` e `describe()` para entender a estrutura da base, verificar tipos de dados, valores ausentes, duplicatas e estatísticas iniciais.

A coluna `Unnamed: 32` aparece totalmente vazia e a coluna `id` é apenas identificadora. Por isso, essas colunas não entram no treinamento.

## 4. Análise exploratória dos dados

Na análise exploratória, eu verifico a distribuição das classes e percebo que há mais casos benignos do que malignos. Isso justifica o uso de divisão estratificada, para preservar proporções parecidas em treino, validação e teste.

Também analiso histogramas, boxplots, correlação e pairplot. Esses gráficos mostram que algumas variáveis têm maior separação entre tumores benignos e malignos, principalmente:

- `worst perimeter`;
- `worst radius`;
- `worst area`;
- `worst concave points`;
- `mean concave points`.

Essas variáveis fazem sentido biologicamente porque estão ligadas a tamanho, forma, perímetro, área e irregularidade das células.

## 5. Pré-processamento

No pré-processamento, removo `id` e `Unnamed: 32`, porque não ajudam na classificação.

Depois, transformo a variável `diagnosis` em número:

- maligno vira `1`;
- benigno vira `0`.

Em seguida, separo:

- `X`, com as variáveis de entrada;
- `y`, com a variável-alvo.

A divisão dos dados foi feita em três partes:

- treino: usado para o modelo aprender;
- validação: usado para comparar modelos e escolher hiperparâmetros;
- teste: usado somente no final, para medir desempenho em dados não vistos.

Usei `stratify=y` para manter a proporção entre benignos e malignos em todos os conjuntos.

## 6. PCA

O PCA foi usado apenas para visualização. Ele reduz as 30 variáveis para duas componentes principais, permitindo enxergar graficamente como as classes se distribuem.

Eu não uso o PCA como entrada do modelo final. Ele aparece no notebook para ajudar na interpretação visual dos dados.

## 7. Modelos comparados

Foram comparados três modelos:

- SVM;
- RandomForest;
- Naive Bayes.

Cada modelo foi colocado em um `Pipeline` com:

- `StandardScaler`, para padronizar as variáveis;
- `SelectKBest`, para selecionar as variáveis mais relevantes;
- classificador, que faz a predição final.

O `StandardScaler` é importante porque várias variáveis têm escalas diferentes. A seleção de features melhora a análise e reduz ruído, trabalhando com as características mais informativas.

## 8. Métricas utilizadas

As métricas usadas foram:

- acurácia: proporção geral de acertos;
- precisão: entre os casos previstos como malignos, quantos realmente eram malignos;
- recall: entre todos os casos malignos reais, quantos o modelo conseguiu identificar;
- F1: equilíbrio entre precisão e recall;
- AUC: capacidade do modelo de separar as classes em diferentes limiares.

Em oncologia, o recall é muito importante porque falso negativo é um erro crítico: significa classificar um caso maligno como benigno, podendo atrasar investigação e tratamento.

## 9. Validação cruzada e otimização

Depois da comparação inicial, usei `GridSearchCV` com `StratifiedKFold`.

O `StratifiedKFold` divide o treino em 5 partes, mantendo a proporção das classes. O `GridSearchCV` testa combinações de hiperparâmetros para encontrar a melhor configuração.

Para SVM, foram testados valores de `C`, `gamma` e `kernel`.

Para RandomForest, foram testados:

- `n_estimators`;
- `max_depth`;
- `min_samples_split`;
- quantidade de features selecionadas.

A métrica principal da otimização foi F1, porque ela equilibra precisão e recall.

## 10. Resultados finais

O modelo final escolhido foi RandomForest.

No teste final após retreino, os resultados foram:

- acurácia: 95,61%;
- precisão: 100%;
- recall: 88,10%;
- F1: 93,67%;
- AUC: 99,36%.

A precisão de 100% indica que, nos dados de teste, os casos previstos como malignos realmente eram malignos. O recall de 88,10% indica que o modelo recuperou boa parte dos casos malignos, mas ainda existe risco de falso negativo.

## 11. Matriz de confusão e curva ROC

A matriz de confusão mostra os acertos e erros por classe. Ela permite enxergar falsos positivos e falsos negativos.

A curva ROC mostra a capacidade do modelo de separar benignos e malignos. A AUC alta indica boa separação entre as classes.

Mesmo com desempenho alto, o modelo deve ser entendido como ferramenta acadêmica. Ele não substitui avaliação médica.

## 12. Importância das features

A análise de importância mostra quais variáveis mais influenciaram o RandomForest.

As mais importantes foram principalmente variáveis relacionadas a perímetro, raio, área e pontos côncavos. Isso fortalece a interpretação, porque essas medidas têm relação com tamanho e irregularidade celular.

## 13. Salvamento do modelo

Após escolher o modelo final, ele foi retreinado com treino + validação e salvo em `model/modelo_final.joblib`.

Também foi salvo `model/feature_names.joblib`, com a lista das variáveis usadas pelo modelo.

Esses arquivos são carregados diretamente pelo app Streamlit, garantindo coerência entre notebook, modelo salvo e aplicação.

## 14. Conclusão do notebook

A versão final corrige a principal melhoria pedida na P1, que era separar treino, validação e teste. Também inclui tabela consolidada de métricas, seleção de features, EDA mais detalhada, interpretação clínica e salvamento do modelo final.

O projeto apresenta um fluxo completo de Machine Learning: leitura dos dados, análise exploratória, pré-processamento, comparação de modelos, otimização, avaliação final e deploy em aplicação.
