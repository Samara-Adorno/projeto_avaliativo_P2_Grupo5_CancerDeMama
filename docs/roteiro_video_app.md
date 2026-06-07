# Roteiro para o vídeo - Demonstração do app Streamlit

Tempo sugerido: até 2 minutos.

## 1. Abertura do app

Agora vou demonstrar a aplicação em Streamlit desenvolvida para usar o modelo salvo no notebook.

A aplicação está no arquivo `app.py` e carrega diretamente:

- `model/modelo_final.joblib`, que é o pipeline final;
- `model/feature_names.joblib`, com as variáveis esperadas pelo modelo;
- `data/dataset.csv`, com a base usada no projeto;
- arquivos de métricas e importância de features para exibir análises no painel.

## 2. Menu lateral

No menu lateral, aparece a identidade visual do projeto com a logo de Outubro Rosa e as páginas principais:

- Painel;
- Predição;
- Análise exploratória;
- Modelo;
- Dados.

O tema visual foi configurado em tons de rosa no arquivo `.streamlit/config.toml`, combinando com a proposta do projeto e com a identidade da campanha de prevenção ao câncer de mama.

## 3. Página Painel

Na página Painel, mostro os indicadores principais:

- quantidade de amostras;
- número de variáveis usadas pelo modelo;
- acurácia final;
- recall para maligno;
- AUC final.

Também aparecem dois gráficos:

- composição da base, mostrando benignos e malignos;
- importância das variáveis do modelo final.

Essa tela serve como resumo executivo do projeto.

## 4. Página Predição

Na página Predição, o usuário pode simular um exame.

Existem três opções de perfil inicial:

- média da base;
- caso benigno real;
- caso maligno real.

Também existem dois modos de preenchimento:

- guiado pelas principais variáveis;
- completo, com todas as 30 variáveis.

Depois de preencher os campos, clico em **Executar predição**.

O app mostra:

- classe prevista: benigno ou maligno;
- probabilidade de malignidade;
- faixa de interpretação;
- texto explicando o resultado;
- casos mais semelhantes encontrados na base;
- opção para baixar os dados da predição.

É importante destacar que essa interpretação é acadêmica e não substitui diagnóstico médico.

## 5. Página Análise exploratória

Na página Análise exploratória, o usuário escolhe uma variável e visualiza:

- estatísticas por classe;
- distribuição por diagnóstico;
- boxplot para dispersão e outliers;
- mapa de correlação entre variáveis selecionadas.

Essa página ajuda a explicar por que algumas variáveis diferenciam melhor tumores benignos e malignos.

## 6. Página Modelo

Na página Modelo, mostro informações sobre o RandomForest final:

- número de features selecionadas;
- precisão final;
- F1 final;
- gráfico de métricas no teste;
- gráfico de importância das features;
- tabela consolidada com os resultados.

Também explico que falsos negativos são críticos em oncologia, porque podem atrasar investigação e tratamento.

## 7. Página Dados

Na página Dados, mostro o dataset usado no projeto.

O usuário pode visualizar todas as linhas e também baixar o CSV.

Essa página deixa o projeto mais transparente, porque permite conferir a base usada pelo modelo.

## 8. Fechamento

Com isso, a aplicação demonstra o ciclo completo do projeto:

- carrega o modelo salvo no notebook;
- permite entrada de dados pelo usuário;
- executa a predição;
- apresenta interpretação clara;
- exibe gráficos, métricas e dados de apoio.

O app é funcional, organizado e coerente com o notebook, o relatório e o modelo final salvo.
