# Projeto de Machine Learning — Classificador de Fake News com Processamento de Linguagem Natural (NLP)

## 1. Objetivo do projeto

O objetivo deste projeto é desenvolver um sistema de classificação automática de notícias falsas (*Fake News*) utilizando técnicas de Processamento de Linguagem Natural (NLP), Python e algoritmos de Deep Learning.

A proposta consiste em construir um pipeline capaz de receber textos associados a notícias ou citações, realizar o pré-processamento textual e utilizar uma rede neural para identificar padrões linguísticos relacionados às duas classes do problema: conteúdo classificado como Fake News e conteúdo classificado como não Fake News.

O projeto faz parte do desafio prático do Módulo 2 — Processamento de Linguagem Natural, da formação de Arquiteto(a) de Machine Learning.

Além de implementar o classificador, o trabalho busca compreender como alterações na arquitetura da rede neural e em seus parâmetros de configuração podem influenciar o aprendizado e o desempenho do modelo.

## 2. Tecnologias e ferramentas utilizadas

O desenvolvimento foi planejado para execução no Google Colab, utilizando Python e bibliotecas especializadas em análise de dados, processamento textual e redes neurais.

**Python:** linguagem principal para manipulação dos dados, implementação do pipeline de NLP, treinamento da rede neural e avaliação dos resultados.

**Google Colab:** ambiente de desenvolvimento em nuvem utilizado para executar o código Python em células, carregar os arquivos CSV e realizar os experimentos.

**Pandas:** biblioteca utilizada para carregar os conjuntos de dados, inspecionar as colunas, identificar valores ausentes e organizar as informações.

**NumPy:** suporte para operações numéricas e manipulação de estruturas de dados.

**TensorFlow e Keras:** ferramentas previstas para construção, configuração e treinamento da rede neural utilizada na classificação textual.

**Scikit-learn:** biblioteca utilizada para separar os dados de treinamento e validação, realizar a divisão estratificada e calcular métricas de avaliação.

**Tokenização e padding:** técnicas de NLP destinadas a transformar textos em sequências numéricas de tamanho compatível com a entrada da rede neural.

**Matplotlib:** ferramenta que pode ser utilizada para visualizar a evolução das métricas de treinamento e validação ao longo das épocas.

## 3. Conjunto de dados

Os arquivos utilizados no ambiente Google Colab foram identificados no diretório `/content/sample_data`:

- `train.csv`: conjunto de dados de treinamento.
- `test.csv`: conjunto de dados destinado à aplicação do modelo em novos registros.
- `submit.csv`: arquivo relacionado à submissão das previsões.

O arquivo de treinamento apresenta as seguintes colunas:

| Coluna | Finalidade |
|---|---|
| `id` | Identificador do registro. |
| `title` | Título associado ao conteúdo textual. |
| `author` | Autor atribuído ao conteúdo, quando disponível. |
| `text` | Texto da notícia ou conteúdo associado. |
| `label` | Classe utilizada como variável-alvo do treinamento. |

A coluna `label` representa a variável que o modelo deve aprender a prever. Em um problema binário desse tipo, é necessário confirmar a correspondência entre os valores numéricos e as classes Fake News e não Fake News antes de interpretar as previsões.

Durante a inspeção inicial, a leitura de `train.csv` apresentou uma quantidade elevada de valores ausentes e indícios de desalinhamento dos dados. Por isso, a integridade do arquivo precisa ser verificada antes de considerar os dados prontos para treinamento.

## 4. Etapas do desenvolvimento

### 4.1. Carregamento e inspeção dos dados

A primeira etapa consiste em importar os arquivos CSV utilizando Pandas e verificar sua estrutura.

São examinadas a quantidade de registros, as colunas existentes, os tipos de dados, a distribuição das classes e a presença de valores ausentes.

Essa análise é importante para identificar problemas que possam comprometer o treinamento, como textos vazios, rótulos inválidos, registros duplicados e possíveis erros de leitura do arquivo.

### 4.2. Preparação e limpeza dos textos

O pré-processamento textual prepara os dados para que possam ser utilizados pela rede neural.

As operações previstas incluem:

- Identificação e tratamento de textos ausentes.
- Remoção ou tratamento de registros sem informação textual utilizável.
- Conversão dos textos para um formato consistente.
- Seleção do campo textual utilizado como entrada do modelo.
- Verificação dos rótulos de classificação.

Na implementação inicialmente proposta, o campo `title` foi escolhido como entrada textual. Essa decisão permite construir uma primeira versão do classificador, mas limita o aprendizado ao conteúdo dos títulos, sem utilizar diretamente o texto integral da notícia.

A utilização de `text` ou de uma combinação entre título e texto completo constitui uma alternativa que pode ser avaliada posteriormente.

### 4.3. Tokenização e transformação textual

Como as redes neurais não processam palavras diretamente em seu formato textual original, é necessário converter os textos em representações numéricas.

A tokenização identifica as palavras e associa cada uma delas a um índice numérico de um vocabulário.

A configuração inicialmente proposta considera:

- Vocabulário limitado às 5.000 palavras mais relevantes identificadas pelo tokenizador.
- Conversão dos textos em sequências de índices.
- Aplicação de padding para padronizar as sequências em 100 posições.

O padding permite que textos de comprimentos diferentes sejam representados por vetores com o mesmo tamanho, facilitando seu processamento em lotes pela rede neural.

### 4.4. Construção da rede neural

A arquitetura inicialmente proposta utiliza uma rede neural recorrente com camada LSTM (*Long Short-Term Memory*).

A LSTM é apropriada para experimentos com sequências textuais porque consegue aprender relações entre elementos ao longo de uma sequência e manter informações relevantes durante o processamento.

A arquitetura proposta é composta pelas seguintes camadas:

| Camada | Configuração proposta | Função |
|---|---|---|
| Entrada | Sequências de 100 posições | Receber os textos tokenizados. |
| Embedding | Vocabulário de 5.000 palavras e dimensão 40 | Representar os índices das palavras em vetores numéricos densos. |
| LSTM | 100 unidades | Aprender padrões sequenciais presentes nos textos. |
| Dense | 1 neurônio com ativação sigmoid | Produzir uma saída para classificação binária. |

A camada Embedding transforma os índices das palavras em representações vetoriais que podem ser ajustadas durante o treinamento.

A camada LSTM processa essas representações para aprender padrões linguísticos associados às classes.

A camada final utiliza a função de ativação sigmoid para produzir um valor entre 0 e 1, interpretado como uma estimativa para uma das classes do problema, conforme a codificação adotada.

É importante destacar que essa arquitetura corresponde à configuração inicialmente proposta no desenvolvimento. A execução bem-sucedida do treinamento e os resultados numéricos precisam ser registrados para confirmar seu desempenho efetivo.

### 4.5. Configuração do treinamento

A configuração inicial proposta utiliza:

- Otimizador: Adam.
- Função de perda: `binary_crossentropy`.
- Divisão entre treinamento e validação: 85% e 15%.
- Separação estratificada para preservar, tanto quanto possível, a proporção das classes.
- Semente aleatória para favorecer a reprodutibilidade dos experimentos.

A função `binary_crossentropy` é adequada para classificação binária, pois mede a diferença entre as probabilidades estimadas pelo modelo e os rótulos reais.

O otimizador Adam ajusta os pesos da rede durante o treinamento com base nos gradientes calculados pela função de perda.

O conjunto de validação permite acompanhar a capacidade de generalização do modelo em registros que não foram utilizados diretamente para ajustar os pesos.

### 4.6. Treinamento e avaliação

Durante o treinamento, o modelo recebe os exemplos e seus respectivos rótulos, calcula as previsões e ajusta seus parâmetros para reduzir o erro de classificação.

O acompanhamento deve considerar a evolução da função de perda e da acurácia, tanto no conjunto de treinamento quanto no de validação.

A comparação entre essas curvas ajuda a identificar situações em que o modelo aprende os exemplos de treinamento, mas apresenta dificuldade para generalizar para novos textos.

As métricas recomendadas para a avaliação incluem:

- **Acurácia:** proporção total de previsões corretas.
- **Precisão:** proporção de previsões positivas que realmente pertencem à classe positiva.
- **Recall:** proporção dos exemplos positivos reais que foram identificados pelo modelo.
- **F1-score:** média harmônica entre precisão e recall.
- **Matriz de confusão:** distribuição dos acertos e erros entre as duas classes.

A interpretação da precisão e do recall depende de qual classe for definida como positiva. Por isso, a codificação dos rótulos deve ser documentada.

## 5. Experimentos com diferentes configurações

Uma parte importante do desafio consiste em avaliar como mudanças na arquitetura e nos parâmetros influenciam o comportamento do classificador.

A configuração inicial deve servir como referência para as comparações posteriores.

As questões 10 a 15 do desafio devem ser testadas individualmente, alterando somente a configuração correspondente a cada experimento e restaurando o estado inicial do notebook antes de iniciar o teste seguinte.

Esse procedimento permite comparar os resultados com maior controle, reduzindo a possibilidade de atribuir a uma alteração um efeito provocado por outra mudança simultânea.

Para cada experimento, devem ser registrados:

1. A configuração original.
2. A alteração aplicada.
3. As métricas de treinamento e validação.
4. O tempo de execução, quando relevante.
5. A diferença observada em relação ao modelo de referência.
6. A conclusão sobre o efeito da alteração.

O objetivo não é apenas obter uma boa métrica, mas compreender como as decisões de arquitetura e treinamento afetam o desempenho e a capacidade de generalização.

## 6. Aplicação do modelo em novos dados

Depois do treinamento e da validação, o modelo pode ser utilizado para gerar previsões para registros do conjunto `test.csv`.

Para que a inferência seja consistente, os novos textos devem passar pelas mesmas etapas de pré-processamento utilizadas no treinamento, com o mesmo tokenizador, vocabulário e tamanho de sequência.

As previsões produzidas pela camada sigmoid podem ser convertidas em classes conforme um limiar de decisão definido e documentado.

O arquivo `submit.csv` deve ser preenchido de acordo com o formato exigido pelo desafio, preservando os identificadores e as colunas esperadas.

É importante distinguir essa etapa de uma avaliação supervisionada: se o arquivo de teste não apresentar rótulos conhecidos, não será possível calcular diretamente a acurácia real sobre ele. Nesse caso, a avaliação quantitativa deve ser realizada com dados rotulados separados para validação.

## 7. Principais desafios técnicos

O projeto envolve desafios que vão além da construção da rede neural.

**Qualidade dos dados:** a presença de valores ausentes e o possível desalinhamento de colunas exigem uma verificação cuidadosa antes do treinamento.

**Representação textual:** a tokenização e o padding precisam produzir entradas consistentes, sem perder informações relevantes em excesso.

**Escolha da arquitetura:** a LSTM deve ser avaliada em relação ao custo computacional, à capacidade de aprender padrões textuais e ao desempenho na validação.

**Generalização:** um modelo com bons resultados no treinamento pode apresentar desempenho inferior em textos novos.

**Reprodutibilidade:** a restauração da configuração inicial entre os experimentos facilita a comparação técnica dos resultados.

**Interpretação das métricas:** a acurácia isolada pode ser insuficiente quando existe desequilíbrio entre as classes, tornando importante observar também precisão, recall, F1-score e matriz de confusão.

## 8. Resultados e situação do desenvolvimento

O trabalho já possui uma estrutura de desenvolvimento definida para execução no Google Colab, com identificação dos arquivos CSV, inspeção inicial das colunas e planejamento de uma arquitetura de Deep Learning baseada em LSTM.

Entretanto, a inspeção inicial apontou problemas de qualidade ou leitura no conjunto de treinamento, e não há resultados numéricos confirmados neste registro que permitam afirmar qual foi a acurácia, a precisão, o recall ou o F1-score alcançado.

Dessa forma, a análise dos resultados deve ser concluída após a validação da integridade dos dados, a execução do treinamento e a realização dos experimentos previstos nas questões 10 a 15.

## 9. Conclusão

O projeto tem como finalidade aplicar técnicas de Processamento de Linguagem Natural e Deep Learning na classificação binária de conteúdos textuais associados a Fake News.

A solução proposta integra carregamento e análise de dados com Pandas, preparação textual, tokenização, representação vetorial por Embedding e processamento sequencial por LSTM, utilizando TensorFlow/Keras para o treinamento e Scikit-learn para apoiar a validação e a avaliação.

Além de implementar um classificador, o desafio permite compreender o ciclo de desenvolvimento de um modelo de Machine Learning: preparação dos dados, definição da arquitetura, treinamento, avaliação, experimentação controlada e aplicação em novos registros.

O resultado esperado é um pipeline reproduzível, tecnicamente documentado e capaz de gerar classificações a partir dos padrões aprendidos nos dados rotulados, respeitando as limitações do conjunto de dados e das métricas utilizadas.

A classificação gerada deve ser interpretada como uma estimativa computacional, e não como prova definitiva de que uma notícia é verdadeira ou falsa. A confiabilidade do sistema depende da qualidade dos dados, da representatividade das amostras e da avaliação de seu desempenho em conteúdos que não participaram do treinamento.
