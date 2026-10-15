# Projeto de Deep Learning — Classificação de Imagens com Fashion MNIST

## 1. Objetivo do projeto

O objetivo deste projeto foi aplicar os fundamentos de Deep Learning por meio da construção, treinamento e avaliação de duas redes neurais para classificação de imagens, utilizando o conjunto de dados Fashion MNIST.

O trabalho teve como foco compreender como diferentes arquiteturas de redes neurais processam imagens, aprendem padrões e realizam previsões. Para isso, foram implementadas duas soluções: a rede convolucional VGG-16, disponibilizada pelo Keras, e uma rede neural densa construída manualmente, seguindo a arquitetura especificada no enunciado.

## 2. Ferramentas e tecnologias utilizadas

O desenvolvimento foi realizado no Google Colab, utilizando Python e bibliotecas especializadas em inteligência artificial e análise de dados.

As principais ferramentas utilizadas foram:

- **Python:** linguagem de programação utilizada para desenvolver a solução.
- **Google Colab:** ambiente de desenvolvimento em nuvem para execução do código.
- **TensorFlow:** biblioteca utilizada para operações de aprendizado profundo e treinamento das redes neurais.
- **Keras:** API utilizada para definir, compilar, treinar e avaliar os modelos.
- **NumPy:** biblioteca utilizada para manipulação de matrizes e preparação das imagens.
- **Matplotlib:** ferramenta utilizada para visualizar imagens e construir gráficos de desempenho.
- **Seaborn:** biblioteca utilizada para visualizar a matriz de confusão.
- **Scikit-learn:** ferramenta utilizada para calcular métricas de classificação, como precisão, recall e F1-score.

## 3. Conjunto de dados Fashion MNIST

O Fashion MNIST é um conjunto de dados composto por 70 mil imagens de roupas e acessórios, divididas em dez categorias: camiseta, calça, suéter, vestido, casaco, sandália, camisa, tênis, bolsa e bota.

Cada imagem possui 28 × 28 pixels e está representada em escala de cinza. O conjunto contém 60 mil imagens para treinamento e 10 mil imagens para teste.

Inicialmente, os dados foram carregados utilizando o módulo de datasets do Keras. Em seguida, os valores dos pixels foram normalizados para o intervalo entre 0 e 1, facilitando o processamento numérico durante o treinamento das redes neurais.

Também foi realizada a visualização de amostras do conjunto de dados, permitindo identificar as categorias e compreender o problema de classificação proposto.

## 4. Desenvolvimento da rede neural VGG-16

A primeira arquitetura utilizada foi a VGG-16, uma rede neural convolucional desenvolvida para classificação de imagens. Essa arquitetura possui camadas convolucionais responsáveis pela extração de características visuais, camadas de pooling que reduzem as dimensões das representações e camadas densas responsáveis pela classificação final.

Como a VGG-16 foi originalmente desenvolvida para receber imagens com três canais de cor, foi necessário adaptar as imagens do Fashion MNIST, que possuem apenas um canal.

O processo de preparação consistiu em adicionar uma dimensão para o canal de cor, replicar o canal de cinza três vezes e redimensionar as imagens de 28 × 28 para 32 × 32 pixels. Dessa forma, as imagens passaram a apresentar o formato de entrada de 32 × 32 × 3, conforme solicitado no enunciado.

A rede foi instanciada por meio do módulo `keras.applications.vgg16`, utilizando o parâmetro `weights=None`. Isso significa que os pesos foram inicializados aleatoriamente, sem aproveitar pesos previamente treinados em outros conjuntos de dados.

Também foi definido o número de dez classes de saída, correspondente às categorias do Fashion MNIST.

Para o treinamento, o modelo foi compilado com o otimizador Adam, a função de perda `sparse_categorical_crossentropy` e a métrica de acurácia. A rede foi treinada inicialmente por três épocas, com lotes de 128 imagens e uma divisão de validação de 10% dos dados utilizados no treinamento.

## 5. Desenvolvimento da rede neural densa personalizada

A segunda arquitetura foi construída manualmente utilizando as camadas disponibilizadas pelo Keras. Diferentemente da VGG-16, essa rede foi definida de acordo com as quantidades de neurônios estabelecidas no enunciado.

A arquitetura implementada foi composta pelas seguintes camadas:

- Entrada com imagens de 28 × 28 pixels.
- Camada Flatten, responsável por transformar cada imagem em um vetor de 784 valores.
- Primeira camada oculta com 784 neurônios.
- Segunda camada oculta com 1.024 neurônios.
- Terceira camada oculta com 2.048 neurônios.
- Quarta camada oculta com 2.048 neurônios.
- Camada de saída com dez neurônios, representando as dez classes do problema.

As camadas ocultas utilizaram a função de ativação ReLU, que permite à rede aprender relações não lineares entre as informações processadas. Na camada de saída, foi utilizada a função Softmax, responsável por produzir valores de probabilidade para cada categoria.

Essa arquitetura permite compreender como uma rede neural densa transforma os pixels de uma imagem em representações numéricas e utiliza essas informações para identificar a classe mais provável.

Assim como a VGG-16, a rede personalizada foi compilada com o otimizador Adam, a função de perda `sparse_categorical_crossentropy` e a métrica de acurácia. O treinamento foi realizado inicialmente por três épocas, com lotes de 128 imagens e 10% dos dados reservados para validação.

## 6. Treinamento e avaliação dos modelos

Após a construção das duas arquiteturas, foram realizados os treinamentos utilizando o conjunto de dados Fashion MNIST.

Durante o treinamento, as redes receberam as imagens, produziram previsões, calcularam o erro em relação às classes corretas e ajustaram seus pesos por meio da retropropagação do erro e do algoritmo de otimização.

Ao final, os dois modelos foram avaliados utilizando o conjunto de teste, composto por imagens que não foram utilizadas para atualizar os pesos durante o treinamento.

As principais métricas analisadas foram:

- **Acurácia:** proporção de imagens classificadas corretamente.
- **Perda (loss):** medida do erro calculado pela função de perda durante o treinamento e a avaliação.
- **Acurácia de validação:** permite acompanhar a capacidade de generalização dos modelos durante o treinamento.
- **Precisão:** indica a proporção de previsões positivas de uma classe que estavam corretas.
- **Recall:** mede a proporção de exemplos reais de uma classe que foram identificados corretamente.
- **F1-score:** combina precisão e recall em uma única métrica.

Também foram construídos gráficos de acurácia e perda por época, permitindo acompanhar o comportamento dos modelos ao longo do treinamento.

## 7. Matriz de confusão

Para aprofundar a análise, foi utilizada a matriz de confusão, gerada com o Scikit-learn e visualizada por meio do Seaborn.

Essa ferramenta permite comparar as classes reais com as classes previstas pelo modelo. Os valores da diagonal principal representam as classificações corretas, enquanto os valores fora da diagonal indicam erros e confusões entre categorias.

Essa análise é importante porque algumas roupas apresentam características visuais semelhantes. Por exemplo, camisetas e camisas podem ser confundidas devido à resolução reduzida das imagens.

A matriz de confusão, juntamente com as métricas de precisão, recall e F1-score, permite avaliar o desempenho individual de cada categoria, complementando a análise da acurácia geral.

## 8. Comparação entre as arquiteturas

A principal diferença entre os modelos está na forma como processam as imagens.

A VGG-16 utiliza convoluções para identificar características espaciais, como contornos, formas e padrões visuais. Essa estrutura é especialmente relevante para problemas de classificação de imagens.

A rede densa personalizada, por sua vez, transforma cada imagem em um vetor de 784 valores e processa essas informações por meio de camadas totalmente conectadas. Embora seja capaz de aprender padrões complexos, ela não explora diretamente a organização espacial dos pixels por meio de convoluções.

A comparação entre as duas arquiteturas permite observar como diferentes estruturas de redes neurais se comportam diante do mesmo problema. O desempenho final deve ser interpretado com base nos valores reais de acurácia, perda e demais métricas obtidos durante a execução.

É importante considerar que a VGG-16 foi inicializada com pesos aleatórios e que o treinamento inicial utilizou apenas três épocas. Portanto, não é possível afirmar antecipadamente qual arquitetura apresenta o melhor resultado.

## 9. Conclusão

O desenvolvimento deste projeto permitiu aplicar conceitos fundamentais de Deep Learning, incluindo preparação de dados, construção de arquiteturas, treinamento de redes neurais e avaliação de modelos de classificação.

A implementação da VGG-16 demonstrou como utilizar uma arquitetura predefinida do Keras e adaptar os dados de entrada às suas exigências. Já a construção da rede neural densa personalizada permitiu compreender como definir manualmente as camadas, especificar a quantidade de neurônios e estabelecer a saída para múltiplas classes.

A utilização de métricas, gráficos e matriz de confusão possibilitou estruturar uma avaliação mais completa dos modelos, identificando seus acertos, erros e diferenças de desempenho.

Dessa forma, o projeto contribuiu para o entendimento prático das redes neurais convolucionais e densas, além de demonstrar a utilização do Python, TensorFlow e Keras na resolução de problemas de classificação de imagens.

Como resultado, foi construída uma solução experimental que permite treinar, avaliar e comparar duas arquiteturas de Deep Learning em um problema de classificação multiclasse, atendendo aos objetivos educacionais do módulo.

**Observação:** para finalizar a documentação com os resultados experimentais, recomenda-se acrescentar os valores reais de acurácia e perda obtidos para cada modelo após a execução no Google Colab.
