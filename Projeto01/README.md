Explicação detalhada do projeto — Fundamentos de Deep Learning
O projeto desenvolvido teve como finalidade construir, treinar e avaliar duas redes neurais capazes de reconhecer e classificar imagens de roupas e acessórios. Para isso, foi utilizado o conjunto de dados Fashion MNIST, com a implementação das arquiteturas VGG-16 e de uma rede neural densa personalizada, utilizando Python, TensorFlow e Keras no Google Colab.
A seguir, apresento uma explicação detalhada do que foi construído, como cada componente funciona e quais conceitos de inteligência artificial foram aplicados.
1. Entendimento do problema: classificação de imagens
O problema abordado é um exemplo de aprendizado supervisionado. Nesse tipo de aprendizado, o modelo recebe exemplos de entrada acompanhados das respostas corretas e aprende a identificar padrões que permitam classificar novos exemplos.
O Fashion MNIST contém 70 mil imagens em tons de cinza, divididas em 60 mil imagens para treinamento e 10 mil para teste. Cada imagem possui 28 × 28 pixels e pertence a uma de dez categorias.









1. Camiseta (T-shirt/top)
Representa camisetas e peças superiores simples.












2. Calça (Trouser)
Peças de vestuário com formato característico de calças.












3. Suéter (Pullover)
Roupas de manga comprida semelhantes a suéteres.












4. Vestido (Dress)
Imagens de vestidos com diferentes contornos.












5. Casaco (Coat)
Peças externas de vestuário.












6. Sandália (Sandal)
Calçados abertos.












7. Camisa (Shirt)
Camisas, uma das classes visualmente mais difíceis de distinguir de algumas outras roupas.












8. Tênis (Sneaker)
Calçados fechados e esportivos.












9. Bolsa (Bag)
Bolsas e objetos semelhantes.












10. Bota (Ankle boot)
Calçados fechados com cano.



O objetivo é ensinar as redes a associar os pixels de uma imagem à categoria correta. Por exemplo, ao receber a imagem de um tênis, o modelo deverá atribuir a maior probabilidade à classe correspondente a tênis.
2. Ambiente de desenvolvimento e bibliotecas
O projeto foi executado no Google Colab, um ambiente que permite escrever e executar código Python em notebooks, sem a necessidade de configurar toda a infraestrutura localmente.
Foram utilizadas as seguintes tecnologias:
Tecnologia	Função no projeto
Python	Linguagem utilizada para implementar a solução.
TensorFlow	Biblioteca responsável pelas operações de aprendizado profundo e treinamento.
Keras	API utilizada para definir, compilar e treinar as redes neurais.
NumPy	Manipulação de matrizes e preparação dos dados.
Matplotlib	Criação de gráficos e visualização das imagens.
Seaborn	Visualização da matriz de confusão.
Scikit-learn	Cálculo de métricas de classificação.
O Keras simplifica a construção de modelos por meio de camadas que podem ser organizadas em sequência, sem exigir a implementação manual de todas as operações matemáticas de uma rede neural.
3. Carregamento e preparação dos dados
A primeira etapa técnica consistiu em carregar o Fashion MNIST e separar as imagens e suas respectivas classes.
Cada imagem é uma matriz de 28 linhas por 28 colunas. Cada posição dessa matriz representa um pixel com um valor de intensidade em escala de cinza.
O código utiliza a função fashion_mnist.load_data(), que retorna quatro conjuntos:
- X_train: imagens utilizadas no treinamento.
- y_train: classes corretas das imagens de treinamento.
- X_test: imagens reservadas para a avaliação final.
- y_test: classes corretas das imagens de teste.
Normalização dos pixels
Os valores dos pixels são convertidos para o tipo float32 e divididos por 255. Dessa forma, os valores passam de um intervalo de 0 a 255 para um intervalo de 0 a 1.
Essa transformação facilita o processamento numérico e costuma favorecer o treinamento das redes neurais.
Visualização das imagens
Também foi construída uma visualização de dez imagens, com seus respectivos nomes de classes. Essa etapa permite verificar visualmente os dados e compreender o problema que o modelo precisa resolver.
4. Primeira arquitetura: VGG-16







arivis.cloud


A VGG-16 é uma arquitetura de rede neural convolucional desenvolvida para classificação de imagens. Diferentemente de uma rede densa simples, ela utiliza filtros convolucionais para identificar características espaciais presentes nas imagens.
A VGG-16 contém blocos de convolução, operações de redução de dimensão e camadas densas responsáveis pela classificação.
Por que a VGG-16 precisou ser adaptada?
O modelo original foi concebido para imagens coloridas. O Fashion MNIST, por outro lado, utiliza imagens em escala de cinza de 28 × 28 pixels.
Por isso, foi necessário transformar os dados antes de fornecê-los à rede.
O processo foi dividido em três operações:
1. Adicionar um canal: a imagem passa de (28, 28) para (28, 28, 1).
2. Replicar o canal: o único canal é copiado três vezes, resultando em (28, 28, 3).
3. Redimensionar a imagem: as dimensões são alteradas para (32, 32, 3).
É importante observar que replicar os canais não transforma a imagem em uma fotografia colorida real. Apenas cria uma representação de três canais compatível com a entrada exigida pela arquitetura utilizada.
Instanciação da rede
A VGG-16 foi criada com os seguintes parâmetros:
- include_top=True: mantém as camadas finais de classificação da arquitetura.
- weights=None: inicializa os pesos aleatoriamente, sem utilizar pesos pré-treinados.
- input_shape=(32, 32, 3): define o formato das imagens recebidas.
- classes=10: define dez saídas para as dez categorias do Fashion MNIST.
Como os pesos começam aleatoriamente, a rede precisa aprender as características das roupas a partir do próprio conjunto de treinamento. Isso é diferente de utilizar transferência de aprendizado com pesos previamente treinados.
Compilação e treinamento
O modelo foi compilado com o otimizador Adam, a função de perda sparse_categorical_crossentropy e a métrica de acurácia.
O Adam ajusta os pesos da rede durante o treinamento. A função de perda mede o erro entre as previsões e as classes corretas. A acurácia indica a proporção de exemplos classificados corretamente.
O treinamento foi configurado inicialmente para três épocas, com lotes de 128 imagens e 10% dos dados de treinamento reservados para validação.
Uma época corresponde a uma passagem pelo conjunto de treinamento utilizado naquela execução. A validação ajuda a observar como o modelo se comporta em exemplos que não são usados para atualizar os pesos naquela época.
5. Segunda arquitetura: rede neural densa personalizada
A segunda rede foi construída manualmente com camadas densas. Esse é um ponto importante do trabalho, pois demonstra que é possível definir uma arquitetura própria em vez de importar uma rede já pronta.
Entrada
Imagem 28 × 28 pixels = 784 valores

Flatten
Transforma a matriz em um vetor de 784 posições

Camada densa 1
784 neurônios
Ativação ReLU

Camada densa 2
1.024 neurônios
Ativação ReLU

Camada densa 3
2.048 neurônios
Ativação ReLU

Camada densa 4
2.048 neurônios
Ativação ReLU

Saída
10 neurônios
Ativação Softmax — uma probabilidade para cada classe


Como funciona cada componente?
Camada de entrada e Flatten
A imagem possui 784 pixels. A operação Flatten transforma a matriz bidimensional em um vetor unidimensional. Ela não aprende características nem modifica o conteúdo dos pixels; apenas reorganiza a representação dos dados.
Camadas ocultas
As quatro camadas densas processam as informações recebidas da camada anterior. Cada neurônio combina entradas com pesos, adiciona um viés e aplica uma função de ativação.
A função ReLU mantém valores positivos e converte valores negativos em zero. Isso permite que a rede aprenda relações não lineares entre os dados.
Camada de saída
A última camada possui dez neurônios, um para cada classe. A função Softmax converte as saídas em valores que podem ser interpretados como probabilidades relativas às dez categorias.
A classe prevista é aquela que apresenta o maior valor de saída.
Por que essa rede é diferente da VGG-16?
A rede densa recebe os pixels como um vetor, sem utilizar diretamente a estrutura espacial bidimensional da imagem. Já a VGG-16 aplica convoluções que identificam padrões locais, como contornos e formas, antes de realizar a classificação.
Além disso, a rede densa personalizada tem um número elevado de parâmetros devido às camadas de 1.024 e 2.048 neurônios. Isso aumenta o custo computacional e pode elevar o risco de sobreajuste.
6. Treinamento das duas redes
As duas arquiteturas foram treinadas com o mesmo conjunto de imagens e as mesmas classes-alvo, embora cada uma tenha recebido as imagens no formato necessário para sua arquitetura.
O treinamento utiliza três mecanismos principais:
- Propagação direta (forward pass): os dados atravessam as camadas e geram uma previsão.
- Cálculo da perda: a previsão é comparada com a classe correta.
- Retropropagação e atualização dos pesos: o algoritmo calcula os gradientes e o otimizador ajusta os parâmetros para reduzir o erro.
Esse ciclo se repete ao longo dos lotes e das épocas.
O uso do mesmo conjunto de teste permite comparar o desempenho final das duas redes de maneira consistente. Entretanto, três épocas constituem apenas uma execução inicial; não garantem que ambas tenham atingido seu melhor desempenho.
7. Avaliação dos modelos
Depois do treinamento, os modelos foram avaliados com as 10 mil imagens de teste.
Foram coletadas duas métricas principais: perda e acurácia.
Métrica	O que representa
Acurácia	Percentual de imagens classificadas corretamente.
Perda (loss)	Erro calculado pela função de perda escolhida.
Também foram construídos gráficos da acurácia e da perda durante o treinamento e a validação. Esses gráficos permitem verificar se o modelo melhora ao longo das épocas e se existe diferença entre seu desempenho no treinamento e na validação.
Por exemplo, se a acurácia de treinamento aumenta enquanto a acurácia de validação permanece estagnada ou diminui, isso pode indicar sobreajuste (overfitting).
Matriz de confusão
A matriz de confusão compara as classes reais com as classes previstas pelo modelo. Cada linha representa uma classe real e cada coluna representa uma classe prevista.
Os valores da diagonal principal correspondem às classificações corretas. Os demais valores indicam confusões entre categorias.
Por exemplo, o modelo pode confundir uma camisa com uma camiseta, pois algumas características visuais dessas peças são semelhantes nas imagens pequenas do Fashion MNIST.
Precisão, recall e F1-score
O projeto também utiliza o classification_report do Scikit-learn para calcular métricas por classe:
- Precisão: entre as imagens que o modelo classificou em determinada categoria, quantas realmente pertencem a ela.
- Recall: entre as imagens que realmente pertencem à categoria, quantas foram identificadas corretamente.
- F1-score: média harmônica entre precisão e recall, equilibrando as duas métricas.
Essas medidas complementam a acurácia geral, pois revelam quais categorias apresentam melhores ou piores resultados.
8. O que foi efetivamente construído?
Ao final do desenvolvimento, o notebook reúne os seguintes componentes:
Preparação dos dados
Carregamento do Fashion MNIST, normalização dos pixels, separação entre treinamento e teste e preparação das imagens para cada arquitetura.



Duas arquiteturas de Deep Learning
Instanciação da VGG-16 com pesos aleatórios e criação manual de uma rede densa com as quantidades de neurônios solicitadas.



Treinamento e comparação
Compilação, ajuste dos pesos, acompanhamento de métricas e avaliação dos modelos sobre as mesmas imagens de teste.



Análise de resultados
Gráficos de desempenho, matriz de confusão e métricas por classe para entender os acertos e erros das previsões.



9. Conclusão técnica
O projeto demonstrou duas formas de construir soluções para classificação de imagens com Deep Learning. A primeira utiliza uma arquitetura convolucional conhecida, enquanto a segunda permite definir explicitamente a quantidade de neurônios e a sequência de camadas.
A implementação também evidenciou que a escolha da arquitetura não é o único fator determinante do resultado. O formato dos dados, a inicialização dos pesos, o número de épocas, o custo computacional e a capacidade de generalização influenciam diretamente o desempenho final.
A VGG-16 foi utilizada sem pesos pré-treinados, conforme o enunciado, e a rede densa foi construída com quatro camadas ocultas e dez saídas. A comparação entre ambas permite exercitar a interpretação de arquiteturas, a construção de redes com Keras e a avaliação de modelos de classificação.
Importante para a entrega: a explicação descreve o que o notebook implementa, mas não afirma qual rede venceu. Essa conclusão deve ser baseada nos valores reais de acurácia, perda e métricas obtidos durante a execução no Google Colab.
