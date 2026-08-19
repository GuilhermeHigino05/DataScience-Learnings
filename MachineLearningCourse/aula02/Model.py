#& =====================================================================
#& MODEL.PY — Perceptron e SVM linear "do zero", com suporte a matrizes
#& densas (numpy) e esparsas (scipy).
#& ---------------------------------------------------------------------
#& O que mudou em relação à versão anterior (e por quê):
#&
#& 1. `T` agora significa ÉPOCAS, não iterações. Antes, `SGD` era chamado
#&    com T = 50_000_000 iterações; como cada iteração sorteava 1 amostra
#&    (com reposição), o modelo convergia muito antes e milhões de passos
#&    eram desperdiçados. Agora cada época percorre TODAS as amostras uma
#&    única vez: o mesmo aprendizado com ~20 épocas.
#&
#& 2. As funções detectam se a entrada é densa ou esparsa. O script de
#&    sentimento usa Bag-of-Words (matriz ~99% zeros); operar sobre a
#&    versão esparsa evita multiplicar zeros e economiza memória.
#&
#& 3. `Acuracia` foi vetorizada: uma única multiplicação matriz-vetor
#&    (BLAS) substitui o laço Python que chamava `classificar` linha a
#&    linha.
#&
#& 4. As atualizações usam operações in-place (`*=`, `+=`), evitando a
#&    criação de vetores novos a cada passo (menos alocação de memória).
#& =====================================================================

import numpy as np
from scipy import sparse
# `scipy` já é dependência do scikit-learn, portanto não é uma biblioteca
# "nova" no projeto. `sparse.issparse` detecta o tipo da matriz para
# escolhermos o caminho de código adequado (denso ou esparso).


def PerceptronBase(features, labels, T):
    """
    Perceptron clássico (Rosenblatt).
    Regra: se um ponto está mal classificado, empurra a reta na direção
    dele. Encontra *uma* reta separadora, mas NÃO otimiza a margem.
    """
    #& ÉPOCAS x ITERAÇÕES ------------------------------------------------
    #& `T` = número de épocas (voltas completas sobre os dados). Cada
    #& época visita todas as amostras exatamente uma vez. Em problemas
    #& pequenos, poucas épocas (10~50) já bastam para convergir.
    #& (Na versão antiga, o `SGD` usava `T` como iterações — foi essa
    #& confusão de unidades que gerou a chamada com 50 milhões de passos.)
    is_sparse = sparse.issparse(features)
    # `Comparacao.py` passa matriz DENSA (numpy 2D, dataset "bolas");
    # `AnaliseSentimento.py` passa matriz ESPARSA (Bag-of-Words). O
    # `issparse` nos deixa atender aos dois casos com o mesmo código.

    n = features.shape[0]               # nº de amostras
    theta = np.zeros(features.shape[1]) # pesos: um por feature
    theta_0 = 0.0                       # viés (bias)

    rng = np.random.default_rng(42)
    # Gerador de números aleatórios com SEMENTE FIXA -> resultados
    # reproduzíveis. Antes usávamos `random.randint` dentro do laço:
    # além de depender do estado global, sorteava com reposição e podia
    # repetir ou pular amostras.

    for _ in range(T):
        #& PERMUTAÇÃO x SORTEIO COM REPOSIÇÃO ----------------------------
        #& `rng.permutation(n)` embaralha os índices 0..n-1. Assim, em
        #& uma época, cada amostra é usada exatamente uma vez. O
        #& `random.randint` antigo sorteava com reposição: algumas
        #& amostras eram vistas várias vezes, outras nenhuma — passos
        #& desperdiçados e aprendizado menos uniforme.
        for i in rng.permutation(n):
            if is_sparse:
                row = features.getrow(i)
                # Em matriz esparsa, o produto escalar percorre só as
                # posições não-nulas da linha (rápido). `[0]` extrai o
                # único valor do resultado 1x1.
                margin = row.dot(theta)[0] + theta_0
            else:
                margin = np.dot(theta, features[i]) + theta_0

            # y_i * (theta·x_i + theta_0) <= 0 significa "ponto errado".
            if labels[i] * margin <= 0:
                if is_sparse:
                    # `row.indices`/`row.data` são as posições e valores
                    # não-nulos. Somar direto nesses índices custa
                    # O(não-zeros), em vez de O(dimensão) como faria uma
                    # atualização sobre o vetor denso inteiro.
                    theta[row.indices] += labels[i] * row.data
                else:
                    theta += labels[i] * features[i]
                theta_0 += labels[i]

    return theta, theta_0


def SGD(x, y, T, eta_0, lambda_):
    """
    SVM linear de margem suave, treinada com gradiente descendente
    estocástico (SGD). Apesar do nome da função, o modelo aqui é a SVM:
    hinge loss + regularização L2.
    """
    #& A MATEMÁTICA ESCONDIDA -------------------------------------------
    #& Objetivo: J = média_i[ max(0, 1 - y_i*(theta·x_i + theta_0)) ]
    #&                 + (lambda/2)*||theta||²
    #&   - hinge(z) = max(0, 1 - z): penaliza erros E pontos dentro da
    #&     margem;
    #&   - (lambda/2)*||theta||²: regularização L2, prefere pesos pequenos
    #&     (margem grande => melhor generalização).
    #& As "derivadas" não aparecem explicitamente: foram calculadas no
    #& papel e embutidas nas regras de atualização abaixo (é um
    #& SUBgradiente, porque a hinge tem uma quina em z = 1).
    is_sparse = sparse.issparse(x)
    n = x.shape[0]
    theta = np.zeros(x.shape[1])
    theta_0 = 0.0

    rng = np.random.default_rng(314) # pi :)
    # Semente fixa => experimento reproduzível.

    t = 0
    # Contador GLOBAL de passos. Ele alimenta o decaimento da taxa de
    # aprendizado (eta = eta_0 / sqrt(t)) e continua crescendo entre as
    # épocas. Na versão antiga, `for t in range(1, T+1)` cumpria esse papel.

    for _ in range(T):
        #& T = ÉPOCAS. Chamar com T=20 em 3200 amostras = 64 mil passos.
        #& Antes, T=50_000_000 iterações — o modelo já tinha convergido
        #& muito antes, e o restante era trabalho desperdiçado.
        for i in rng.permutation(n):
            t += 1
            eta = eta_0 / np.sqrt(t)
            # Taxa de aprendizado decrescente: passos grandes no início
            # (avança rápido) e pequenos perto do mínimo (não oscila).

            if is_sparse:
                row = x.getrow(i)
                margin = y[i] * (row.dot(theta)[0] + theta_0)
            else:
                margin = y[i] * (np.dot(theta, x[i]) + theta_0)

            if margin < 1:
                #& margin < 1 => ponto errado OU dentro da margem.
                #& Subgradiente da hinge + L2:  lambda*theta - y_i*x_i.
                #& Passo: theta <- theta - eta * subgradiente, ou seja:
                #&   theta <- (1 - eta*lambda)*theta + eta*y_i*x_i.
                theta *= (1 - eta * lambda_)
                # In-place: multiplica o theta inteiro pelo fator de
                # regularização. Antes, `(1 - eta*lambda_)*theta` criava
                # um vetor novo a cada iteração — 50 milhões de alocações.
                if is_sparse:
                    theta[row.indices] += eta * y[i] * row.data
                    # Soma só nas posições não-nulas da amostra.
                else:
                    theta += eta * y[i] * x[i]
                theta_0 += eta * y[i]
                # O viés NÃO é regularizado: só recebe a contribuição da hinge.
            else:
                #& Ponto confortavelmente correto (fora da margem): a hinge
                #& não contribui, sobra só a regularização -> theta encolhe
                #& um pouco em direção à origem (margem maior).
                theta *= (1 - eta * lambda_)

    return theta, theta_0


def Acuracia(theta, theta_0, x, y):
    """
    Fração de amostras classificadas corretamente (entre 0 e 1).
    Vetorizada: calcula todas as predições de uma vez.
    """
    if x.shape[0] == 0:
        # Antes era `len(x) == 0`. Em matriz esparsa do scipy, `len()`
        # levanta TypeError ("length is ambiguous"); o correto é shape[0].
        return 0.0

    #& VETORIZAÇÃO --------------------------------------------------------
    #& `x @ theta` é uma multiplicação MATRIZ x VETOR: calcula o score de
    #& todas as linhas de uma vez, em código C otimizado (BLAS). A versão
    #& antiga fazia um laço Python chamando `classificar` linha a linha —
    #& muito mais lento. Regra geral: laço sobre amostras com aritmética
    #& -> troque por uma operação de álgebra linear.
    preds = np.where(x @ theta + theta_0 > 0, 1, -1)
    # Reproduz exatamente o antigo `classificar`: 1 se o score > 0, senão -1.

    return float(np.mean(preds == y))
    # `preds == y` gera um vetor de True/False; a média é a fração de acertos.
