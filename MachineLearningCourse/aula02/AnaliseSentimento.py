#& =====================================================================
#& ANALISESENTIMENTO.PY — classifica resenhas de produtos (positiva x
#& negativa) usando Bag of Words + Perceptron/SVM.
#& ---------------------------------------------------------------------
#& Fluxo: carrega texto -> pré-processa -> Bag of Words ESPARSO ->
#& divide treino/teste -> treina Perceptron e SVM -> compara acurácias.
#&
#& O que mudou nesta versão:
#&   * `create_bow_vectorizer` devolve também os textos processados;
#&   * `features` é a matriz ESPARSA (sem `.toarray()`);
#&   * `SGD(..., T=20, ...)` agora usa 20 ÉPOCAS (não 500 mil iterações).
#& =====================================================================

from loadDataset import carregar_dataset_sentimento
from BagOfWords import create_bow_vectorizer, GetFeatures
from sklearn.model_selection import train_test_split
from visualizacao import visualizar_comparacao_acuracia

from Model import SGD, Acuracia, PerceptronBase

import os
import time

inicio_total = time.perf_counter()

t0 = time.perf_counter()
texts, labels = carregar_dataset_sentimento(os.path.join("dataset", "reviewes.csv"))
print(f"[Tempo] Carregamento do dataset: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
# Desempacota a tupla: o vetorizador E os textos já pré-processados.
# (Antes a função devolvia só o vetorizador e o pré-processamento era
# refeito dentro de `GetFeatures`.)
vectorizer, textos_processados = create_bow_vectorizer(texts)
print(f"[Tempo] Criação do vetorizador (Bag of Words): {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
# `features` agora é uma matriz ESPARSA (scipy CSR), não mais densa.
# Para 4000 resenhas x vocabulário, isso economiza memória e acelera
# todos os produtos escalares do treinamento.
features = GetFeatures(textos_processados, vectorizer)
print(f"[Tempo] Extração de features: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
# `train_test_split` preserva o formato esparso da matriz de features.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(f"[Tempo] Divisão treino/teste: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
thetaP, theta_0P = PerceptronBase(X_train, y_train, 20)
print(f"[Tempo] Treinamento Perceptron: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
# T=20 agora significa 20 ÉPOCAS (~64 mil passos). Antes, T=500_000 era
# o nº de ITERAÇÕES e o treino levava ~62s; agora leva ~2,5s.
thetaSGD, theta_0SGD = SGD(X_train, y_train, 20, 0.6, 0.04)
print(f"[Tempo] Treinamento SGD: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
acuraciaP_treino = Acuracia(thetaP, theta_0P, X_train, y_train)
acuraciaSGD_treino = Acuracia(thetaSGD, theta_0SGD, X_train, y_train)

acuraciaP_teste = Acuracia(thetaP, theta_0P, X_test, y_test)
acuraciaSGD_teste = Acuracia(thetaSGD, theta_0SGD, X_test, y_test)
print(f"[Tempo] Cálculo das acurácias: {time.perf_counter() - t0:.4f}s")

print(f"Perceptron - Acurácia no treino: {acuraciaP_treino:.4f}")
print(f"Perceptron - Acurácia no teste: {acuraciaP_teste:.4f}")
print(f"SGD - Acurácia no treino: {acuraciaSGD_treino:.4f}")
print(f"SGD - Acurácia no teste: {acuraciaSGD_teste:.4f}")

visualizar_comparacao_acuracia(
    ['Perceptron', 'SGD'],
    [acuraciaP_treino, acuraciaSGD_treino],
    [acuraciaP_teste, acuraciaSGD_teste],
    salvar_arquivo='analise_sentimento_comparacao_acuracia.png'
)

print(f"[Tempo] Tempo total de execução: {time.perf_counter() - inicio_total:.4f}s")