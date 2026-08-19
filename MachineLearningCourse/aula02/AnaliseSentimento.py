import os
import time

from sklearn.model_selection import train_test_split

from loadDataset import carregar_dataset_sentimento
from BagOfWords import create_bow_vectorizer, GetFeatures
from visualizacao import visualizar_comparacao_acuracia
from Model import SGD, Acuracia, PerceptronBase


inicio_total = time.perf_counter()

t0 = time.perf_counter()
texts, labels = carregar_dataset_sentimento(os.path.join("dataset", "reviewes.csv"))
print(f"[Tempo] Carregamento do dataset: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
vectorizer = create_bow_vectorizer(texts)
print(f"[Tempo] Criação do vetorizador (Bag of Words): {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
features = GetFeatures(texts, vectorizer)
print(f"[Tempo] Extração de features: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(f"[Tempo] Divisão treino/teste: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
thetaP, theta_0P = PerceptronBase(X_train, y_train, 30)
print(f"[Tempo] Treinamento do Perceptron: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
thetaSGD, theta_0SGD = SGD(X_train, y_train, 35, 0.3, 0.02)
print(f"[Tempo] Treinamento do SGD: {time.perf_counter() - t0:.4f}s")

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