from loadDataset import carregar_dataset_sentimento
from BagOfWords import create_bow_vectorizer, text_to_bow_vector, GetFeatures
from sklearn.model_selection import train_test_split
from visualizacao import visualizar_comparacao_acuracia

from Model import SGD, Acuracia, PerceptronBase

import os

texts, labels = carregar_dataset_sentimento(os.path.join("dataset", "reviewes.csv"))

vectorizer = create_bow_vectorizer(texts) 

features = GetFeatures(texts, vectorizer) 

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

thetaP, theta_0P = PerceptronBase(X_train, y_train, 20)

thetaSGD, theta_0SGD = SGD(X_train, y_train, 50000000, 0.6, 0.04)


acuraciaP_treino = Acuracia(thetaP, theta_0P, X_train, y_train)
acuraciaSGD_treino = Acuracia(thetaSGD, theta_0SGD, X_train, y_train)

acuraciaP_teste = Acuracia(thetaP, theta_0P, X_test, y_test)
acuraciaSGD_teste = Acuracia(thetaSGD, theta_0SGD, X_test, y_test)

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