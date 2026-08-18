from loadDataset import carregar_dataset_bolas
from Model import PerceptronBase, SGD, Acuracia
from visualizacao import visualizar_perceptron
from sklearn.model_selection import train_test_split
import os

dataset, features, labels = carregar_dataset_bolas(os.path.join("dataset", "bolas.csv"))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

thetaP, theta_0P = PerceptronBase(X_train, y_train, 20)

thetaSGD, theta_0SGD = SGD(X_train, y_train, 50000000, 0.6, 0.4)

acuraciaP_treino = Acuracia(thetaP, theta_0P, X_train, y_train)
acuraciaSGD_treino = Acuracia(thetaSGD, theta_0SGD, X_train, y_train)

acuraciaP_teste = Acuracia(thetaP, theta_0P, X_test, y_test)
acuraciaSGD_teste = Acuracia(thetaSGD, theta_0SGD, X_test, y_test)

print(f"Perceptron - Acurácia no treino: {acuraciaP_treino:.4f}")
print(f"Perceptron - Acurácia no teste: {acuraciaP_teste:.4f}")
print(f"SGD - Acurácia no treino: {acuraciaSGD_treino:.4f}")
print(f"SGD - Acurácia no teste: {acuraciaSGD_teste:.4f}")

visualizar_perceptron(X_train, y_train, thetaP, theta_0P, acuraciaP_treino, dataset_type='treino',
                     salvar_arquivo="perceptron_treino.png")
visualizar_perceptron(X_train, y_train, thetaSGD, theta_0SGD, acuraciaSGD_treino, dataset_type='treino',
                     salvar_arquivo="sgd_treino.png")

visualizar_perceptron(X_test, y_test, thetaP, theta_0P, acuraciaP_teste, dataset_type='teste',
                     salvar_arquivo="perceptron_teste.png")
visualizar_perceptron(X_test, y_test, thetaSGD, theta_0SGD, acuraciaSGD_teste, dataset_type='teste',
                     salvar_arquivo="sgd_teste.png")