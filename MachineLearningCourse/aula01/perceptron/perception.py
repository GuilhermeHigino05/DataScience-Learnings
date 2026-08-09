import numpy as np
import pandas as pd


def Loss(theta, position, label):
    """
    Função de perda (Loss Function) para o Perceptron.
    Calcula a perda com base nos parâmetros theta, posição e rótulo.

    Args:
        theta (numpy.ndarray): Vetor de parâmetros do modelo.
        position (numpy.ndarray): Posição dos dados de entrada.
        label (int): Rótulo verdadeiro da amostra.
    Returns:
        float: Valor da perda.
    """
    aux = np.dot(theta, position)
    aux = aux * label
    return aux;

def Perceptron(T, theta, lr, data, labels, size):

    """
    Implementação do algoritmo Perceptron.

    Args:
        T (int): Número de iterações sobre o dataset.
        theta (numpy.ndarray): Vetor de parâmetros do modelo.
        lr (float): Taxa de aprendizado.
        data (numpy.ndarray or pd.DataFrame): Conjunto de dados de entrada.
        labels (numpy.ndarray): Rótulos verdadeiros para cada amostra.
        size (int): Tamanho do conjunto de dados.

    Returns:
        numpy.ndarray: Vetor atualizado de parâmetros do modelo.
    """
    # If data is a pandas DataFrame (3D case), convert and add bias
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy().T
        aux = np.ones(size)
        data = np.vstack((data, aux))
        
    #logica principal do Perceptron
    for i in range(T):
        flag = 0
        for j in range(size):
            y = labels[j]
            if Loss(theta, data[:, j], y) <= 0:
                theta = theta + y * lr * data[:, j]
                flag = 1
        if flag == 0:
            print (f'Fim do ciclo {i}')
            break
    return theta;