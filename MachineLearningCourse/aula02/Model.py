import random
import numpy as np


def PerceptronBase(features: np.ndarray, labels: np.ndarray, T: int):
    theta = np.zeros(features.shape[1])
    theta_0 = 0.0

    for _ in range(T):
        for _ in range(len(features)):
            i = random.randint(0, len(features) - 1)
            if labels[i] * (np.dot(theta, features[i]) + theta_0) <= 0:
                theta = theta + labels[i] * features[i]
                theta_0 = theta_0 + labels[i]

    return theta, theta_0


def SGD(x: np.ndarray, y: np.ndarray, T: int, eta_0: float, lambda_: float):
    theta = np.zeros(x.shape[1])
    theta_0 = 0.0

    for t in range(1, T + 1):
        i = random.randint(0, len(x) - 1)
        eta = eta_0 / np.sqrt(t)
        margin = y[i] * (np.dot(theta, x[i]) + theta_0)

        if margin < 1:
            theta = (1 - eta * lambda_) * theta + eta * y[i] * x[i]
            theta_0 = theta_0 + eta * y[i]
        else:
            theta = (1 - eta * lambda_) * theta

    return theta, theta_0


def classificar(theta: np.ndarray, theta_0: float, x: np.ndarray):
    return 1 if np.dot(theta, x) + theta_0 > 0 else -1


def Acuracia(theta, theta_0, x, y):
    if len(x) == 0:
        return 0.0

    acertos = 0
    for i in range(len(x)):
        if classificar(theta, theta_0, x[i]) == y[i]:
            acertos += 1

    return acertos / len(x)
