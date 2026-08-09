import random
import numpy as np

def SGD (x: np.array, y: np.array, T: int, eta_0: float, lambda_: float):
    theta: np.array = np.zeros(x[0].shape)
    theta_0: float = 0
    for _ in range (T):
        i = random.randint(0, len(x) - 1)
        eta = eta_0 / np.sqrt(T)
        z = x[i] * (np.dot(theta, x[i] + theta_0))
        if z < 1:
            theta = theta + (eta * (x[i] * y[i]) - (eta * (lambda_ * theta[i])))
            theta_0 = theta_0 + (eta * y[i]);
        else:
            theta = theta - eta * (lambda_ * theta);
    return theta, theta_0;

def classificar(theta: np.array, theta_0: float, x: np.array):
    value = np.dot(theta, x) + theta_0

    if value <= 0: 
        return -1
    else:
        return 1

def acuracia(theta, theta_0, x, y):
    acertos: int = 0

    for i in range(len(x)):
        if classificar(theta, theta_0, x[i]) == y[i]:
            acertos+=1
    return acertos/len(x);

