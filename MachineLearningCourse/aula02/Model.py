import numpy as np
from scipy import sparse


def PerceptronBase(features, labels, T):
    is_sparse = sparse.issparse(features)
    n = features.shape[0]
    theta = np.zeros(features.shape[1])
    theta_0 = 0.0
    rng = np.random.default_rng(42)

    for _ in range(T):
        for i in rng.permutation(n):
            if is_sparse:
                row = features.getrow(i)
                margin = row.dot(theta)[0] + theta_0
            else:
                margin = np.dot(theta, features[i]) + theta_0

            if labels[i] * margin <= 0:
                if is_sparse:
                    theta[row.indices] += labels[i] * row.data
                else:
                    theta += labels[i] * features[i]
                theta_0 += labels[i]

    return theta, theta_0


def SGD(x, y, T, eta_0, lambda_):
    is_sparse = sparse.issparse(x)
    n = x.shape[0]
    theta = np.zeros(x.shape[1])
    theta_0 = 0.0
    rng = np.random.default_rng(314)
    t = 0

    for _ in range(T):
        for i in rng.permutation(n):
            t += 1
            eta = eta_0 / np.sqrt(t)

            if is_sparse:
                row = x.getrow(i)
                margin = y[i] * (row.dot(theta)[0] + theta_0)
            else:
                margin = y[i] * (np.dot(theta, x[i]) + theta_0)

            if margin < 1:
                theta *= (1 - eta * lambda_)
                if is_sparse:
                    theta[row.indices] += eta * y[i] * row.data
                else:
                    theta += eta * y[i] * x[i]
                theta_0 += eta * y[i]
            else:
                theta *= (1 - eta * lambda_)

    return theta, theta_0


def Predicao(theta, theta_0, x):
    if sparse.issparse(x):
        scores = x.dot(theta) + theta_0
    else:
        scores = x @ theta + theta_0

    return np.where(scores > 0, 1, -1)


def Acuracia(theta, theta_0, x, y):
    if x.shape[0] == 0:
        return 0.0

    preds = Predicao(theta, theta_0, x)
    return float(np.mean(preds == y))
