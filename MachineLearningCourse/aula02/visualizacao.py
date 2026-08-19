import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def visualizar_perceptron(features, labels, theta, theta_0, acuracia, dataset_type='treino', mostrar=True, salvar_arquivo=None):
    """
    Visualiza o resultado do Perceptron, mostrando os pontos de dados e a reta de decisão.
    
    Args:
        features (np.array): Array 2D de features
        labels (np.array): Array 1D de classes (+1 ou -1)
        theta (np.array): Vetor de pesos do Perceptron
        theta_0 (float): Bias do Perceptron
        acuracia (float): Acurácia do modelo
        dataset_type (str): 'treino' ou 'teste' para rótulos de gráfico
        mostrar (bool): Se True, mostra o gráfico com plt.show()
        salvar_arquivo (str, optional): Se fornecido, salva a imagem no caminho especificado
    
    Returns:
        plt.Figure: A figura do matplotlib gerada
    """
    # Preparando o plot
    fig = plt.figure(figsize=(10, 6))

    # Criando arrays separados para os exemplos de cada classe
    exemplos_positivos = features[labels == 1]
    exemplos_negativos = features[labels == -1]

    # Plotando os pontos de cada classe com cores diferentes
    plt.scatter(exemplos_positivos[:, 0], exemplos_positivos[:, 1], color='blue', label='Classe +1')
    plt.scatter(exemplos_negativos[:, 0], exemplos_negativos[:, 1], color='red', label='Classe -1')

    # Plotando a reta de decisão
    # Verificando se theta[1] não é zero para evitar divisão por zero
    if theta[1] != 0:
        # Encontrando os limites min e max para x
        x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        
        # Calculando os valores de y correspondentes
        x_valores = np.array([x_min, x_max])
        y_valores = -(theta[0] * x_valores + theta_0) / theta[1]
        
        # Plotando a reta de decisão
        plt.plot(x_valores, y_valores, 'g-', label=f'Reta de Decisão: {theta[0]:.2f}*x + {theta[1]:.2f}*y + {theta_0:.2f} = 0')
    else:
        # Se theta[1] for zero, a reta é vertical
        x_valor = -theta_0 / theta[0]
        plt.axvline(x=x_valor, color='g', label=f'Reta de Decisão: x = {x_valor:.2f}')

    # Adicionando detalhes ao plot
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Classificação do Perceptron ({dataset_type.capitalize()}) - Acurácia: {acuracia:.2%}')
    plt.legend()
    plt.grid(True)

    # Mostrando o valor de theta e theta_0
    plt.figtext(0.5, 0.01, f'Dataset: {dataset_type.capitalize()} | Theta: [{theta[0]:.4f}, {theta[1]:.4f}], Theta_0: {theta_0:.4f}', 
                ha='center', fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

    plt.tight_layout()
    
    # Salvar a figura se um caminho for fornecido
    if salvar_arquivo:
        plt.savefig(salvar_arquivo)
    
    # Mostrar a figura se solicitado
    if mostrar:
        plt.show()
    
    return fig


def visualizar_comparacao_acuracia(model_names, train_scores, test_scores, mostrar=True, salvar_arquivo=None):
    """Exibe uma comparação de acurácia entre treino e teste para diferentes modelos."""
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(x - width / 2, train_scores, width, label='Treino', color='#4C72B0')
    plt.bar(x + width / 2, test_scores, width, label='Teste', color='#DD8452')

    plt.xticks(x, model_names)
    plt.ylim(0, 1)
    plt.ylabel('Acurácia')
    plt.title('Comparação de Acurácia: Perceptron vs SGD')
    plt.legend()

    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        plt.text(i - width / 2, train + 0.02, f'{train:.2f}', ha='center', va='bottom')
        plt.text(i + width / 2, test + 0.02, f'{test:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    if salvar_arquivo:
        plt.savefig(salvar_arquivo)

    if mostrar:
        plt.show()

    return fig


def visualizar_matrizes_confusao(model_names, labels_reais, predicoes, mostrar=True, salvar_arquivo=None):
    """Exibe uma matriz de confusão para cada modelo avaliado."""
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    axes = np.atleast_1d(axes)

    for axis, model_name, previsoes in zip(axes, model_names, predicoes):
        matriz = confusion_matrix(labels_reais, previsoes, labels=[-1, 1])
        display = ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=['-1', '1']
        )
        display.plot(ax=axis, cmap='Blues', colorbar=False)
        axis.set_title(model_name)

    fig.suptitle('Matrizes de Confusão - Conjunto de Teste')
    fig.tight_layout()

    if salvar_arquivo:
        fig.savefig(salvar_arquivo)

    if mostrar:
        plt.show()

    return fig