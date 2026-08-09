import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def Gráfico_2D(Data, theta, Labels):
    """
    Plots the 2D data and the decision boundary.
    """
    plt.figure(figsize=(8, 6))

    # Plot data points based on labels
    plt.scatter(Data[0, Labels == 1], Data[1, Labels == 1], color='blue', label='Iris-versicolor')
    plt.scatter(Data[0, Labels == -1], Data[1, Labels == -1], color='red', label='Iris-setosa')

    # Plot decision boundary
    # theta[0]*x + theta[1]*y + theta[2] = 0
    # y = (-theta[0]*x - theta[2]) / theta[1]
    x_vals = np.array(plt.gca().get_xlim())
    y_vals = (-theta[0] * x_vals - theta[2]) / theta[1]
    plt.plot(x_vals, y_vals, '--', color='green', label='Decision Boundary')

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Perceptron 2D Classification on Iris Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()


def Gráfico_3D(Data, theta, Labels):
    """
    Plots the 3D data and the decision boundary.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points from the DataFrame based on labels
    ax.scatter(Data['sepal_length'][Labels == 1], Data['sepal_width'][Labels == 1], Data['petal_length'][Labels == 1], color='blue', label='Iris-versicolor')
    ax.scatter(Data['sepal_length'][Labels == -1], Data['sepal_width'][Labels == -1], Data['petal_length'][Labels == -1], color='red', label='Iris-setosa')

    # Plot decision boundary plane
    # theta[0]*x + theta[1]*y + theta[2]*z + theta[3] = 0
    # z = (-theta[0]*x - theta[1]*y - theta[3]) / theta[2]
    x_vals = np.linspace(Data['sepal_length'].min(), Data['sepal_length'].max(), 10)
    y_vals = np.linspace(Data['sepal_width'].min(), Data['sepal_width'].max(), 10)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    
    if theta[2] != 0:
        z_grid = (-theta[0] * x_grid - theta[1] * y_grid - theta[3]) / theta[2]
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, color='green', rstride=100, cstride=100)

    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Length')
    ax.set_title('Perceptron 3D Classification on Iris Dataset')
    ax.legend()
    plt.show()
