import numpy as np
import pandas as pd
from data.generator import Gen_data, Data_3D
from perceptron.perception import Perceptron
from vizualização.graphics import Gráfico_2D, Gráfico_3D

interations = int(input('Quantas iterações sobre o dataset você deseja realizar? '))
flag = int(input('Deseja visualizar o gráfico 2D ou 3D? (2 ou 3) '))
if flag == 2:
    Data, Labels = Gen_data()
    theta = np.zeros(3);
    theta = Perceptron(interations, theta, 1, Data, Labels, 100);
    Gráfico_2D(Data, theta, Labels);
else: 
    Data, Labels = Data_3D()
    theta = np.zeros(4);
    theta = Perceptron(interations, theta, 1, Data, Labels, 100);
    Gráfico_3D(Data, theta, Labels);