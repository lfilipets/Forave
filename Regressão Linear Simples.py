# Regressão Linear Simples - Aula 01 dia 04/Fevereiro/2026
# UC00674 - Implementar Sistema de Aprendizagem Automática 
# Filipe Teixeira Silva
# Forave 2026

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)   
n_samples = 100

horas_funcionamiento = np.random.uniform(100, 5000, n_samples)
ruido = np.random.normal(0, 10, n_samples)
dias_ate_falha=200-0.02*horas_funcionamiento+ruido
dias_ate_falha = np.clip(dias_ate_falha, 1, None)

df=pd.DataFrame({
    'horas_funcionamiento': horas_funcionamiento,
    'dias_ate_falha': dias_ate_falha
})

print(df.head())


print(df.columns.tolist())

print("Dimensão:",df.shape)

from sklearn.model_selection import train_test_split

X=df[["horas_funcionamiento"]]
y=df["dias_ate_falha"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tamanho do conjunto de treino:", X_train.shape)
print("Tamanho do conjunto de teste:", X_test.shape)   

from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("Intercepto (b0):", modelo.intercept_)
print("Coeficientes (b1):", modelo.coef_[0])


from sklearn.metrics import r2_score, mean_absolute_error #R² e MAE são métricas comuns para avaliar a performance de modelos de regressão. R² indica a proporção da variância em y que é explicada por X, enquanto MAE é a média das diferenças absolutas entre as previsões e os valores reais.
y_pred = modelo.predict(X_test) #Gerar previsões usando o modelo treinado no conjunto de teste.

print("R²:", r2_score(y_test, y_pred)) #R² é a proporção da variância em y que é explicada por X. Varia entre 0 e 1, onde valores mais próximos de 1 indicam um modelo melhor.
print("MAE:", mean_absolute_error(y_test, y_pred)) #MAE é a média das diferenças absolutas entre as previsões e os valores reais. Valores mais baixos indicam um modelo melhor.


plt.figure(figsize=(8, 5)) #Criar uma figura com tamanho específico para a visualização.
plt.scatter(X, y, label='Dados Reais', color='blue') #Criar um gráfico de dispersão dos dados reais, onde X é o eixo x e y é o eixo y. Os pontos são coloridos de azul e rotulados como 'Dados Reais'.
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)#Gerar um array de 100 pontos igualmente espaçados entre o valor mínimo e máximo de X. O reshape(-1, 1) é necessário para transformar o array em uma matriz de uma coluna, que é o formato esperado pelo modelo para fazer previsões.
y_line = modelo.predict(x_line)#Usar o modelo treinado para prever os valores de y correspondentes aos pontos em x_line. Isso gera a linha de regressão que será plotada no gráfico.
plt.plot(x_line, y_line, label='Regressão Linear', color='red')#Plotar a linha de regressão no gráfico, onde x_line é o eixo x e y_line é o eixo y. A linha é colorida de vermelho e rotulada como 'Regressão Linear'.
plt.xlabel('Horas de Funcionamento')#Definir o rótulo do eixo x como 'Horas de Funcionamento'.
plt.ylabel('Dias até Falha')#Definir o rótulo do eixo y como 'Dias até Falha'.
plt.title('Regressão Linear Simples')#Definir o título do gráfico como 'Regressão Linear Simples'.
plt.legend()#Adicionar uma legenda ao gráfico para identificar os dados reais e a linha de regressão.
plt.show()#Exibir o gráfico gerado.


