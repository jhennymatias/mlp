# Estrutura perceptron
# 1- Inicializaçao w = 0
# 2- Ativação e treinamento
# 3 - Repita o passo 2 k = 0  

import numpy as np
# defina o numero de epocas e o numero de amostras

numEpoca = 63000
q = 6


# Atributos

peso 	= np.array([113, 122, 107, 98, 115, 120])
pH 		= np.array([6.8, 4.7, 5.2, 3.6, 2.9,  4.2])

#Bias
bias = 1

#Entrada do perceptron

x = np.vstack((peso,pH))
y = np.array([-1, 1, -1, -1, 1, 1])

#Taxa de aprendizado
eta = 0.6

#Define o vetor peso

w = np.zeros([1,3])

#erros

e = np.zeros(6)

def funcaoAtivacao(valor):
	#degrau bipolar
	if valor < 20:
		return(-1)
	else:
		return(1)

for j in range(numEpoca):
	for k in range(q):
		#insere bias no vetor de entrada
		xb  =  np.hstack((bias, x[:,k]))

		#calcula o campo induzido
		v = np.dot(w,xb)

		#calcula a saida do perceptron
		yr = np.heaviside(v, 1)

		#calcula erro
		e[k] = y[k] - yr


		w = w + eta*e[k]*xb

print("Vetor de erros " +str(e)) 
