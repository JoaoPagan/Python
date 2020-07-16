###########################################################################

#   Resolução REO 1 - João Pedro Gomes Pagan 2019160285
#   06/07/2020
#   e-mail: pagan_jp@hotmail.com
#   GITHUB: JoaoPagan

############################################################################

'''
print(' ')
print(' ')
print(' ')


print('EXERCÍCIO 01')
print(' ')

# a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.

import numpy as np
lista = [43.5,150.30,17,28,35,79,20,99.07,15]
print('lista: ' +str(lista))
print('tipo: ')
print(type(lista))
vetor = np.array(lista)
print('vetor: ' +str(vetor))
print('tipo: ')
print(type(vetor))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor

print('vetor: ' +str(vetor))
dim_vetor = len(vetor)
print('dimensão: ' +str(dim_vetor))
media_vetor = np.mean(vetor)
print('média: ' +str(media_vetor))
maximo_vetor = np.max(vetor)
print('máximo: ' +str(maximo_vetor))
minimo_vetor = np.min(vetor)
print('mínimo: ' +str(minimo_vetor))
var_vetor = np.var(vetor)
print('variância: ' +str(var_vetor))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença entre cada elemento do vetor declarado
# na letra a e o valor da média deste.

vetor_novo = (vetor-media_vetor)
vetor_novo_quadrado = (vetor_novo**2)
print('Vetor novo: ' +str(vetor_novo_quadrado))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

#d) Obtenha um novo vetor que contenha todos os valores superiores a 30.

bool_maior_30 = vetor>30
print(bool_maior_30)
vetor_maior_30 = vetor[bool_maior_30]
print('Novo vetor com valores maiores que 30: ' +str(vetor_maior_30))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# e) Identifique quais as posições do vetor original possuem valores superiores a 30

pos_maior_30 = np.where(vetor>30)
print('Posições dos valores maiores que 30 no vetor original: ' +str(pos_maior_30))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.

print(vetor)
vetor_primeiro_quinto_ultimo = vetor[0],vetor[4],vetor[-1]
print('Vetor com primeiro quinto e último valor: ' +str(vetor_primeiro_quinto_ultimo))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua respectiva posição durante as iterações

import time

print('Vetor: ' +str(vetor))
print('------------------------------------------------------------------------------------------------------------')

it = 0
for i in range(0,len(vetor),1):
    it = it + 1
    print('Iteração: ' + str(it))
    print('Na posição ' + str(i) + ' há o elemento: ' + str(vetor[int(i)]))
    time.sleep(0.75)
    print('--------------------------------------------------------------------------------------------------------')

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# h) Crie uma estrutura de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.

import time

print('Vetor: ' +str(vetor))
print('-----------------------------------------------------------------------------------------------------------')

def somatorio_quadrado (vetor):
    somador = 0
    it = 0
    for el in vetor:
        print('Elemento: {}'.format(el))
        print('Somador: {}'.format(somador))
        somador = somador + el**2
        it = it + 1
        print('Iteração {} - Somatório: {}'.format(it, somador))
        print('------------------------------------------------------------------------------------')
        time.sleep(0.75)
    return somador
print(vetor)
print(' ')
somatorio_quadrado(vetor)

print(' ')

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor

print('Cria uma lista de valores de 0 a 15')
valores = -1
while valores != 15:
    valores = valores + 1
    print(valores)

print('----------------------------------------------------------------------------------------------')
print(' ')

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.

print('Sequência de valores: {}'.format(list(range(1, len(vetor), 1))))

print('-----------------------------------------------------------------------------------------------')

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

#l) Concatene o vetor da letra a com o vetor da letra j.

conca = list(range(1, len(vetor), 1))
vetores_concatenados = np.concatenate((vetor, conca), axis=0)
print('vetores concatenados: ' +str(vetores_concatenados))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')
print(' ')
print(' ')
print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')
print(' ')
print( 'EXERCÍCIO 2')
print(' ')

#a) Declare a matriz abaixo com a biblioteca numpy.
# 1 3 22
# 2 8 18
# 3 4 22
# 4 1 23
# 5 2 52
# 6 2 18
# 7 2 25

import numpy as np

matriz = np.array([[1,3,22],[2,8,18],[3,4,22],[4,1,23],[5,2,52],[6,2,18],[7,2,25]])
print('matriz: ')
print(matriz)

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# b) Obtenha o número de linhas e de colunas desta matriz

nl, nc = np.shape(matriz)
print('Número de linhas da matriz: ' +str(nl))
print('Número de colunas da matriz: ' +str(nc))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# c) Obtenha as médias das colunas 2 e 3.

submatriz_c2 = matriz[:,1]
media_c2 = np.average(submatriz_c2)

submatriz_c3 = matriz[:,2]
media_c3 = np.average(submatriz_c3)
print('Média da coluna 2: ' +str(media_c2))
print('Média da coluna 3: ' +str(media_c3))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# d) Obtenha as médias das linhas considerando somente as colunas 2 e 3

submatriz_l1 = matriz[0,[1,2]]
media_l1 = np.average(submatriz_l1)
print('Média da linha 1 considerando as colunas 2 e 3: ' +str(media_l1))
submatriz_l2 = matriz[1,[1,2]]
media_l2 = np.average(submatriz_l2)
print('Média da linha 2 considerando as colunas 2 e 3: ' +str(media_l2))
submatriz_l3 = matriz[2,[1,2]]
media_l3 = np.average(submatriz_l3)
print('Média da linha 3 considerando as colunas 2 e 3: ' +str(media_l3))
submatriz_l4 = matriz[3,[1,2]]
media_l4 = np.average(submatriz_l4)
print('Média da linha 4 considerando as colunas 2 e 3: ' +str(media_l4))
submatriz_l5 = matriz[4,[1,2]]
media_l5 = np.average(submatriz_l5)
print('Média da linha 5 considerando as colunas 2 e 3: ' +str(media_l5))
submatriz_l6 = matriz[5,[1,2]]
media_l6 = np.average(submatriz_l6)
print('Média da linha 6 considerando as colunas 2 e 3: ' +str(media_l6))
submatriz_l7 = matriz[6,[1,2]]
media_l7 = np.average(submatriz_l7)
print('Média da linha 7 considerando as colunas 2 e 3: ' +str(media_l7))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade inferior a 5.

col_2 = (matriz[:,1])
notas_menor_5 = np.where(col_2<5)
print('Posições dos genótipos que possuem notas de severidade a doença inferiores a 5: ' +str(notas_menor_5[0]))

bol_notas_menor_5 = col_2<5
col_1 = (matriz[:, 0])
genotipos_notas_menor_5 = col_1[bol_notas_menor_5]
print('Genótipos com notas inferiores a 5: ' +str(genotipos_notas_menor_5))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# f) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de peso de 100 grãos superior ou igual a 22.

col_3 = (matriz[:, 2])
notas_maiorigual_22 = np.where(col_3>=22)
print('Posições dos genótipos que possuem notas de peso de 100 grãos superior ou igual a 22: ' +str(notas_maiorigual_22[0]))

bol_notas_maiorigual_22 = col_3>=22
col_1 = (matriz[:, 0])
genotipos_notas_maiorigual_22 = col_1[bol_notas_maiorigual_22]
print('Genótipos com notas de peso de 100 grãos superior ou igual a 22: ' +str(genotipos_notas_maiorigual_22))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

#g) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade igual ou inferior a 3 e peso de 100
# grãos igual ou superior a 22.

col_2 = (matriz[:,1])
notas_menorigual_3 = np.where(col_2<=3)
bol_notas_menorigual_3 = col_2<=3
col_1 = (matriz[:, 0])
genotipos_notas_menorigual_3 = col_1[bol_notas_menorigual_3]

print('Posições dos genótipos que possuem nota inferior ou igual a 3 são: ' +str(notas_menorigual_3[0]))
print('Posições dos genótipos com peso de 100 grãos superior ou igual a 22: ' +str(notas_maiorigual_22[0]))
genotipos_q2g = col_1[bol_notas_maiorigual_22 & bol_notas_menorigual_3]
print('Genótipos com peso de 100 grãos superior ou igual a 22 e nota de severidade de doença inferior ou igual a 3: ' +str(genotipos_q2g))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada uma das posições da matriz e o seu
#  respectivo valor. Utilize um iterador para mostrar ao usuário quantas vezes está sendo repetido.
#  Apresente a seguinte mensagem a cada iteração "Na linha X e na coluna Y ocorre o valor: Z".
#  Nesta estrutura crie uma lista que armazene os genótipos com peso de 100 grãos igual ou superior a 25

matriz_pesomaior_25 = []
iteration = 0
for i in np.arange(0, nl, 1):
    for j in np.arange(0, nc, 1):
        iteration += 1
        print('Iteração: ' + str(iteration))
        print('Na linha ' + str(i+1) +
              ' e coluna ' + str(j+1) +
              ' ocorre o valor: ' + str(matriz[int(i), int(j)]))
        matriz_pesomaior_25 = (matriz[:, 2] >= 25)
        matriz_25 = (matriz[matriz_pesomaior_25])

print('----------------------------------------------------------------------------------------------------')
print('Os genótipos ' + str(matriz_25[:, 0]) + ' apresentam peso de 100 grãos maior ou igual a 25')

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

print(' ')
print(' ')
print(' ')
print(' ')

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')
print('EXERCÍCIO 3')
print(' ')
# a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral um vetor qualquer, baseada em um loop (for).

print('A função esta no arquivo funcoes_joap.py ')

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')

# b) Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal com média 100 e variância 2500. Pesquise na documentação do numpy por funções de simulação.

import numpy as np

vetor_10= np.random.normal(loc=100, scale=50, size=10)
print('Vetor com 10 amostras aleatórias de média 100 e variância de 2500: ' +str(vetor_10))
vetor_100 = np.random.normal(loc=100, scale=50, size=100)
print('Vetor com 100 amostras aleatórias de média 100 e variância de 2500: ' +str(vetor_100))
vetor_1000 = np.random.normal(loc=100, scale=50, size=1000)
print("Vetor com 1000 amostras aleatórias de média 100 e variância de 2500: " +str(vetor_1000))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')
# c) Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b.
from funcoes_joaop import media
from funcoes_joaop import variancia
print('Vetor com 10 amostras: ')
print('média: ' +str(media(vetor_10)))
print('variância: ' +str(variancia(vetor_10)))
print('Vetor com 100 amostras')
print('media ' +str(media(vetor_100)))
print('variancia ' +str(variancia(vetor_100)))
print('Vetor com 1000 amostras')
print('media ' +str(media(vetor_1000)))
print("variancia " +str(variancia(vetor_1000)))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')

# d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000.

vetor_100000 = np.random.normal(loc=100, scale=50, size=100000)

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

histogra_exemplo = plt.hist(vetor_1000, bins=15)
plt.hist(vetor_10, bins=15)
plt.hist(vetor_100, bins=15)
plt.hist(vetor_1000, bins=15)
plt.hist(vetor_100000, bins=15)
fig, axs = plt.subplots(1, tight_layout=True)
N, bins, patches = axs.hist(vetor_10, bins=5)
fracs = N / N.max()

norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.title('Histograma 10')
plt.xlabel('Número de elementos na classe')
plt.ylabel('Valor médio da classe')


fig, axs = plt.subplots(1, tight_layout=True)
N, bins, patches = axs.hist(vetor_100, bins=10)
fracs = N / N.max()

norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.title('Histograma 100')
plt.xlabel('Número de elementos na classe')
plt.ylabel('Valor médio da classe')


fig, axs = plt.subplots(1, tight_layout=True)
N, bins, patches = axs.hist(vetor_1000, bins=15)
fracs = N / N.max()

norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.title('Histograma 1000')
plt.xlabel('Número de elementos na classe')
plt.ylabel('Valor médio da classe')


fig, axs = plt.subplots(1, tight_layout=True)

N, bins, patches = axs.hist(vetor_100000, bins=70)

fracs = N / N.max()

norm = colors.Normalize(fracs.min(), fracs.max())

plt.show()

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

print(' ')
print(' ')
print(' ')
print(' ')
'''
print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')
print('EXERCÍCIO 4')
print(' ')

# a) O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) em repetições (segunda coluna) quanto a quatro
# variáveis (terceira coluna em diante). Portanto, carregue o arquivo dados.txt com a biblioteca numpy, apresente os dados
# e obtenha as informações de dimensão desta matriz.

import numpy as np

dados = np.loadtxt('dados.txt')
print('Dados: ' +str(dados))

nl,nc = np.shape(dados)

print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')

# b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy

help(np.unique)
help(np.where)

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')

# c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas

print('Genótipos: ')

genotipos = np.unique(dados[0:30,0:1], axis=0)
nlg,ncg = np.shape(genotipos)

print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))

print(np.unique(dados[0:30,0:1], axis=0))
print('Repetições: ')
print(np.unique(dados[0:30,1:2], axis=0))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')

# d) Apresente uma matriz contendo somente as colunas 1, 2 e 4

submatriz_124 = dados[:,[0,1,3]]
print('Submatriz com colunas 1, 2 e 4: ' +str(submatriz_124))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')

# e) Obtenha uma matriz que contenha o máximo, o mínimo, a média e a variância de cada genótipo para a variavel da coluna 4.
# Salve esta matriz em bloco de notas.

minimos = np.zeros((nlg,1))
maximos = np.zeros((nlg,1))
medias = np.zeros((nlg,1))
variancias = np.zeros((nlg,1))
it=0
for i in np.arange(0,nl,3):

    minimos[it,0] = np.min(submatriz_124[i:i + 3, 2], axis=0)
    maximos[it,0] = np.max(submatriz_124[i:i + 3, 2], axis=0)
    medias[it,0] = np.mean(submatriz_124[i:i + 3, 2], axis=0)
    variancias[it,0] = np.var(submatriz_124[i:i + 3, 2], axis=0)
    it = it + 1

matriz_concatenada = np.concatenate((genotipos,minimos,maximos,medias,variancias),axis=1)
print('Genótipos     Min     Max      Média    Variância')
print(matriz_concatenada)
np.savetxt('matriz_ex4.txt', matriz_concatenada, fmt='%10.2f', delimiter=' ')

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')

# f) Obtenha os genótipos que possuem média (médias das repetições) igual ou superior a 500 da matriz gerada na letra anterior.

dados2 = np.loadtxt('matriz_ex4.txt')
genotipos_maior_500 = np.squeeze (np.asarray(dados2[:,3]))>=500
print('Genótipos que possuem média maior ou igual a 500: ' +str(dados2[genotipos_maior_500,0]))

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')

# g) Apresente os seguintes graficos:
#    - Médias dos genótipos para cada variável. Utilizar o comando plt.subplot para mostrar mais de um grafico por figura

from matplotlib import pyplot as plt

dados = np.loadtxt('dados.txt')
media1 = np.zeros((nlg,1))
media2 = np.zeros((nlg,1))
media3 = np.zeros((nlg,1))
media4 = np.zeros((nlg,1))
media5 = np.zeros((nlg,1))
it=0
for i in np.arange(0,30,3): #percorre as 30 linhas do vetor original de acordo com o numero de repetições
    media1[it,0] = np.mean(dados[i:i + 3, 2], axis=0)
    media2[it,0] = np.mean(dados[i:i + 3, 3], axis=0)
    media3[it,0] = np.mean(dados[i:i + 3, 4], axis=0)
    media4[it,0] = np.mean(dados[i:i + 3, 5], axis=0)
    media5[it,0] = np.mean(dados[i:i + 3, 6], axis=0)
    it = it + 1

dados_medias = np.concatenate((genotipos,media1,media2,media3,media4,media5),axis=1)
nl,nc = np.shape(dados_medias)

plt.style.use('ggplot')
plt.figure('Gráfico Médias')
plt.subplot(2,3,1)
plt.bar(dados_medias[:,0],dados_medias[:,1])
plt.title('Variável 1')
plt.xticks(dados_medias[:,0])

plt.subplot(2,3,2)
plt.bar(dados_medias[:,0],dados_medias[:,2])
plt.title('Variável 2')
plt.xticks(dados_medias[:,0])

plt.subplot(2,3,3)
plt.bar(dados_medias[:,0],dados_medias[:,3])
plt.title('Variável 3')
plt.xticks(dados_medias[:,0])

plt.subplot(2,3,4)
plt.bar(dados_medias[:,0],dados_medias[:,4])
plt.title('Variável 4')
plt.xticks(dados_medias[:,0])

plt.subplot(2,3,5)
plt.bar(dados_medias[:,0],dados_medias[:,5])
plt.title('Variável 5')
plt.xticks(dados_medias[:,0])
plt.show()

#    - Disperão 2D da médias dos genótipos (Utilizar as três primeiras variáveis). No eixo X uma variável e no eixo Y outra.

plt.style.use('ggplot')
fig = plt.figure('Disperão 2D da médias dos genótipos')
plt.subplot(2,2,1)

cores = ['yellow','red','green','black','pink','black','orange','cyan','slategray','darkviolet']

for i in np.arange(0,nl,1):
    plt.scatter(dados_medias[i,1], dados_medias[i,2],s=50,alpha=0.8,label = dados_medias[i,0],c = cores[i])

plt.xlabel('Var 1')
plt.ylabel('Var 2')
plt.subplot(2,2,2)
for i in np.arange(0,nl,1):
    plt.scatter(dados_medias[i,2], dados_medias[i,3],s=50,alpha=0.8,label = dados_medias[i,0],c = cores[i])

plt.xlabel('Var 2')
plt.ylabel('Var 3')
plt.subplot(2,2,3)
for i in np.arange(0,nl,1):
    plt.scatter(dados_medias[i,1], dados_medias[i,3],s=50,alpha=0.8,label = dados_medias[i,0],c = cores[i])

plt.xlabel('Var 1')
plt.ylabel('Var 3')
plt.legend(bbox_to_anchor=(2.08, 0.7), title='Genotipos', borderaxespad=0., ncol=5)
plt.show()

print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print(' ')

#'''