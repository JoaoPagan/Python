########################################################################################################################
################################## LISTA DE EXERCÍCIOS - REO 3 #########################################################
########################################################################################################################

'''
EXERCÍCIO 1 - Selecione uma imagem a ser utilizada no trabalho prático e realize os seguintes processos utilizando o
pacote OPENCV e Scikit-Image do Python:
a) Aplique o filtro de média com cinco diferentes tamanhos de kernel e compare os resultados com a imagem original.
'''

#bibliotecas
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

#Carregamento da imagem
img = "17.jpg"
img_bgr = cv2.imread(img, 1) #leitura colorida,em BGR
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#Filtro de média - convolução
img_fm1 = cv2.blur(img_rgb,(11,11))
img_fm2 = cv2.blur(img_rgb,(31,31))
img_fm3 = cv2.blur(img_rgb,(51,51))
img_fm4 = cv2.blur(img_rgb,(71,71))
img_fm5 = cv2.blur(img_rgb,(91,91))

'''
#imagens
plt.figure("Filtros de média")
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title("Imagem original RGB")

plt.subplot(2,3,2)
plt.imshow(img_fm1)
plt.xticks([])
plt.yticks([])
plt.title("Filtro 11x11")

plt.subplot(2,3,3) 
plt.imshow(img_fm2)
plt.xticks([])
plt.yticks([])
plt.title("Filtro 31x31")

plt.subplot(2,3,4)
plt.imshow(img_fm3)
plt.xticks([])
plt.yticks([])
plt.title("Filtro 51x51")

plt.subplot(2,3,5)
plt.imshow(img_fm4)
plt.xticks([])
plt.yticks([])
plt.title("Filtro 71x71")

plt.subplot(2,3,6)
plt.imshow(img_fm5)
plt.xticks([])
plt.yticks([])
plt.title("Filtro 91x91")
plt.show()
'''
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")


#b) Aplique diferentes tipos de filtros com pelo menos dois tamanhos de kernel e compare os resultados entre si e com a
# imagem original.


#Filtro Média
img_media1 = cv2.blur(img_rgb,(21,21))
img_media2 = cv2.blur(img_rgb,(51,51))

#Filtro Gaussiano
img_gaussian1 = cv2.GaussianBlur(img_rgb,(21,21),0)
img_gaussian2 = cv2.GaussianBlur(img_rgb,(51,51),0)

#Filtro Mediana
img_mediana1 = cv2.medianBlur(img_rgb,21)
img_mediana2 = cv2.medianBlur(img_rgb,51)

#Filtro Bilateral
img_bi1 = cv2.bilateralFilter(img_rgb,21,21,11)
img_bi2 = cv2.bilateralFilter(img_rgb,51,51,31)

'''
#imagens
plt.figure("Média")
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title("Imagem original")

plt.subplot(1,3,2)
plt.imshow(img_media1)
plt.xticks([])
plt.yticks([])
plt.title("Filtro média 21x21")

plt.subplot(1,3,3)
plt.imshow(img_media2)
plt.xticks([])
plt.yticks([])
plt.title("Filtro média 91x91")
plt.show()

plt.figure("Gaussiana")
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title("Imagem original")

plt.subplot(1,3,2)
plt.imshow(img_gaussian1)
plt.xticks([])
plt.yticks([])
plt.title("Filtro Gaussiano 21x21")

plt.subplot(1,3,3)
plt.imshow(img_gaussian2)
plt.xticks([])
plt.yticks([])
plt.title("Filtro Gaussiano 91x91")
plt.show()

plt.figure("Mediana")
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title("Imagem original")

plt.subplot(1,3,2)
plt.imshow(img_mediana1)
plt.xticks([])
plt.yticks([])
plt.title("Filtro mediana 21x21")

plt.subplot(1,3,3)
plt.imshow(img_mediana2)
plt.xticks([])
plt.yticks([])
plt.title("Filtro mediana 91x91")
plt.show()


plt.figure("Bilateral")
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title("Imagem original")

plt.subplot(1,3,2)
plt.imshow(img_bi1)
plt.xticks([])
plt.yticks([])
plt.title("Filtro bilateral 21x21")

plt.subplot(1,3,3)
plt.imshow(img_bi2)
plt.xticks([])
plt.yticks([])
plt.title("Filtro bilateral 91x91")
plt.show()
'''
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

#c) Realize a segmentação da imagem utilizando o processo de limiarização. Utilizando o reconhecimento de contornos,
#identifique e salve os objetos de interesse. Além disso, acesse as bibliotecas Opencv e Scikit-Image, verifique as
#variáveis que podem ser mensuradas e extraia as informações pertinentes (crie e salve uma tabela com estes dados).
#Apresente todas as imagens obtidas ao longo deste processo.


##Testando os canais RGB
r,g,b = cv2.split(img_rgb)

##Filtro mediana nos canais r,g,b
r_mediana = cv2.medianBlur(r,9)
g_mediana = cv2.medianBlur(g,9)
b_mediana = cv2.medianBlur(b,9)

##Testando os canais HSV
img_hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(img_hsv)

#Filtro mediana nos canais h,s,v
h_mediana = cv2.medianBlur(h,9)
s_mediana = cv2.medianBlur(s,9)
v_mediana = cv2.medianBlur(v,9)

imagens = [r,g,b,r_mediana, g_mediana,b_mediana]
titulos = ['R','G','B', 'R_mediana', 'G_mediana', 'B_mediana']

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(imagens[i], cmap='gray')
    plt.xticks([]);plt.yticks([])
    plt.title(titulos[i])
plt.show()

imagens2 = [h,s,v,h_mediana, s_mediana,v_mediana]
titulos2 = ['H','S','V', 'H_mediana', 'S_mediana', 'V_mediana']

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(imagens2[i], cmap='gray')
    plt.xticks([]);plt.yticks([])
    plt.title(titulos2[i])
plt.show()

#O canal S é o mais indicado para a segmentação da imagem.

#Histograma e limiarização do canal S original
hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
l, img_l = cv2.threshold(s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#l = limiar; img_l = imagem binarizada com o threshold
#Histograma e limiarização do canal S com filtro mediana
hist_s_mediana=cv2.calcHist([s_mediana],[0],None,[256],[0,256])
l_f, img_l_f=cv2.threshold(s_mediana,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

'''
#imagens
plt.figure("Histogramas")
plt.subplot(2,3,1)
plt.imshow(s, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Canal S sem filtro")

plt.subplot(2,3,2)
plt.plot(hist_s, color='black')
plt.axvline(x=l, color='red')
plt.title("L: "+str(l))
plt.xlim([0,256])
plt.ylabel("Número de Pixels")

plt.subplot(2,3,3)
plt.imshow(img_l,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Máscara')

plt.subplot(2,3,4)
plt.imshow(s_mediana,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Canal S com filtro')

plt.subplot(2,3,5)
plt.plot(hist_s_mediana,color = 'black')
plt.axvline(x=l_f,color = 'red')
plt.title("L: " + str(l_f))
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,6)
plt.imshow(img_l_f,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Máscara')
plt.show()
'''

#Segmentação
img_segmentada = cv2.bitwise_and(img_rgb, img_rgb, mask=img_l_f) #imagem colorida sem fundo

#Objetos de interesse - folhas e lesões
## Folhas - canal s
cnts = cv2.findContours(img_l_f,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1] #interesse apenas nos contornos; opção - função 'grabcountours'


## Lesoes - canal Cr
folhas_ycrcb = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2YCrCb)
y,cr,cb = cv2.split(folhas_ycrcb)
limiar,img_lim_cr = cv2.threshold(cr,135,255,cv2.THRESH_BINARY)


dimensoes=[]
for (i,c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    obj_rgb= img_segmentada[y:y+h,x:x+w]
    obj_bgr = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2BGR) #transforma em BRG pra salvar
    cv2.imwrite(f'folha{i+1}.png',obj_bgr)
    area = cv2.contourArea(c)
    area_count = cv2.countNonZero(obj_rgb[:,:,1])
    razao = (h/w).__round__(2)

    #Lesoes
    les = img_lim_cr[y:y + h, x:x + w]
    cv2.imwrite(f'Lesao_folha{i+1}.png', les)
    area_lesao = cv2.countNonZero(les)
    contorno = cv2.findContours(les, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno = contorno[0] if len(contorno) == 2 else contorno[1]
    contorno = len(contorno)
    razao_lesao = ((area_lesao/area_count)*100).__round__(2)
    dimensoes += [[str(i + 1), str(h), str(w), str(area), str(razao),
               str(area_lesao), str(contorno), str(razao_lesao)]]
    cv2.namedWindow('Folha', cv2.WINDOW_NORMAL)
    cv2.imshow('Folha', les)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Criando uma tabela com a biblioteca Pandas
import pandas as pd
tabela = pd.DataFrame(dimensoes)
tabela = tabela.rename(columns={0:'FOLHA',
                              1: 'ALTURA_FOLHA',
                              2:'LARGURA_FOLHA',
                              3:'AREA_FOLHA',
                              4:'RAZAO_FOLHA',
                              5:'AREA_LESÃO',
                              6:'NUMERO DE PUSTULAS',
                              7:'RAZAO DA LESÃO'})
tabela.to_csv('tabela.csv',index=False)

print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")


#d) Utilizando máscaras, apresente o histograma somente dos objetos de interesse.

red = cv2.calcHist([img_rgb],[0],img_l,[256],[0,255])
green = cv2.calcHist([img_rgb],[1],img_l,[256],[0,255])
blue = cv2.calcHist([img_rgb],[2],img_l,[256],[0,255])

plt.subplot(3,1,1)
plt.plot(red,color ='r')
plt.xticks([])
plt.title('Red')

plt.subplot(3,1,2)
plt.plot(green,color ='g')
plt.xticks([])
plt.title('Green')

plt.subplot(3,1,3);plt.plot(blue,color ='b')
plt.title('Blue')
plt.show()

print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")


#e) Realize a segmentação da imagem utilizando a técnica de k-means. Apresente as imagens obtidas neste processo.

print('INFORMAÇÕES')
print('Dimensão: ',np.shape(img_rgb))
print(np.shape(img_rgb)[0], ' x ',np.shape(img_rgb)[1], ' = ', np.shape(img_rgb)[0] * np.shape(img_rgb)[1])


#formatação
pixel_values = img_rgb.reshape((-1,3))
pixel_values = np.float32(pixel_values)

# kmeans

# Critério de Parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k=2
dist, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print('SQ das Distâncias de Cada Ponto ao Centro: ', dist)
print('Dimensão labels: ', labels.shape)
print('Valores únicos: ',np.unique(labels))
print('Tipo labels: ', type(labels))

# flatten the labels array
labels = labels.flatten()

print('Dimensão flatten labels: ', labels.shape)
print('Tipo labels (f): ', type(labels))

val_unicos,contagens = np.unique(labels,return_counts=True)
val_unicos = np.reshape(val_unicos,(len(val_unicos),1))
contagens = np.reshape(contagens,(len(contagens),1))

#valores unicos em uma coluna e outra coluna com a contagem
hist = np.concatenate((val_unicos,contagens),axis=1) #axis=1 >> cbind
print("Hist")
print(hist)

#conversao de decimal pra uint8 >> cada classe: [x,y,z] - contagem do R,G,B
centers2 = np.uint8(centers)
print("Centroides decimais")
print(centers)
print("Centroides inteiros")
print(centers2)


#conversao dos pixels para as cores dos centroides
matriz_segmentada = centers2[labels]


print('Dimensão Matriz Segmentada: ',matriz_segmentada.shape)
print('Matriz Segmentada')

#mostrando um pedaço dessa matriz
print(matriz_segmentada[0:5,:]) #'0:5'mostra as linhas 0 a 4 e todas as colunas


# Reformatar a matriz na imagem de formato original
img_segmentada = matriz_segmentada.reshape(img_rgb.shape)

print("Dimensão img_segmentada:", np.shape(img_segmentada))

# Grupo 1
original_01 = np.copy(img_rgb)
matriz_01 = original_01.reshape((-1, 3))
matriz_01[labels != 0] = [0, 0, 0]
img_final_01 = matriz_01.reshape(img_rgb.shape)

# Grupo 2
original_02 = np.copy(img_rgb)
matriz_02 = original_02.reshape((-1, 3))
matriz_02[labels != 1] = [0, 0, 0]
img_final_02 = matriz_02.reshape(img_rgb.shape)

'''
#imagens
# Apresentar Imagem
plt.figure('Imagens')
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title('ORIGINAL')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(img_segmentada)
plt.title('ROTULOS')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(img_final_01)
plt.title('Grupo 1')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(img_final_02)
plt.title('Grupo 2')
plt.xticks([])
plt.yticks([])
plt.show()
'''
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

#f) Realize a segmentação da imagem utilizando a técnica de watershed. Apresente as imagens obtidas neste processo.

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

plt.imshow(img_segmentada)
plt.show()

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
ret, sure_fg = cv2.threshold(dist_transform,0.75*dist_transform.max(),255,0)

########################################################################################################################
########################################################################################################################
