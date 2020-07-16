###########################################################################

#   FUNÇÃO P/ EXERCÍCIO 3 - REO1 - João Pedro Gomes Pagan 2019160285
#   06/07/2020
#   e-mail: pagan_jp@hotmail.com
#   GITHUB: JoaoPagan

############################################################################

def media(vetor):
    somador = 0
    for num in vetor:
        somador = somador + num
    media = somador / len(vetor)
    return media

import numpy as np

def variancia(vetor):
    var = 0
    med = np.mean(vetor)
    comp = len(vetor)
    for num in vetor:
        var = (var + ((num-med)**2)/comp)
    return var