def rota(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def escala(factoresEscala):
    dimension = len(factoresEscala)
    matrizEscala = np.zeros((dimension, dimension))
    for i in range(dimension):
        matrizEscala[i, i] = factoresEscala[i]
    return matrizEscala

def rota_y_escala(angulo, factoresEscala):
    matrizRotacion = rota(angulo)
    matrizEscala = escala(factoresEscala)
    return matrizEscala @ matrizRotacion

def afin(angulo, factoresEscala, vector):
    matrizRotaYEscala = rota_y_escala(angulo,factoresEscala)
    matrizAfin = np.eye(3)                   # np.eye hace la identidad(el 3 es por 3x3)
    matrizAfin[:2, :2] = matrizRotaYEscala   # [:2, :2] significa que matrizRotaYEscala la pongo en las fila 0 y 1 y en las columnas 0 y 1
    matrizAfin[:2, 2] = vector               # Osea que tengo[1,0,0 y tomo la le asigno a la submatriz [1,0  la matrizRotaYEscala
                                             #                0,1,0                                     0,1]
                                             #               0,0,1] En [:2, :2] es asigno a las posiciones de las fila 0 y 1 de la columna 2
    return matrizAfin

def trans_afin(vector, angulo, factoresEscala, vectorDeTraslacion):
    matrizAfin = afin(angulo, factoresEscala, vectorDeTraslacion)
    vectorEnR3 = np.array([vector[0], vector[1], 1])
    vectorTransformacion = matrizAfin @ vectorEnR3

    return vectorTransformacion[:2]          # Tomo solo los 2 primeros numeros
