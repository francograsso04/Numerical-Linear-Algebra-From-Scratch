from imports import np, plt,lb3,lb1,lb6
def transiciones_al_azar_continuas(n):
    """
    n: cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, con entradas al azar en [0,1].
    """
    T = np.random.random((n, n))

    for j in range(n):
        T[:, j] = lb3.normaliza([T[:, j]], 1)[0]

    return T

def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas.
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres.
    Todos los elementos de la columna $j$ son iguales
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """

    T = np.random.random((n, n))

    for i in range(n):
        for j in range(n):
            if T[i, j] <= thres:
                T[i, j] = 1.0
            else:
                T[i, j] = 0.0

    for j in range(n):
        col = T[:, j]
        if np.sum(col) == 0:
            col[:] = 1.0 / n # Si la columna me queda de 0 lleno todo de 1/n
        else:
            col = lb3.normaliza([col], 1)[0]
        T[:, j] = col

    return T

def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """

    ATA = lb1.matmulti(lb1.transpuesta(A),A)

    resultado = lb6.diagRH(ATA, tol=tol)
    if resultado is None:
        return None

    S, D = resultado

    autovalores = np.diag(D) #Tomo los elementos de la diagonal
    indices_nucleo = np.where(np.abs(autovalores) <= tol)[0]


    if len(indices_nucleo) == 0:
        return np.array([])

    return S[:, indices_nucleo]

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default.
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    matriz_rala = {}

    if  len(listado) == 0:
        return matriz_rala, (m_filas, n_columnas)

    lista_i, lista_j, lista_valores = listado
    n = len(lista_valores)

    for k in range(n):
        i = lista_i[k]
        j = lista_j[k]
        val = lista_valores[k]
        if abs(val) >= tol:
            matriz_rala[(i, j)] = val

    return matriz_rala, (m_filas, n_columnas)



def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v.
    Retorna un vector w resultado de multiplicar A con v
    """
    ADic, dims = A
    n, m = dims

    w = np.zeros(m)

    for (i, j), val in ADic.items():
        w[i] += val * v[j]

    return w



def es_markov(T,tol=1e-6):
    """
    T una matriz cuadrada.
    tol la tolerancia para asumir que una suma es igual a 1.
    Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if T[i,j]<0:
                return False
    for j in range(n):
        suma_columna = sum(T[:,j])
        if np.abs(suma_columna - 1) > tol:
            return False
    return True

def es_markov_uniforme(T,thres=1e-6):
    """
    T una matriz cuadrada.
    thres la tolerancia para asumir que una entrada es igual a cero.
    Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    if not es_markov(T,thres):
        return False
    # cada columna debe tener entradas iguales entre si o iguales a cero
    m = T.shape[1]
    for j in range(m):
        non_zero = T[:,j][T[:,j] > thres]
        # all close
        close = all(np.abs(non_zero - non_zero[0]) < thres)
        if not close:
            return False
    return True


def esNucleo(A,S,tol=1e-5):
    """
    A una matriz m x n
    S una matriz n x k
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
    """
    for col in S.T:
        res = A @ col
        print(res)
        if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
            return False
    return True
