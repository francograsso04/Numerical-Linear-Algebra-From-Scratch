import numpy as np
import  pandas  as pd
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt

"""
Módulo: alc.py
Trabajo Práctico - Álgebra Lineal Computacional
------------------------------------------------
Implementaciones manuales de funciones matriciales
utilizando las rutinas desarrolladas en el laboratorio.

IMPORTANTE:
    - NO usar funciones de numpy.linalg (inv, pinv, svd, qr, cholesky, etc.)
    - Usar únicamente las funciones desarrolladas por el grupo.
    - Todas las funciones deben respetar la forma matricial solicitada.

Autores:

-Franco V. Grasso 132/23
-Ramiro Arbetman 1307/24
-Nicolas Marchetto 581/23
-Axel Campoverde 258/22

Materia: Álgebra Lineal Computacional (UBA)
"""


####################################
# 0. LABORATORIO
####################################

# Funciones del laboratorio 1

def error(x, y):
    y64 = np.float64(y)   # convertir y a float64
    return abs(x - y64)
def error_relativo(x, y):

    x64 = np.float64(x)
    y64 = np.float64(y)
    return abs(x64 - y64) / abs(x64)

def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x, y), 0, atol=atol)

def matricesIguales(unaMatriz,otraMatriz):
  for i in range(unaMatriz.shape[0]):
    for j in range(unaMatriz.shape[1]):
      if not sonIguales(unaMatriz[i,j], otraMatriz[i,j]):
        return False
  return True


def matmulti(A, B):
    """
    Multiplicación de matriz por matriz o matriz por vector
    """
    if  len(B.shape) == 1:  # vector
        n, m = A.shape
        result = np.zeros(n)
        for i in range(n):
          result[i] = sum(A[i, j] * B[j] for j in range(m))
        return result
    else:  # matriz
        n, m = A.shape
        m2, p = B.shape
        result = np.zeros((n, p))
        for i in range(n):
          for j in range(p):
            result[i, j] = sum(A[i, k] * B[k, j] for k in range(m))
        return result

def vector_dot(v, w):
    """
    Producto interno de dos vectores
    """
    v = np.array(v)
    w = np.array(w)

    n = min(len(v), len(w))
    return sum(v[i] * w[i] for i in range(n))

def outer(v, w):
    """
    Calcula el producto externo entre dos vectores 
    """
    v = np.array(v)
    w = np.array(w)
    n = len(v)
    m = len(w)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i, j] = v[i] * w[j]
    return M

def transpuesta(A):
    """
    Transpone una matriz
    """
    filas, columnas = A.shape
    ATranspuesta = np.zeros((columnas, filas))
    for i in range(filas):
        for j in range(columnas):
            ATranspuesta[j, i] = A[i, j]
    return ATranspuesta

def inversa(A):
    """
    Calcula la inversa de una matriz cuadrada A usando eliminación gaussiana.
    """
    n = A.shape[0]

    AI = np.zeros((n, 2*n))
    for i in range(n):
        for j in range(n):
            AI[i, j] = A[i, j]
        AI[i, n+i] = 1.0  

    for i in range(n):
        
        if AI[i, i] == 0:
            for r in range(i+1, n):
                if AI[r, i] != 0:
                    AI[[i, r]] = AI[[r, i]]  
                    break
            else:
                return None #Matriz singular

     
        pivot = AI[i, i]
        for j in range(2*n):
            AI[i, j] /= pivot

   
        for j in range(n):
            if j != i:
                factor = AI[j, i]
                for k in range(2*n):
                    AI[j, k] -= factor * AI[i, k]


    Inv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Inv[i, j] = AI[i, n+j]

    return Inv

# Funciones del laboratorio 2

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
    return matmulti(matrizEscala,matrizRotacion)

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
    vectorTransformacion = matmulti(matrizAfin, vectorEnR3)

    return vectorTransformacion[:2]          # Tomo solo los 2 primeros numeros


# Funciones del laboratorio 3


def norma(unVectorASacarNorma, unValorDeNorma):
    if unValorDeNorma == 1:
        return np.sum(np.abs(unVectorASacarNorma))
    elif unValorDeNorma == 'inf':
        return np.max(np.abs(unVectorASacarNorma))
    else:
        return np.sum(np.abs(unVectorASacarNorma)**unValorDeNorma)**(1/unValorDeNorma)

def normaliza(X,p):
  Y = []
  for x in X:
    norma_val = norma(x,p)
    if(norma_val == 0):
      Y.append(x)
    else:
      Y.append(x/norma_val)
  return Y

def normaMatMC(A, q, p, Np):
  max = 0.0
  X = [np.random.rand(A.shape[0]) for _ in range(Np)]
  x_normalize = normaliza(X, p)
  best_x = None
  for i in range(Np):
    y = matmulti(A,x_normalize[i])
    norm_q = norma(y, q)
    if norm_q > max:
      max = norm_q
      best_x = x_normalize[i]
  return (max, best_x)

def normaInfinito(unaMatrizANormalizar):
  norma = 0

  for i in range(unaMatrizANormalizar.shape[0]):
    suma = 0
    for j in range(unaMatrizANormalizar.shape[1]):
      suma += abs(unaMatrizANormalizar[i, j])
    if suma > 0:
      norma = suma

  return norma

def norma1(unaMatrizANormalizar):
  norma = 0

  for j in range(unaMatrizANormalizar.shape[1]):
      suma = 0
      for i in range(unaMatrizANormalizar.shape[0]):
        suma += abs(unaMatrizANormalizar[i, j])
      if suma > 0:
        norma = suma

  return norma


def normaExacta(unaMatrizANormalizar, p = [1, 'inf']):
#if isinstance(p, list):
  if p == [1, 'inf']:
    norma_1 = norma1(unaMatrizANormalizar)
    norma_inf = normaInfinito(unaMatrizANormalizar)
    return (norma_1,norma_inf)
#else:
  elif p == 1:
    return norma1(unaMatrizANormalizar)
  elif p == 'inf':
    return normaInfinito(unaMatrizANormalizar)

  return None


def condMC(A, p, Np=1000):
  inv_A = inversa(A)
  return normaMatMC(A, p, p, Np)[0] * normaMatMC(inv_A, p, p, Np)[0]


def condExacta(unaMatriz, unValorDeNorma):
  if unValorDeNorma not in [1, 'inf'] and unValorDeNorma != [1, 'inf']:
    return None

  inv = inversa(unaMatriz)
  norma = normaExacta(unaMatriz, unValorDeNorma)
  normaInv = normaExacta(inv, unValorDeNorma)
  return norma * normaInv

# Funciones del laboratorio 4

def calculaLU(A):
    """
    Calcula la factorización LU de la matriz A con unos en la diagonal de L.
    Retorna (L, U, nops).
    Si no se puede factorizar, retorna (None, None, 0).

    nops cuenta las multiplicaciones y sumas/restas hechas durante la eliminación.
    """
    if A is None:
      return None, None, 0

    n , m = A.shape
    if n != m:
      return None, None, 0
    L = np.eye(n)
    U = A.copy()
    nops = -n  #Esto es porque en el programa cuento como una resta poner el 0 y no lo deberia contar

    for k in range(n-1):
        pivot = U[k, k]
        if np.abs(pivot) < 1e-08: 
            return None, None, 0

        for i in range(k+1, n):
            L[i, k] = U[i, k] / pivot
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]
                nops += 2
                
    if np.abs(U[n-1, n-1]) < 1e-08: 
         return None, None, 0

    return L, U, nops


def res_tri(L, b, inferior=True):
  n = L.shape[0]
  x = np.zeros(n)
  if  inferior:
    for i in range(n):
      x[i] += b[i] / L[i, i]
      for j in range(i):
        x[i] -= (L[i, j] * x[j])/L[i, i]
  else:
    for i in range(n-1,-1,-1):
      x[i] += b[i] / L[i, i]
      for j in range(n-1,i,-1):
        x[i] -= (L[i, j] * x[j])/L[i, i]

  return x

def inversa(A):
  L,U,nops = calculaLU(A)
  if L is None:
    return None
  else:
    n = A.shape[0]
    inversa =np.zeros((n, n))

    for i in range(n):
      b = np.zeros(n)
      b[i] = 1
      d = res_tri(L, b, inferior=True)
      x = res_tri(U, d, inferior=False)
      inversa[:, i] = x

    return inversa


def diagonal_de_matriz(A):
  x,y = A.shape
  if x != y:
    return None

  nueva_matriz = np.zeros((x,y))
  for i in range(x):
    for j in range(y):
      if j == i:
        nueva_matriz[i,j] = A[i,j]
  return nueva_matriz


def calculaLDV(A):
    n = A.shape[0]
    L,U,nops = calculaLU(A)
    if L is None:
      return None,None,None
    D = np.eye(n)
    for i in range(n):
      D[i][i] = U[i][i]
    for i in range(n):
      for j in range(n):
        U[i, j] = U[i,j] / D[i, i]
    return L,D,U

def esSimetrica(A):
    AT = transpuesta(A)
    return matricesIguales(A, AT)


def la_diagonal_es_positiva(A):
    diag = np.diag(A)
    for x in diag:
        if x <= 0:
            return False
    return True

def esSDP(A,atol=1e-8):
    """
    Checkea si la matriz A es simétrica definida positiva (SDP) usando
    la factorización LDV.
    """
    L,D,V = calculaLDV(A)
    iguales = True
    if L is None:
        return False
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        if not sonIguales(A[i,j], transpuesta(A)[i,j],atol):
          iguales = False
    return iguales and la_diagonal_es_positiva(D)

# Funciones del laboratorio 5

def gen_Q(A, tol=1e-12):
    n, m = A.shape
    k = min(n, m)
    Q = np.zeros((n, k))
    for i in range(k):
        v = A[:, i].copy()
        for j in range(i):
            inner_product = vector_dot(Q[:, j], v)
            v = v - inner_product * Q[:, j]
        norm_2 = norma(v,2)
        if norm_2 > tol:
            Q[:, i] = v / norm_2
        else:
            Q[:, i] = np.zeros(n)

    return Q


def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de m x n (con m >= n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A tiene dimensiones m < n, debe retornar None
    """
    m,n = A.shape
    if(m < n):
      return None

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        v = A[:, i].copy()
        for j in range(i):
            inner_product = vector_dot(Q[:, j], v)
            R[j,i] = inner_product
            v = v - inner_product * Q[:,j]
        norm_2 = norma(v,2)
        if norm_2 > tol:
            R[i,i] = norm_2
            Q[:,i] = v/norm_2
        else:
            R[i,i] = 0.0
            Q[:,i] = 0.0

    return Q,R

def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    m, n = A.shape
    if m < n:
        return None

    R = A.copy()
    Q = np.eye(m)

    for i in range(n):
        x = R[i:, i]
        norm_2 = norma(x, 2)

        if norm_2 < tol:
            continue
        
        sign = 1.0 if x[0] >= 0 else -1.0
        v = x.copy()
        v[0] += sign * norm_2
        v = v / norma(v, 2)

        R[i:, i:] -= 2 * outer(v, vector_dot(v.T, R[i:, i:]))

        Q[:, i:] -= 2 * outer(matmulti(Q[:, i:], v), v)

    return Q, R


def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if(metodo == 'RH'):
      return QR_con_HH(A,tol)
    elif(metodo == 'GS'):
      return QR_con_GS(A,tol)
    else:
      return None
    
# Funciones del laboratorio 6

def metpot2k(A, tol=1e-15, K=1000):
    n = A.shape[0]
    vectorResultante = np.random.rand(n)
    vectorResultante = vectorResultante / norma(vectorResultante, 2)
    cuentas = 0

    for i in range(int(K)):
        Av = matmulti(A, vectorResultante)
        vectorSiguiente = Av / norma(Av, 2)
        cuentas += 1

        dif = vectorSiguiente - vectorResultante
        if norma(dif, 2) < tol:
            break
        vectorResultante = vectorSiguiente

    Av = matmulti(A, vectorSiguiente)
    autovalor = vector_dot(vectorSiguiente, Av) / vector_dot(vectorSiguiente, vectorSiguiente)
    return vectorSiguiente, autovalor, cuentas

def diagRH(A, tol=1e-15, K=1000):
    n = A.shape[0]
    v1, lambda1, k = metpot2k(A, tol, K)

    e1 = np.zeros_like(v1)
    e1[0] = 1

    sign = 1.0 if v1[0] >= 0 else -1.0
    norm_v1 = norma(v1, 2)
    w = v1 + sign * norm_v1 * e1
    nw = norma(w, 2)

    if nw < tol:
        H_v1 = np.eye(n)
    else:
        w = w / nw

        H_v1 = np.eye(n)
        for i in range(n):
            for j in range(n):
                H_v1[i, j] -= 2 * w[i] * w[j]

    if n == 1:
        S = H_v1
        D = matmulti(matmulti(H_v1, A), H_v1)
        return S, D
    else:
        B = matmulti(matmulti(H_v1, A), H_v1)
        A_prima = B[1:, 1:]
        S_prima, D_prima = diagRH(A_prima, tol, K)

        D = np.zeros((n, n))
        D[0, 0] = lambda1
        D[1:, 1:] = D_prima

        S = np.eye(n)
        S[1:, 1:] = S_prima
        S = matmulti(H_v1, S)
        return S, D
    

# Funciones del laboratorio 7

def transiciones_al_azar_continuas(n):
    """
    n: cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, con entradas al azar en [0,1].
    """
    T = np.random.random((n, n))

    for j in range(n):
        T[:, j] = normaliza([T[:, j]], 1)[0]

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
            col = normaliza([col], 1)[0]
        T[:, j] = col

    return T

def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """

    ATA = matmulti(transpuesta(A),A)

    resultado = diagRH(ATA, tol=tol)
    if resultado is None:
        return None

    S, D = resultado

    autovalores = np.diag(D)
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

# Funciones del laboratorio 8

def svd_reducida(A,k="max",tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """
    m, n = A.shape
    if k == "max":
        k = min(A.shape)
    if m > n:
      ATA = matmulti(transpuesta(A),A)
      return obtenerSVD(A,ATA, k, tol=tol, Mmayor=True)

    else:
      AAT = matmulti(A,transpuesta(A))
      return obtenerSVD(A,AAT, k, tol=tol, Mmayor=False)



def obtenerSVD(A, M, k, tol=1e-15, Mmayor=True):

    S, D = diagRH(M, tol=tol)
    S = gen_Q(S, tol=tol)  

   
    autovalores = np.diag(D).copy()

    autovaloresAux = []
    for x in autovalores:
      if x < 0:
        autovaloresAux.append(0)
      else:
        autovaloresAux.append(x)
    autovalores = np.array(autovaloresAux)
    valores_singulares = np.sqrt(autovalores)


    idxNoCero = []
    for i in range(len(valores_singulares)):
      if valores_singulares[i] > tol:
        idxNoCero.append(i)

    if len(idxNoCero) == 0:
     
        return np.zeros((A.shape[0], 0)), np.zeros(0), np.zeros((A.shape[1], 0))

    valores_singulares = valores_singulares[idxNoCero]
    S = S[:, idxNoCero]

 
    k = min(k, len(valores_singulares))
    valores_singulares = valores_singulares[:k]
    S = S[:, :k]

    if Mmayor:
        B = matmulti(A,S)
        U = gen_Q(B, tol=tol)
        V = S
        U = U[:, :k]
        return U, valores_singulares, V
    else:
        B = matmulti(transpuesta(A),S)
        V = gen_Q(B, tol=tol)
        U = S
        V = V[:, :k]
        return U, valores_singulares, V



####################################
# 1. LECTURA DE DATOS
####################################


def cargarDataset(basePath):
    """
    Carga el dataset completo de gatos y perros a partir de un path
    
    La función espera que dentro de basePath existan dos subcarpetas principales:
    - train: con las carpetas cats y dogs que contienen los embeddings de entrenamiento.
    - val: con las carpetas cats y dogs que contienen los embeddings de validación.

    Internamente, llama a cargarCarpeta para procesar cada par de carpetas (gatos y perros)
    y devuelve las matrices con los datos y etiquetas correspondientes.
    
    Retorna:
        Xt, Yt -> datos y etiquetas de entrenamiento
        Xv, Yv -> datos y etiquetas de validación
    """
    trainCats = os.path.join(basePath, "train", "cats")
    trainDogs = os.path.join(basePath, "train", "dogs")
    valCats   = os.path.join(basePath, "val", "cats")
    valDogs   = os.path.join(basePath, "val", "dogs")

    Xt, Yt = cargarCarpeta(trainCats, trainDogs)
    Xv, Yv = cargarCarpeta(valCats, valDogs)

    return Xt, Yt, Xv, Yv

def cargarCarpeta(pathGatos, pathPerros):
    """
    Carga los embeddings de gatos y perros desde dos carpetas dadas y construye las matrices X e Y.
    
    - Recorre todos los archivos .npy de ambas carpetas.
    - Cada archivo se carga como un vector columna .
    - Las etiquetas (Y) se asignan como:
        * [1, 0] para gatos
        * [0, 1] para perros
    
    Retorna:
        X -> matriz de embeddings
        Y -> matriz de etiquetas 
    """
    XList = []
    YList = []


    for archivo in os.listdir(pathGatos):
        if archivo.endswith(".npy"):
            embedding = np.load(os.path.join(pathGatos, archivo))
            XList.append(embedding)
            aux = np.array([[1, 0]] * embedding.shape[1])
            YList.append(aux)

    for archivo in os.listdir(pathPerros):
        if archivo.endswith(".npy"):
            embedding = np.load(os.path.join(pathPerros, archivo))
            XList.append(embedding)
            aux = np.array([[0, 1]] * embedding.shape[1])
            YList.append(aux)

    X =  np.array(XList)
    Y = np.array(YList)

    return np.hstack((X[0][:,], X[1][:,])), np.vstack((Y[0][:,], Y[1][:,]))


####################################
# 2. ECUACIONES NORMALES
####################################


def pinvEcuacionesNormales(X,L,Y):
    """
    Calcula los pesos W usando pseudo-inversa por ecuaciones normales y Cholesky.

    X: matriz de entrada 
    Y: matriz de tipos de imagenes 
    L: L de Cholesky 

    Devuelve:
    W: pesos (m x n)
    """
    n, p = X.shape
    rangoX = rango(X)
    LT = transpuesta(L)

    if rangoX == p and n > p:
        XT = transpuesta(X)

        U = np.zeros_like(XT)
        for col in range(n):
            b = XT[:, col]
            z = res_tri(L, b, inferior=True)
            u = res_tri(LT, z, inferior=False)
            U[:, col] = u

        W = matmulti(Y, U)
    elif rangoX == n and n < p:
        V = np.zeros_like(XT)
        for col in range(n):
            b = X[:, col]
            z = res_tri(L, b, inferior=True)
            v = res_tri(LT, z, inferior=False)
            V[:, col] = v

        W = matmulti(Y, transpuesta(V))
    elif rangoX == n and n == p:
        XInv = inversa(X)
        W = matmulti(Y, XInv)

    return W

####################################
# 3. DESCOMPOSICIÓN EN VALORES SINGULARES (SVD)
####################################

def pinvSVD(U, S, V, Y):
    """
    Calcula los pesos W utilizando la pseudo-inversa obtenida por SVD.

    Parámetros:
        U, S, V : matrices de la descomposición SVD propia
        Y       : matriz de targets de entrenamiento

    Retorna:
        W = Y @ V @ S⁻¹ @ U.T
    """

    v_transpuesta = V.T #Hago transpuesta para pasarla como parametro a calcularPseudoInversa luego

    V, S_inversa, U_transpuesta = calcularPseudoInversa(U, S, v_transpuesta)

    W = calculo_W_SVD(V, S_inversa, U_transpuesta, Y)

    return W



####################################
# 4. DESCOMPOSICIÓN QR
####################################

def pinvHouseHolder(Q, R, Y):
    """
    Calcula los pesos W usando la descomposición QR (Householder).

    Parámetros:
        Q, R : matrices obtenidas de la factorización QR (Householder)
        Y    : matriz de targets

    Retorna:
        W : pesos que minimizan ||Y - W X||²
    """
    # Q R = X.T con  X de dim {n x p}, n < p por lo tanto X.T es de dim {p x n} y Q es de dim {p x p} y R de dim {p x n}
    # Luego X+ = (X.T X)^{-1} X.T = (R.T R)^{-1} R.T Q.T = M_inv R.T Q.T, donde M_inv = (R.T R)^{-1}

    R_transpuesta = transpuesta(R)

    M = matmulti(R_transpuesta, R)  # M = R.T R

    L,LT= descCholesky(M)
    M_inv = np.zeros_like(M)

    for i in range(M.shape[0]):

        b = np.zeros(M.shape[0])
        b[i] = 1
        y = res_tri(L, b, inferior=True)
        x = res_tri(LT, y, inferior=False) 
        M_inv[:, i] = x

    X_plus = matmulti((matmulti(M_inv, transpuesta(R))), transpuesta(Q))  # X+ = M_inv R.T Q.T
    W = matmulti(Y, transpuesta(X_plus))

    return W, X_plus


def pinvGramSchmidt(Q, R, Y):
    """
    Calcula los pesos W usando la descomposición QR (Gram-Schmidt clásico).

    Parámetros:
        Q, R : matrices obtenidas con Gram-Schmidt
        Y    : matriz de targets

    Retorna:
        W : pesos óptimos
    """
    # Como Gram-Schmidt se puede hacer únicamente con una matriz en ℝ^{n×p} con n >= p, hacemos la desc QR en X.T
    # Si QR = X.T -> X+ = Q (R.T)-1 -> X+ R.T = Q
    # Si X.T es de dims {pxn} -> Q es de {pxn} y R es de {nxn} (al igual que (R.T)-1)
    # Luego X+ tiene dims {pxn}
    # Para resolver X+ R.T = Q, transponemos ambos lados. Así, R (X+).T = Q.T (con R triangular superior)
    # Resolvemos este sistema triangular para cada fila de Q (cada una es una columna de Q.T) para encontrar todas las columnas de (X+).T

    p, n = Q.shape
    pX_T = np.zeros((n, p))

    for i in range(n):
        pX_T[:, i] = res_tri(R, Q[i, :], inferior=False)
        
    pX = transpuesta(pX_T)
    
    # Calculamos W = YX+
    # Si Y es de dims {2xp}, y X+ de {pxn} -> dim (W) = {2xn}
    W = matmulti(Y, pX)
    
    return W


####################################
# 5. PSEUDO-INVERSA DE MOORE-PENROSE
####################################

def esPseudoInversa(X, pX, tol=1e-8):
    """
    Verifica si pX es la pseudo-inversa de X según Moore-Penrose.

    X: matriz original
    pX: matriz candidata a pseudo-inversa

    Devuelve True si cumple las 4 condiciones, False si no.
    """

    # Condición 1: X * pX * X ≈ X
    if not sonIguales(matmulti(matmulti(X , pX) , X), X, tol):
        return False

    # Condición 2: pX * X * pX ≈ pX
    if not sonIguales(matmulti(matmulti(pX ,X) , pX), pX, tol):
        return False

    # Condición 3: (X * pX)^T ≈ X * pX
    if not sonIguales(transpuesta(matmulti(X,pX)), matmulti(X,pX), tol):
        return False

    # Condición 4: (pX * X)^T ≈ pX * X
    if not sonIguales(transpuesta(matmulti(pX,X)), matmulti(pX,X), tol):
        return False

    return True

####################################
# 6. EVALUACIÓN Y BENCHMARKING
####################################

def evaluarModelo(W, Xv, Yv):
    """
    Evalúa el modelo sobre el conjunto de validación.
    Devuelve la matriz de confusión y el accuracy.
    """
    Y_pred = W @ Xv
    preds = np.argmax(Y_pred, axis=0)
    reales = np.argmax(Yv, axis=0)

    # Matriz de confusión 2x2
    conf = np.zeros((2, 2), dtype=int)
    for i in range(len(preds)):
        conf[reales[i], preds[i]] += 1

    accuracy = np.trace(conf) / np.sum(conf)
    return conf, accuracy




####################################
# 0. Funciones auxiliares
####################################
def obtenerL(X):
    """
    Calcula la L a partir de X dependiendo de en que caso
    se encuentra.
    """
    n, p = X.shape
    rangoX = rango(X)

    if rangoX == p and n > p:
        XTX = matmulti(transpuesta(X), X)
        L,_ = descCholesky(XTX)
        return L

    elif rangoX == n and n < p:
        XXT = matmulti(X, transpuesta(X))
        L,_  = descCholesky(XXT)
        return L

def descCholesky(A):
    """
    Calcula la descomposición de Cholesky A = L L^T
    usando la LDV si A es simétrica definida positiva.
    """
    Lprima, D, _ = calculaLDV(A)

    if not esSDP(A):
        return None, None

    if Lprima is None or D is None:
        return None, None

    raizD = np.zeros_like(D)
    for i in range(D.shape[0]):
        raizD[i, i] = np.sqrt(D[i, i])

    L = matmulti(Lprima, raizD)

    LT = transpuesta(L)

    return L, LT

def rango(A):
    """
    Retorna el rango de una matriz a partir de la cantidad de valores
    singulares distintos de 0
    """
    _,D,_ = svd_reducida(A)
    rango = sum(1 for d in D if d > 0)
    return rango


def calcularPseudoInversa(U,S,V_transpuesta):
    """
    Entrada: Descomposicion de valores singulares de una matriz
    Retorna V,S+,U transpuesta que representa la pseudoInversa de A
    """

    V = V_transpuesta.T
    U_transpuesta = U.T

    filas, columnas = S.shape
    S_inversa = np.zeros((columnas, filas))

    #Uso tolerancia para no hacer divisiones de numeros muy chiquitos
    #Consultar si se tiene que hacer asi a priori
    tol = 1e-12
    for i in range(min(filas, columnas)):
        if S[i, i] > tol:
            S_inversa[i, i] = 1 / S[i, i]


    # A^+ = V * S^+ * U^T
    return V, S_inversa, U_transpuesta




def calculo_W_SVD(V,S_inversa,U_transpuesta, Y):
   # El propósito de esta función es comprender el motivo por el cual resulta conveniente
   # emplear la forma "reducida" de la descomposición SVD en el cálculo de la pseudoinversa.
   # Hasta este punto disponemos de las matrices V, S_inversa, U_transpuesta y Y sin ningún tipo de reducción.
   # El objetivo es ir simplificando gradualmente la representación, conservando solo
   # la información relevante para el cálculo de W.

   # 1) Cálculo del rango de la matriz S_inversa:
   # El rango de S_inversa coincide con la cantidad de valores singulares distintos de cero de X.
   # Este valor es fundamental, ya que determina cuántas direcciones (o componentes) aportan información útil.
   # A partir del rango, podemos realizar las particiones necesarias en U y V para
   # limitar las operaciones únicamente a las dimensiones efectivamente informativas.
   #
   # Por ejemplo, si U es una matriz de 1000x1000 pero X tiene solo 100 valores singulares no nulos,
   # no resulta eficiente operar con las 1000 columnas: basta con conservar las primeras 100,
   # que contienen toda la información relevante. El resultado obtenido es el mismo,
   # pero con un costo computacional mucho menor.
   #
   # De esta manera, la pseudoinversa puede expresarse como:
   #
   #     X⁺ = V Σ⁺ Uᵀ  ≈  V₁ Σ₁⁻¹ U₁ᵀ
   #
   # donde las matrices V₁, Σ₁ y U₁ corresponden a la forma reducida,
   # es decir, solo las componentes asociadas a valores singulares distintos de cero.


    rango = 0
    filas_u, columnas_u = np.shape(S_inversa)
    for i in range(filas_u):
        for j in range(columnas_u):
            if (S_inversa[i,j] > 0):
                rango = rango + 1

    #Ahora la variable rango es la cantidad de elementos > 0 de la diagonal sigma.

    # 2) Partición de las matrices U y V según el rango:
    # Tomamos únicamente las primeras 'rango' columnas o filas necesarias.

    V1 = V[:, :rango] # p×r
    U1_T = U_transpuesta[:rango, :]  #r×n
    S1_inversa = S_inversa[:rango, :rango] #Te queda rxr


    # 3) Justificación de dimensiones:
    # Sabemos que X ∈ ℝ^{n×p}  ⇒  X⁺ ∈ ℝ^{p×n}.
    #
    # La forma reducida de la pseudoinversa es:
    #     X⁺ = V₁ Σ₁⁻¹ U₁ᵀ
    #
    # Analizando las dimensiones:
    #     V₁ Σ₁⁻¹ U₁ᵀ  ⇒  (p×r)(r×r)(r×n)
    #  Multiplicando de derecha a izquierda:
    #     (r×r)(r×n) = (r×n)
    #     (p×r)(r×n) = (p×n)
    #
    #  Por lo tanto, X⁺ ∈ ℝ^{p×n}, cumpliendo con las dimensiones esperadas.
    #
    # Este chequeo asegura la compatibilidad matricial y evita errores
    # de producto al construir la pseudoinversa.

    #Luego,

    #Si el rango efectivo de X es menor que su rango máximo, la pseudoinversa puede expresarse en forma reducida como X⁺ = V₁ Σ₁⁻¹ U₁ᵀ,
    # utilizando únicamente las componentes asociadas a valores singulares no nulos.”

    #Las U2 y V2 no me importan porque son las particiones que se multiplican por los 0's de sigma. No es relevante. Sumaria costo computacional a la funcion

    # 4) Cálculo de la pseudoinversa reducida:
    #     X⁺ = V₁ Σ₁⁻¹ U₁ᵀ
    producto = matmulti(V1, S1_inversa)          # V₁ Σ₁⁻¹
    pseudo_inv = matmulti(producto, U1_T) # V₁ Σ₁⁻¹ U₁ᵀ


    # 5) Cálculo final de W:
    #     W = Y X⁺ = Y (V₁ Σ₁⁻¹ U₁ᵀ)
    W = matmulti(Y, pseudo_inv)                   # Y * (V₁ Σ₁⁻¹ U₁ᵀ)

    return W











