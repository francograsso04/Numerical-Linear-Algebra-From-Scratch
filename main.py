from imports import *
import os

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
            XList.append(embedding.reshape(-1, 1)) #El reshape -1 es para hacerlo vector columna(Preguntar bien esto)
            YList.append(np.array([[1], [0]]))      

    for archivo in os.listdir(pathPerros):
        if archivo.endswith(".npy"):
            embedding = np.load(os.path.join(pathPerros, archivo))
            XList.append(embedding.reshape(-1, 1))  
            YList.append(np.array([[0], [1]]))      


    cantidadImagenes = XList[0].shape[0]#creo que no hay una mejor manera,salvo que XList no sea una lista
    cantidadDeTiposDeImagenes = len(XList)           


    X = np.zeros((cantidadImagenes, cantidadDeTiposDeImagenes))
    Y = np.zeros((2, cantidadDeTiposDeImagenes))     

    for i in range(cantidadDeTiposDeImagenes):
        X[:, i] = XList[i].flatten() #Sino se aplana pincha
        Y[:, i] = YList[i].flatten()

    return X, Y

def descCholesky(A):
    """
    Calcula la descomposición de Cholesky A = L L^T
    usando la LDV si A es simétrica definida positiva.
    """
    Lprima, D, _ = lb4.calculaLDV(A)

    if not lb4.esSDP(A):
        return None, None

    if Lprima is None or D is None:
        return None, None

    raizD = np.zeros_like(D)
    for i in range(D.shape[0]):
        raizD[i, i] = np.sqrt(D[i, i])

    L = lb1.matmulti(Lprima, raizD)

    LT = lb1.transpuesta(L)

    return L, LT

def pinvEcuacionesNormales(X, Y):
    return none



if __name__ == "__main__":
    basePath = r"C:\Users\Casa\OneDrive\Escritorio\tpALC\ALC\cats_and_dogs"
    Xt, Yt, Xv, Yv = cargarDataset(basePath)

    print("Xt:", Xt.shape)
    print("Yt:", Yt.shape)
    print("Xv:", Xv.shape)
    print("Yv:", Yv.shape)

