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



def obtenerSVD(A,M,k,tol=1e-15,Mmayor = True):
      resultado = diagRH(M, tol=tol)
      if resultado is None:
          return None
      S, D = resultado
      S = gen_Q(S,tol=tol)
      autovalores = np.diag(D).copy()
      for i in range(len(autovalores)):
          if autovalores[i] < tol:
              autovalores[i] = 0
      valores_singulares = np.sqrt(autovalores)
      valores_singularesInv = np.zeros(len(valores_singulares))
      for i in range(len(valores_singulares)):
          if valores_singulares[i] == 0:
              valores_singularesInv[i] = 0
              S[:,i] = np.zeros(S.shape[ 1])
          else:
              valores_singularesInv[i] = 1/valores_singulares[i]
      valores_singulares = np.diag(valores_singulares)


      if Mmayor:
        B = matmulti(A, S)
        U = gen_Q(B, tol=tol)
        valores_singulares = valores_singulares[:k]
        V = S[:, :k]
        U = U[:, :k]
        return U,np.diag(valores_singulares),V
      else:
        B = matmulti(transpuesta(A),S)
        V = gen_Q(B, tol=tol)
        valores_singulares = valores_singulares[:k]
        U = S[:, :k]
        V = V[:, :k]
        return U,np.diag(valores_singulares),V


