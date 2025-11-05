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

