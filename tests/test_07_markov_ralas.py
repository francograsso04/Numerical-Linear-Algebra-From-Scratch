from lab_imports import np,lb7
for i in range(1,100):
    T = lb7.transiciones_al_azar_continuas(i)
    assert lb7.es_markov(T), f"transiciones_al_azar_continuas fallo para n={i}"

    T = lb7.transiciones_al_azar_uniformes(i,0.3)
    assert lb7.es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    # Si no atajan casos borde, pueden fallar estos tests. Recuerden que suma de columnas DEBE ser 1, no valen columnas nulas.
    T = lb7.transiciones_al_azar_uniformes(i,0.01)
    assert lb7.es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    T = lb7.transiciones_al_azar_uniformes(i,0.01)
    assert lb7.es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"

#Debe estar mal la de sacar autovalores y autovectores


# nucleo
A = np.eye(3)
S = lb7.nucleo(A)
assert S.shape[0]==0, "nucleo fallo para matriz identidad"
A[1,1] = 0
S = lb7.nucleo(A)
msg = "nucleo fallo para matriz con un cero en diagonal"
assert lb7.esNucleo(A,S), msg
assert S.shape==(3,1), msg
assert abs(S[2,0])<1e-2, msg
assert abs(S[0,0])<1e-2, msg

v = np.random.random(5)
v = v / np.linalg.norm(v)
H = np.eye(5) - np.outer(v, v)  # proyección ortogonal
S = lb7.nucleo(H)
msg = "nucleo fallo para matriz de proyeccion ortogonal"
assert S.shape==(5,1), msg
v_gen = S[:,0]
v_gen = v_gen / np.linalg.norm(v_gen)
assert np.allclose(v, v_gen) or np.allclose(v, -v_gen), msg


# crea rala
listado = [[0,17],[3,4],[0.5,0.25]]
A_rala_dict, dims = lb7.crea_rala(listado,32,89)
assert dims == (32,89), "crea_rala fallo en dimensiones"
assert A_rala_dict[(0,3)] == 0.5, "crea_rala fallo"
assert A_rala_dict[(17,4)] == 0.25, "crea_rala fallo"
assert len(A_rala_dict) == 2, "crea_rala fallo en cantidad de elementos"

listado = [[32,16,5],[3,4,7],[7,0.5,0.25]]
A_rala_dict, dims = lb7.crea_rala(listado,50,50)
assert dims == (50,50), "crea_rala fallo en dimensiones con tol"
assert A_rala_dict.get((32,3)) == 7
assert A_rala_dict[(16,4)] == 0.5
assert A_rala_dict[(5,7)] == 0.25

listado = [[1,2,3],[4,5,6],[1e-20,0.5,0.25]]
A_rala_dict, dims = lb7.crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con tol"
assert (1,4) not in A_rala_dict
assert A_rala_dict[(2,5)] == 0.5
assert A_rala_dict[(3,6)] == 0.25
assert len(A_rala_dict) == 2

# caso borde: lista vacia. Esto es una matriz de 0s
listado = []
A_rala_dict, dims = lb7.crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con lista vacia"
assert len(A_rala_dict) == 0, "crea_rala fallo en cantidad de elementos con lista vacia"

# multiplica rala vector
listado = [[0,1,2],[0,1,2],[1,2,3]]
A_rala = lb7.crea_rala(listado,3,3)
v = np.random.random(3)
v = v / np.linalg.norm(v)
res = lb7.multiplica_rala_vector(A_rala,v)
A = np.array([[1,0,0],[0,2,0],[0,0,3]])
res_esperado = A @ v
assert np.allclose(res,res_esperado), "multiplica_rala_vector fallo"

A = np.random.random((5,5))
A = A * (A > 0.5)
listado = [[],[],[]]
for i in range(5):
    for j in range(5):
        listado[0].append(i)
        listado[1].append(j)
        listado[2].append(A[i,j])

A_rala = lb7.crea_rala(listado,5,5)
v = np.random.random(5)
assert np.allclose(lb7.multiplica_rala_vector(A_rala,v), A @ v)