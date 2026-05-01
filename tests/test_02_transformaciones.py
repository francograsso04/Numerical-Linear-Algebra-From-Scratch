from lab_imports import np,lb2
assert np.allclose(lb2.rota(0), np.eye(2))
assert np.allclose(lb2.rota(np.pi / 2), np.array([[0, -1], [1, 0]]))
assert np.allclose(lb2.rota(np.pi), np.array([[-1, 0], [0, -1]]))

assert np.allclose(lb2.escala([2, 3]), np.array([[2, 0], [0, 3]]))
assert np.allclose(lb2.escala([1, 1, 1]), np.eye(3))
assert np.allclose(
    lb2.escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]])
)

assert np.allclose(
    lb2.rota_y_escala(0, [2, 3]), np.array([[2, 0], [0, 3]])
)
assert np.allclose(
    lb2.rota_y_escala(np.pi / 2, [1, 1]), np.array([[0, -1], [1, 0]])
)
assert np.allclose(
    lb2.rota_y_escala(np.pi, [2, 2]), np.array([[-2, 0], [0, -2]])
)


assert np.allclose(
    lb2.afin(0, [1, 1], [1, 2]),
    np.array([[1, 0, 1],
              [0, 1, 2],
              [0, 0, 1]])
)

assert np.allclose(
    lb2.afin(np.pi / 2, [1, 1], [0, 0]),
    np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 1]])
)

assert np.allclose(
    lb2.afin(0, [2, 3], [1, 1]),
    np.array([[2, 0, 1],
              [0, 3, 1],
              [0, 0, 1]])
)


assert np.allclose(
    lb2.trans_afin(np.array([1, 0]), np.pi / 2, [1, 1], [0, 0]),
    np.array([0, 1])
)
assert np.allclose(
    lb2.trans_afin(np.array([1, 1]), 0, [2, 3], [0, 0]),
    np.array([2, 3])
)

assert np.allclose(
    lb2.trans_afin(np.array([1, 0]), np.pi / 2, [3, 2], [4, 5]),
    np.array([4, 7])
)