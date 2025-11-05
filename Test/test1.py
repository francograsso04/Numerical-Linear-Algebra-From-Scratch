from util.imports import np,lb1
assert(not lb1.sonIguales(1, 1.1))
assert(lb1.sonIguales(1, 1 + np.finfo('float64').eps))
assert(not lb1.sonIguales(1, 1 + np.finfo('float32').eps))
assert(not lb1.sonIguales(np.float16(1), np.float16(1) + np.finfo('float32').eps))
assert(lb1.sonIguales(np.float16(1), np.float16(1) + np.finfo('float16').eps, atol=1e-3))

assert(np.allclose(lb1.error_relativo(1, 1.1), 0.1))
assert(np.allclose(lb1.error_relativo(2, 1), 0.5))
assert(np.allclose(lb1.error_relativo(-1, -1), 0))
assert(np.allclose(lb1.error_relativo(1, -1), 2))

assert(lb1.matricesIguales(np.diag([1, 1]), np.eye(2)))
assert(lb1.matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]])) @ np.array([[1,2],[3,4]]), np.eye(2)))
assert(not lb1.matricesIguales(np.array([[1,2],[3,4]]).T, np.array([[1,2],[3,4]])))
