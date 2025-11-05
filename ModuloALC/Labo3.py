from imports import np, plt
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
  max = 0.0;
  X = [np.random.rand(A.shape[0]) for _ in range(Np)]
  x_normalize = normaliza(X, p);
  best_x = None;
  for i in range(Np):
    y = A @ x_normalize[i];
    norm_q = norma(y, q);
    if norm_q > max:
      max = norm_q;
      best_x = x_normalize[i];
  return (max, best_x);

def normaInfinito(unaMatrizANormalizar):
  normaInf = 0
  maximo = 0

  for i in range(unaMatrizANormalizar.shape[0]):
    suma = 0
    for j in range(unaMatrizANormalizar.shape[1]):
      suma += abs(unaMatrizANormalizar[i, j])
    if suma > maximo:
      normaInf = suma

  return suma

def norma1(unaMatrizANormalizar):
  norma1 = 0
  maximo = 0

  for j in range(unaMatrizANormalizar.shape[1]):
      suma = 0
      for i in range(unaMatrizANormalizar.shape[0]):
        suma += abs(unaMatrizANormalizar[i, j])
      if suma > maximo:
        norma1 = suma

  return suma


def normaExacta(unaMatrizANormalizar, p = [1, 'inf']):
  if p not in [1, 'inf'] and p != [1, 'inf']:
    return None

  if isinstance(p, list):
    norma_1 = norma1(unaMatrizANormalizar)
    norma_Inf = normaInfinito(unaMatrizANormalizar)
    return (norma_1,norma_Inf)

  elif p == 1:
    return norma1(unaMatrizANormalizar)

  elif p == 'inf':
    return normaInfinito(unaMatrizANormalizar)


def condMC(A, p, Np=1000):
  inv_A = np.linalg.inv(A)
  return normaMatMC(A, p, p, Np)[0] * normaMatMC(inv_A, p, p, Np)[0]


def condExacta(unaMatriz, unValorDeNorma):
  if unValorDeNorma not in [1, 'inf'] and unValorDeNorma != [1, 'inf']:
    return None

  inversa = np.linalg.inv(unaMatriz)
  norma = normaExacta(unaMatriz, unValorDeNorma)
  normaInv = normaExacta(inversa, unValorDeNorma)
  return norma * normaInv
