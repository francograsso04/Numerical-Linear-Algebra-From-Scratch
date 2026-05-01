from lab_imports import np,lb3

# Tests L03-Normas
#from moduloALC import norma, normaliza, normaExacta, normaMatMC, condMC, condExacta
#import numpy as np

# Tests norma
print("TESTS NORMA")
assert(np.allclose(lb3.norma(np.array([0,0,0,0]),1), 0))
assert(np.allclose(lb3.norma(np.array([4,3,-100,-41,0]),"inf"), 100))
assert(np.allclose(lb3.norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(lb3.norma(np.array([1]*10),2),np.sqrt(10)))
assert(lb3.norma(np.random.rand(10),2)<=np.sqrt(10))
assert(lb3.norma(np.random.rand(10),2)>=0)

print("------ÉXITO!!!!\n")

# Tests normaliza
print("TEST NORMALIZA")

# caso borde
# print("---TEST NORMALIZA NULO")
# test_borde = normaliza([np.array([0,0,0,0])],2)
# assert(len(test_borde) == 1)
# assert(np.allclose(test_borde[0],np.array([0,0,0,0])))
# print("------ÉXITO!!!!")

# normaliza norma 2
print("---TEST NORMALIZA 2")
test_n2 = lb3.normaliza([np.array([1]*k) for k in range(1,11)],2)
assert(len(test_n2) != 0)
for x in test_n2:
    assert(np.allclose(lb3.norma(x,2),1))
print("------ÉXITO!!!!")

# normaliza norma 1
print("---TEST NORMALIZA 1")
test_n1 = lb3.normaliza([np.array([1]*k) for k in range(2,11)],1)
assert(len(test_n1) != 0)
for x in test_n1:
    assert(np.allclose(lb3.norma(x,1),1))
print("------ÉXITO!!!!")

# normaliza norma inf
print("---TEST NORMALIZA INF")
test_nInf = lb3.normaliza([np.random.rand(k) for k in range(1,11)],'inf')
assert(len(test_nInf) != 0)
for x in test_nInf:
    assert(np.allclose(lb3.norma(x,'inf'),1))

print("------ÉXITO!!!!\n")

# Tests normaExacta
print("TEST normaExacta")

assert(np.allclose(lb3.normaExacta(np.array([[1,-1],[-1,-1]]))[0],2))
assert(np.allclose(lb3.normaExacta(np.array([[1,-1],[-1,-1]]))[1],2))
assert(np.allclose(lb3.normaExacta(np.array([[1,-2],[-3,-4]]))[0] ,6))
assert(np.allclose(lb3.normaExacta(np.array([[1,-2],[-3,-4]]))[1],7))
assert(lb3.normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(lb3.normaExacta(np.random.random((10,10)))[0] <=10)
assert(lb3.normaExacta(np.random.random((4,4)))[1] <=4)

print("------ÉXITO!!!!\n")

# Test normaMatMC
print("TEST normaMatMC")

nMC = lb3.normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = lb3.normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = lb3.normaMatMC(A=A,q='inf',p='inf',Np=1000000)
assert(np.allclose(nMC[0],lb3.normaExacta(A)[1],rtol=1e-1))

print("------ÉXITO!!!!\n")

# Test condMC
print("TEST condMC")

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = lb3.normaMatMC(A,2,2,10000)
normaA_ = lb3.normaMatMC(A_,2,2,10000)
condA = lb3.condMC(A,2)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-2))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = lb3.normaMatMC(A,2,2,10000)
normaA_ = lb3.normaMatMC(A_,2,2,10000)
condA = lb3.condMC(A,2)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-2))

print("------ÉXITO!!!!\n")

# Test condExacta
print("TEST condExacta")

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = lb3.normaExacta(A)[0]
normaA_ = lb3.normaExacta(A_)[0]
condA = lb3.condExacta(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = lb3.normaExacta(A)[1]
normaA_ = lb3.normaExacta(A_)[1]
condA = lb3.condExacta(A,'inf')
assert(np.allclose(normaA*normaA_,condA))

print("------ÉXITO!!!!\n")

print("---FINALIZADO LABO 3!---")