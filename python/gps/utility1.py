import numpy as np


def CGM(A, b, x0, iters):
    r = b - np.matmul(A, b)
    p = r
    k = 0
    x = x0
    while k < iters:
        rTr = np.matmul(r.T, r)[0][0]
        alpha = rTr/np.matmul(p.T, np.matmul(A, p))[0][0]
        #print(x.shape, alpha, p.shape)
        x += alpha*p
        r -= alpha*np.matmul(A, p)
        beta = np.matmul(r.T, r)[0][0]/rTr
        p = r + beta*p
        k += 1
    return x
