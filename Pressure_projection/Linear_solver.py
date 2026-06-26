
def ConjugateGradient(A, b, x0, tol=1e-6, max_iter=500):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    r2_old = r @ r
    if r2_old < tol:
        return x
    for i in range(max_iter):
        Ap = A @ p
        alpha = r2_old / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        r2_new = r @ r
        if r2_new < tol:
            break
        beta = r2_new / r2_old
        p = r + beta * p
        r2_old = r2_new
    return x
