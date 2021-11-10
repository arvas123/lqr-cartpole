import numpy as np


def lqr(A, B, m, Q, R, M, q, r, T):
    """
    Compute optimal policise by solving
    argmin_{\pi_0,...\pi_{T-1}} \sum_{t=0}^{T-1} x_t^T Q x_t + u_t^T R u_t + x_t^T M u_t + q^T x_t + r^T u_t
    subject to x_{t+1} = A x_t + B u_t + m, u_t = \pi_t(x_t)

    Let the shape of x_t be (N_x,), the shape of u_t be (N_u,)
    Let optimal \pi*_t(x) = K_t x + k_t

    Parameters:
    A (2d numpy array): A numpy array with shape (N_x, N_x)
    B (2d numpy array): A numpy array with shape (N_x, N_u)
    m (1d numpy array): A numpy array with shape (N_x,)
    Q (2d numpy array): A numpy array with shape (N_x, N_x)
    R (2d numpy array): A numpy array with shape (N_u, N_u)
    M (2d numpy array): A numpy array with shape (N_x, N_u)
    q (1d numpy array): A numpy array with shape (N_x,)
    r (1d numpy array): A numpy array with shape (N_u,)
    T (int): The number of total steps in finite horizon settings

    Returns:
        ret (list): A list, [(K_0, k_0), (K_1, k_1), ..., (K_{T-1}, k_{T-1})]
        and the shape of K_t is (N_u, N_x), the shape of k_t is (N_u,)
    """
    R_inv = np.linalg.inv(R)
    K_T1 = -0.5 * (R_inv @ M.T)
    k_T1 = -0.5 * (R_inv @ r)
    res = [(K_T1, k_T1.flatten())]
    S = Q + (K_T1.T @ R @ K_T1) + (M @ K_T1)
    U = (2 * (k_T1.T @ R @ K_T1)) + (k_T1 @ M.T) + (r.T @ K_T1) + q.T
    v = (k_T1.T @ R @ k_T1) + (r.T @ k_T1)
    for h in range(T-1):
        N = Q + (A.T @ S @ A)
        K = R + (B.T @ S @ B)
        O = M + (2 * (A.T @ S @ B))
        l = q.T + (2 * (m.T @ S @ A)) + (U @ A)
        w = r.T + (2 * (m.T @ S @ B)) + (U @ B)
        z = (m.T @ S @ m) + (U @ m + v)
        K_inv = np.linalg.inv(K)
        K_h = -0.5 * (K_inv @ O.T)
        k_h = -0.5 * (K_inv @ w)
        res.append((K_h, k_h.flatten()))
        S = N + (K_h.T @ K @ K_h) + (O @ K_h)
        U = (2 * (k_h.T @ K @ K_h)) + (k_h @ O.T) + (w @ K_h) + l
        v = (k_h.T @ K @ k_h) + (w @ k_h) + z
    res.reverse()
    return res
