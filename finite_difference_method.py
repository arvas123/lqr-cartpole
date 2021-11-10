import numpy as np


def gradient(f, x, delta=1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    n, = x.shape
    x = x.astype('float64')
    gradient = np.zeros(n).astype('float64')
    for i in range(n):
        x[i] += delta
        gplus = f(x)
        x[i] -= 2 * delta
        gminus = f(x)
        x[i] += delta
        gradient[i] = (gplus - gminus) / (2 * delta)
    return gradient


def jacobian(f, x, delta=1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    n, = x.shape
    m, = f(x).shape
    x = x.astype('float64')
    gradient = np.zeros((m, n)).astype('float64')
    for i in range(n):
        x[i] += delta
        gplus = f(x)
        x[i] -= 2 * delta
        gminus = f(x)
        x[i] += delta
        gradient[:, i] = (gplus - gminus) / (2 * delta)
    return gradient


def hessian(f, x, delta=1e-5):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    def grad(x):
        return gradient(f, x, delta)
    x = jacobian(grad, x, delta)
    return x


def test():
    """
    Run tests on the above functions against some known case.
    """
    Q = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    q = np.array([10, 11, 12], dtype=np.float32)
    x_s = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    def f1(x): return Q @ (x - x_s) + np.ones(3)
    def f2(x): return (x - x_s) @ Q @ (x - x_s) + q @ (x - x_s) + 1
    assert np.allclose(jacobian(f1, np.array(x_s)), Q)
    assert np.allclose(gradient(f2, np.array(x_s)), q)
    assert np.allclose(hessian(f2, np.array(x_s)), Q+Q.T)


if __name__ == "__main__":
    import gym
    x_s = np.array([0, 0, 0, 0], dtype=np.double)
    u_s = np.array([0], dtype=np.double)
    def f(x, u):
        """
        State transition function of the environment.
        Return the next state by executing action `u` at the state `x`
        Parameter:
            x (1D numpy array) with shape (4,) 
            u (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert x.shape == (4,)
        assert u.shape == (1,)
        env = gym.make("env:CartPoleControlEnv-v0")
        env.reset(state=x)
        next_observation, cost, done, info = env.step(u)
        return next_observation
    def f_vec(x):
            return f(x[0:4], x[4:])
    x_vec = np.insert(x_s, 4, u_s)
    print(x_vec)
    jac = jacobian(f_vec, x_vec)
    A = jac[:4, :4]
    B = jac[:4, [4]]
    print(np.matrix(A))
    print(np.matrix(B))
