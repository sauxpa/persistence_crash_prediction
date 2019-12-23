import numpy as np

def random_henon(n, b, sigma, T=1.4, x0=0.0, y0=0.0, a0=0.0):
    x = np.empty(n)
    y = np.empty(n)
    a = np.empty(n)

    dt = T/n

    x[0] = x0
    y[0] = y0
    a[0] = a0
    for i in range(1, n):
        noise = sigma*np.random.randn()*np.sqrt(dt)
        x[i] = 1 - a[i-1]*x[i-1]**2 + b*y[i-1] + noise
        y[i] = x[i] + noise
        a[i] = a[i-1] + dt
    return x, y, a
