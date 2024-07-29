import numpy as np
import matplotlib.pyplot as plt

def euler(f, y0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    # Euler
    for i in range(n):
        k1 = h * f(t[i], y[i])
        y[i + 1] = y[i] + k1
        print(f"y({round(t[i],2)}) = {round(y[i],4)}")

    return t, y

# Ejemplo de uso
def f(t, y):
    return -y+t+2

y0 = 2
t0 = 0
tf = 1
h = 0.1

t, y = euler(f, y0, t0, tf, h)
print(f"Valor del punto final con Euler: y({round(tf,2)}) = {round(y[-1],4)}")

plt.plot(t, y, label='Euler')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()