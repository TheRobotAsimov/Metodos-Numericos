import numpy as np
import matplotlib.pyplot as plt

def euler_mejorado(f, y0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h, y[i] + k1)
        y[i + 1] = y[i] + (k1 + k2) / 2
        print(f"y({t[i]}) = {y[i]}")

    return t, y

# Ejemplo de uso
def f(t, y):
    return y - t

y0 = 2
t0 = 0
tf = 1
h = 0.1

t, y = euler_mejorado(f, y0, t0, tf, h)

plt.plot(t, y, label='Euler Mejorado')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()

# Mostrar el valor del punto final
print(f"Valor del punto final con Euler Mejorado: y({tf}) = {y[-1]}")