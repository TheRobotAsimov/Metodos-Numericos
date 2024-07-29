import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4(f, y0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        print(f"y({t[i]}) = {y[i]}")

    return t, y

# Ejemplo de uso
def f(t, y):
    return -2 * t * y

y0 = 1
t0 = 0
tf = 5
h = 0.1

t, y = runge_kutta_4(f, y0, t0, tf, h)

plt.plot(t, y, label='Runge-Kutta 4')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()

# Mostrar el valor del punto final
print(f"Valor del punto final con Euler Mejorado: y({tf}) = {y[-1]}")