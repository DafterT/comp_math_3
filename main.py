"""
Привести дифференциальное уравнение: ty``-(t+1)y`-2(t-1)y=0
к системе двух дифференциальных уравнений первого порядка.

Начальные условия: y(t=1)=e^2; y`(t=1)=2e^2
Точное решение: y(t)=e^(2t)
Решить на интервале 1<=t<=2

1) Используя программу RKF45 с шагом печати h_print = 0.1 и выбранной вами погрешностью EPS в диапазоне 0.001 – 0.00001,
а также составить собственную программу и решить с шагом интегрирования h_int = 0.1
2) Используя метод Рунге-Кутты 3-й степени точности
"""
import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot


def rkf45(f, T, X0):
    """
    Решает `x' = f(t, x)` для каждого `t` в `T`
    С начальным значением `X0`, используя аналог rkf45
    """
    runge = ode(f).set_integrator('dopri5', atol=0.0001).set_initial_value(X0, T[0])
    X = [X0, *[runge.integrate(T[i]) for i in range(1, len(T))]]
    return np.array([i[0] for i in X])


def Runge_Kutta(f, T, X0):
    """
    Решает `x' = f(t, x)` для каждого `t` в `T`
    С начальным значением `X0`, используя формулы Рунге-Кутты 3 степени
    """
    X = np.zeros((len(T), len(X0)))
    X[0] = X0
    h = T[1] - T[0]
    for i in range(0, len(T) - 1):
        k_1 = h * f(T[i], X[i])
        k_2 = h * f(T[i] + h / 2, X[i] + k_1 / 2)
        k_3 = h * f(T[i] + 3 * h / 4, X[i] + 3 * k_2 / 4)
        X[i + 1] = (X[i] + (2 * k_1 + 3 * k_2 + 4 * k_3) / 9)
    return X[:, 0]


def f(t, X):
    """
    Правая часть `x' = f(t, x)`.
    """
    dX = np.zeros(X.shape)
    dX[0] = (t + 1) / t * X[0] + 2 * (t - 1) / t * X[1]
    dX[1] = X[0]
    return dX


def g(T):
    """
    Точное решение
    """
    return np.e ** (2 * T)


def main():
    pass


if __name__ == '__main__':
    main()
