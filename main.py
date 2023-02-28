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
    Solves `x' = f(t, x)` for each `t` in `T`
    with initial values of `X0`.
    """
    runge = ode(f).set_integrator('dopri5').set_initial_value(X0, T[0])
    X = [X0, *[runge.integrate(T[i]) for i in range(1, len(T))]]
    return X


def main():
    pass


if __name__ == '__main__':
    main()
