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
import matplotlib as mpl
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def rkf45(f, T, X0):
    """
    Решает `x' = f(t, x)` для каждого `t` в `T`
    С начальным значением `X0`, используя аналог rkf45
    """
    runge = ode(f).set_integrator('dopri5', atol=0.0001).set_initial_value(X0, T[0])
    X = [X0, *[runge.integrate(T[i]) for i in range(1, len(T))]]
    return np.array([i[0] for i in X]), np.array([i[1] for i in X])


def RK3(f, T, X0):
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
    dX[0] = X[1]
    dX[1] = (t + 1) / t * X[1] + 2 * (t - 1) / t * X[0]
    return dX


def g(T):
    """
    Точное решение
    """
    return np.e ** (2 * T)


def print_one_graph(t, y, title, id, count_graphs):
    """
    Функция для отрисовки одного графика
    """
    plt.subplot(1, count_graphs, id)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid()
    plt.title(title)
    plt.plot(t, y, '-o')


def print_graph(t_find, y_real, Y_RKF45, Y_Runge_Kutta, h):
    """
    Функция для отрисовки всех графиков
    """
    mpl.use('TkAgg')
    plt.figure(figsize=(15, 4))
    print_one_graph(t_find, y_real, 'Исходный график', 1, 3)
    print_one_graph(t_find, Y_RKF45, 'График RKF45', 2, 3)
    print_one_graph(t_find, Y_Runge_Kutta, 'График Рунге-Кутты', 3, 3)
    plt.savefig(f"Graphs_{h}.jpg")
    plt.show()


def print_error_graph(t_find, Y_RKF45_error, Y_Runge_Kutta_error, h):
    """
    Функция для отрисовки погрешности
    """
    mpl.use('TkAgg')
    plt.figure(figsize=(15, 4))
    # Собственно сам график
    print_one_graph(t_find, Y_RKF45_error, 'Погрешность RKF45', 1, 2)
    print_one_graph(t_find, Y_Runge_Kutta_error, 'Погрешность Рунге-Кутты', 2, 2)
    plt.savefig(f"Error_{h}.jpg")
    plt.show()


def print_table(t_find, y_real, Y_RKF45, Y_RKF45_error, Y_Runge_Kutta, Y_Runge_Kutta_error, h):
    """
    Функция для отрисовки таблицы
    """
    print(f'h = {h}')
    koef = {0.1: 1, 0.05: 2, 0.025: 4, 0.0125: 8}.get(h)
    pt = PrettyTable()
    pt.add_column('t', [f'{i:.1f}' for i in t_find[::koef]])
    pt.add_column('real y', [f'{i:.15f}' for i in y_real[::koef]])
    pt.add_column('RKF45 y', [f'{i:.15f}' for i in Y_RKF45[::koef]])
    pt.add_column('Delta RKF45 y', [f'{i:.15f}' for i in Y_RKF45_error[::koef]])
    pt.add_column('Runge Kutta y', [f'{i:.15f}' for i in Y_Runge_Kutta[::koef]])
    pt.add_column('Delta Runge Kutta y', [f'{i:.15f}' for i in Y_Runge_Kutta_error[::koef]])
    print(pt)
    print('First step of RKF45:', Y_RKF45_error[1])
    print('First step of Runge Kutta:', Y_Runge_Kutta_error[1])
    print('Global of RKF45:', Y_RKF45_error.sum())
    print('Global of Runge Kutta:', Y_Runge_Kutta_error.sum())
    print('h^4 is about:', h ** 4)
    print('h^4 / Runge Kutta first step:', h ** 4 / Y_Runge_Kutta_error[1])


def print_additiontal_research(T, Y_derivative_real, Y_derivative_RKF45):
    pt = PrettyTable()
    pt.add_column("T", [f'{i:.4f}' for i in T])
    pt.add_column("Y' real", Y_derivative_real)
    pt.add_column("Y' RKF45", Y_derivative_RKF45)
    pt.add_column("Y' delta", Y_derivative_real - Y_derivative_RKF45)
    print(pt)
    print('=' * 110)


def evaluate(h):
    """
    Получение решения при разных шагах
    """
    # Начальные значения
    X0 = np.array([np.e ** 2, 2 * np.e ** 2])
    # Значения в узлах
    T = np.arange(1, 2 + h, h)
    Y = g(T)
    # Расчет RKF45
    Y_RKF45, Y_derivative_RKF45 = rkf45(f, T, X0)
    # Расчет Рунге-Кутты
    Y_Runge_Kutta = RK3(f, T, X0)
    # Погрешности
    Y_RKF45_error = Y - Y_RKF45
    Y_Runge_Kutta_error = Y - Y_Runge_Kutta
    # Рисуем графики
    print_graph(T, Y, Y_RKF45, Y_Runge_Kutta, h)
    print_error_graph(T, Y_RKF45_error, Y_Runge_Kutta_error, h)
    # Выводим данные в консоль
    print_table(T, Y, Y_RKF45, Y_RKF45_error, Y_Runge_Kutta, Y_Runge_Kutta_error, h)
    # Вывод для дополнительных исследований
    print_additiontal_research(T, 2 * (np.e ** (2 * T)), Y_derivative_RKF45)
    return Y_RKF45_error, Y_Runge_Kutta_error


def print_table_error(Y_RKF45_error, Y_Runge_Kutta_error, h_list):
    pt = PrettyTable()
    pt.add_column('h', [f'{i:.4f}' for i in h_list])
    pt.add_column("Runge Kutta Error local", [f'{i[1]:.15f}' for i in Y_Runge_Kutta_error])
    pt.add_column('h**4 / Runge Kutta Error local',
                  [f'{i ** 4 / j[1]:.15f}' for i, j in zip(h_list, Y_Runge_Kutta_error)])
    print(pt)
    pt.clear()
    pt.add_column('h', [f'{i:.4f}' for i in h_list])
    pt.add_column("Runge Kutta Error global", [f'{i.sum():.15f}' for i in Y_Runge_Kutta_error])
    pt.add_column('h**2 / Runge Kutta Error global',
                  [f'{i ** 2 / j.sum():.15f}' for i, j in zip(h_list, Y_Runge_Kutta_error)])
    print(pt)
    pt.clear()
    pt.add_column('h', [f'{i:.4f}' for i in h_list])
    pt.add_column("RKF45 Error local", [f'{i[1]:.15f}' for i in Y_RKF45_error])
    pt.add_column("RKF45 Error global", [f'{i.sum():.15f}' for i in Y_RKF45_error])
    pt.add_column("Runge Kutta Error local", [f'{i[1]:.15f}' for i in Y_Runge_Kutta_error])
    pt.add_column("Runge Kutta Error global", [f'{i.sum():.15f}' for i in Y_Runge_Kutta_error])
    print(pt)


def main():
    h_list = [0.1 / (2 ** i) for i in range(4)]
    Y_RKF45_error = []
    Y_Runge_Kutta_error = []
    for h in h_list:
        error = evaluate(h)
        Y_RKF45_error.append(error[0])
        Y_Runge_Kutta_error.append(error[1])
    print_table_error(Y_RKF45_error, Y_Runge_Kutta_error, h_list)


if __name__ == '__main__':
    main()
