# Третья лабораторная по вычислительной математике в СПбПУ
11 Вариант:
Привести дифференциальное уравнение: `ty''-(t+1)y'-2(t-1)y=0`  
к системе двух дифференциальных уравнений первого порядка.

Начальные условия: `y(t=1)=e^2`; `y'(t=1)=2e^2`  
Точное решение: `y(t)=e^(2t)`  
Решить на интервале `1<=t<=2`  

1) Используя программу RKF45 с шагом печати h_print = 0.1 и выбранной вами погрешностью EPS в диапазоне 0.001 – 0.00001,
а также составить собственную программу и решить с шагом интегрирования h_int = 0.1
2) Используя метод Рунге-Кутты 3-й степени точности  

Сравнить результаты, полученные заданными приближенными способами, с точным решением.   
Исследовать влияние величины шага интегрирования h_int на величины локальной и глобальной 
погрешностей решения заданного уравнения для чего решить уравнение, используя 2 – 3 значения 
шага интегрирования, существенно меньшие исходной величины 0.1 
(например, h_int = 0.05; h_int= 0.025; h_int= 0.0125)
___
Для выполнения работы RKF45 был взят из библиотеки `scipy`, а конкретно метод
`integrate.ode`, который был настроен специальным образом:
```Python
ode(f).set_integrator('dopri5', atol=0.0001).set_initial_value(X0, T[0])
```
Метод Рунге-Кутты же необходимо написать отдельно, сделать это не сложно:
```Python
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
```
Остается лишь посчитать значения при разных шагах и вывести на экран. Для этого
используются библиотеки:
- `matplotlib` - для графиков (их можно увидеть в проекте)
- `prettytable` - для вывода таблицы в консоль