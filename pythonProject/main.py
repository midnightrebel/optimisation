from math import cos, sin
from typing import Tuple, Callable

import numpy as np


def func(x):
    return cos(x) / x ** 2


def derivative(x):
    return -(sin(x) / (x ** 2)) - ((2 * cos(x)) / (x ** 3))


def second_derivative(x):
    return (sin(x) + (6 * cos(x) / x) - (18 * sin(x) / (x ** 2)) - (24 * cos(x) / (x ** 3))) / (x ** 2)


def sven(function, x0: float, h: float = 0.0001) -> Tuple[float, float]:
    f0 = function(x0)
    a = b = x0
    if f0 > function(x0 + h):
        a = x0
    elif function(x0 - h) >= f0:
        a = x0 - h
        b = x0 + h
        return a, b
    else:
        b = x0
        h = -h

    def xk(next_element: int):
        return x0 + (2 ** (next_element - 1)) * h

    def assign_if(is_a, value):
        nonlocal a, b
        if is_a:
            a = value
        else:
            b = value

    k = 2
    while True:
        xk0, xk1 = xk(k), xk(k - 1)
        if func(xk0) >= func(xk1):
            assign_if(h < 0, xk0)
            break
        else:
            assign_if(h > 0, xk1)
        k += 1
    return a, b


def newton(precision, x0):
    iterations = 1
    x1 = x0 - derivative(x0) / second_derivative(x0)
    xn1 = x1
    while True:
        iterations += 1
        xn = xn1
        xn1 = xn - derivative(xn) / second_derivative(xn)
        if abs(derivative(xn1)) <= precision:
            return iterations, xn1


class FibonacciImpl:
    arr = [0, 1, 1]

    def calculate(self, num: int):
        if num < len(self.arr):
            return self.arr[num]
        else:
            for i in range(len(self.arr) - 1, num):
                new_fib = self.arr[i - 1] + self.arr[i]
                self.arr.append(new_fib)
            return self.arr[num]


fib_impl = FibonacciImpl()


def fib(num):
    global fib_impl
    return fib_impl.calculate(num)


def metod_fib(function, a, b, eps=0.001, sigma=0.001 / 10):
    n = int((b - a) / (2 * eps))
    print('tyt N - ', n)
    x1 = a + fib(n - 2) / fib(n) * (b - a)
    x2 = a + fib(n - 1) / fib(n) * (b - a)
    for k in range(2, n - 2):
        if function(x1) <= function(x2):
            b = x2
            x2 = x1
            x1 = a + fib(n - k - 3) / fib(n - k - 1) * (b - a)
        else:
            a = x1
            x1 = x2
            x2 = a + fib(n - k - 2) / fib(n - k - 1) * (b - a)
    x2 = x1 + sigma
    if function(x1) <= function(x2):
        b = x2
    else:
        a = x1
    return (a + b) / 2


def passive_search(a: float, b: float, n: int, function: Callable[[float], float]) -> float:
    step = (b - a) / n
    x_points = np.arange(a + step, b, step)
    value_points = np.array([function(x) for x in x_points])

    min_index = value_points.argmin()
    return x_points[min_index]


def main():
    select_mode = int(input(f"Выберите метод минимизации:\n"
                            f"1-Метод равномерного поиска\n"
                            f"2-Метод Ньютона\n"
                            f"3-Метод Фибоначчи\n"))
    stancionary_points = []
    x_points = [-9.21096438740149,
                -5.95939190757933,
                -2.45871417599962,
                2.45871417599962,
                5.95939190757933,
                9.21096438740149]

    for x in x_points:
        stancionary_point = func(x)
        stancionary_points.append(stancionary_point)
    global_min_extremum = 0.0
    left_border = derivative(-10.0)
    right_border = derivative(10.0)
    for stancionary_point in stancionary_points:
        if left_border < stancionary_point > right_border:
            if second_derivative(stancionary_point) < 0:
                global_min_extremum = stancionary_point
    print(f'Глобальный минимум функции: {global_min_extremum}')
    interval = sven(func, global_min_extremum, 0.001)
    print(f"Интервал неопределённости:{interval}")
    if select_mode == 1:
        count_of_intervals = int(input('Количество интервалов:'))
        print('-' * 25 + "Метод равномерного поиска" + '-' * 25)
        print(f'Минимум:{passive_search(interval[0], interval[1], count_of_intervals, func)}')
        print(f'Значение функции:{func(passive_search(interval[0], interval[1], count_of_intervals, func))}')
    elif select_mode == 2:
        print('-' * 25 + "Метод Ньютона" + '-' * 25)
        precision = float(input('Точность вычислений:'))
        x0 = float(input('Начальное значение:'))
        print(f'Значение функции:{func(newton(precision, x0)[1])}')
        print(f'Значение X:{newton(precision, x0)[1]}')
        print(f'Количество итераций:{newton(precision, x0)[0]}')
    elif select_mode == 3:
        print('-' * 25 + "Метод Фибоначчи" + '-' * 25)
        print(f'Минимум:{metod_fib(func, interval[0], interval[1])}')
        print(f'Значение функции:{func(metod_fib(func, interval[0], interval[1]))}')


if __name__ == "__main__":
    main()
