import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from matplotlib import cm

# Формирование сетки
X = np.arange(-2, 2, 0.1)
Y = np.arange(-1.5, 3, 0.1)
X, Y = np.meshgrid(X, Y)
# Функция Розенброка
Z = X ** 2 - 2 * X * Y + 6 * (Y ** 2) + X - Y
#
fig = plt.figure()
# Будем выводить 3d-проекцию графика функции
ax = fig.add_subplot(projection='3d')
# Вывод поверхности
surf = ax.plot_surface(X, Y, Z, cmap=cm.Spectral, linewidth=0, antialiased=False)
# Метки осей координат
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Настройка оси X
for label in ax.xaxis.get_ticklabels():
    label.set_color('black')
    label.set_rotation(-45)
    label.set_fontsize(9)
# Настройка оси Y
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(9)
# Настройка оси Z
for label in ax.zaxis.get_ticklabels():
    label.set_fontsize(9)
# Изометрия
ax.view_init(elev=30, azim=45)
# Шкала цветов
fig.colorbar(surf, shrink=0.5, aspect=5)
# Отображение результата (рис. 1)
plt.show()


def func(X):
    return X[0] ** 2 - 2 * X[0] * X[1] + 6 * (X[1] ** 2) + X[0] - X[1]


n = 2
x0 = np.zeros(2, dtype=float)  # Вектор с двумя элементами типа float
# Начальная точка поиска минимума функции
x0[0] = -5.0
x0[1] = 10.0
xtol = 1.0e-3  # Точность поиска экстремума
# Находим минимум функции
res = opt.minimize(func, x0, method='Nelder-Mead', options={'xtol': xtol, 'disp': True})
res1 = opt.minimize(func, x0, tol=xtol, method='Newton-CG')
print(f'Метод Нелдера-Мида:{res}')
print(f'Метода Ньютона-Рафсона: {res1}')
