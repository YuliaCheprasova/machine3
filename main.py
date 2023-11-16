import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
from random import randint
import scipy.optimize as optimize
import statistics

def Y(x1, x2, outliers_percent=10):
    y=np.zeros(len(x1))
    for i in range(len(x1)):
        y[i]=math.sin(x1[i]) + x2[i]/10
        percent = randint(0, 5)
        noise = np.random.normal(0, 0.5, 1)[0]
        noise = np.clip(noise, (-y[i]*percent)/100, (y[i] * percent)/100)
        y[i] += noise
        if np.random.rand() < outliers_percent / 100:
            intensity = np.random.uniform(-5.0, 7.0)  # Интенсивность выброса
            y[i] += intensity
    return y


def linear_func(x1, x2, a, b, c):
    return a + b * x1 + c * x2


def cubic_func(x1, x2, a, b1, b2, c1, c2):
    return a + b1 * x1 + b2 * x2 + c1 * x1**2 + c2 * x2**2


def weighted_least_squares(params, x1, x2, y, w, kind_of_func):
    if kind_of_func:
        residuals = (y - linear_func(x1, x2, *params)) * w
    else:
        residuals = (y - cubic_func(x1, x2, *params)) * w
    return np.sum(residuals ** 2)


"""def Y_true(x1, x2):
    y=np.zeros(len(x1))
    for i in range(len(x1)):
        y[i]=m.sin(x1[i]) + x2[i]/10
    return y"""


def H(i, r, x1, x2):
    hmax1, hmax2 = 0, 0
    for j in range(i - r, i + r):
        if j >= 0 and j < len(x1):
            h1 = abs(x1[i]-x1[j])
            if h1 > hmax1:
                hmax1=h1
            h2 = abs(x2[i] - x2[j])
            if h2 > hmax2:
                hmax2 = h2
    return hmax1, hmax2


def count_W(z):
    if abs(z)<1:
        W = (1-z**2)**2
    else:
        W = 0
    return W


def count_w(x1, x2, i, h1, h2):
    w = np.zeros(len(x1))
    for j in range(len(x1)):
        w[j] = count_W((x1[j]-x1[i])/h1)*count_W((x2[j]-x2[i])/h2)
    return w


def count_w_robast(E, mead):
    w = np.zeros(len(E))
    for j in range(len(E)):
        w[j] = count_W(E[j] / (6*mead))
    return w


if __name__ == '__main__':
    s = 100
    kind_of_func = False  # True if linear
    if kind_of_func:
        num_param = 3
    else:
        num_param = 5
    m = 2
    f = 0.2
    r = int(math.ceil(f * s))
    len_test = int(s * 0.2)
    x_start1 = np.zeros(s + len_test)
    x_start2 = np.zeros(s + len_test)
    for i in range(s + len_test):
        x_start1[i] = randint(1, 100)
        x_start2[i] = randint(1, 100)
    y_start = Y(x_start1, x_start2)
    x1 = x_start1[:s]
    x2 = x_start2[:s]
    x_test1 = x_start1[s:]
    x_test2 = x_start2[s:]
    y = y_start[:s]
    y_test = y_start[s:]
    ypredict = np.zeros(s)
    ypredict_test = np.zeros(len_test)
    E = np.zeros(s)
    ME = np.zeros(s)
    for i in range(s):
        h1, h2= H(i, r, x1, x2)
        w = count_w(x1, x2, i, h1, h2)
        params = optimize.minimize(weighted_least_squares, x0=np.ones(num_param), args=(x1, x2, y, w, kind_of_func)).x
        if kind_of_func:
            ypredict[i] = params[0] + params[1]*x1[i]+params[2]*x2[i]
        else:
            ypredict[i] = params[0] + params[1] * x1[i] + params[2] * x2[i] + params[3] * x1[i]**2 + params[4] * x2[i]**2
        E[i] = (y[i]-ypredict[i])**2
    for iter in range(m):
        med = statistics.median(E)
        for i in range(s):
            ME[i] = abs(E[i]-med)
        mead = statistics.median(ME)
        for i in range(s):
            w = count_w_robast(E, mead)
            params = optimize.minimize(weighted_least_squares, x0=np.ones(num_param), args=(x1, x2, y, w, kind_of_func)).x
            if kind_of_func:
                ypredict[i] = params[0] + params[1] * x1[i] + params[2] * x2[i]
            else:
                ypredict[i] = params[0] + params[1] * x1[i] + params[2] * x2[i] + params[3] * x1[i] ** 2 + params[4] * x2[i] ** 2
            E[i] = (y[i] - ypredict[i]) ** 2

    points = np.arange(s)
    plt.plot(points, y, label='Истинные значения')
    plt.plot(points, ypredict, label='Сглаженные значения')
    plt.legend()
    plt.show()
    MSE = sum(E) / s
    print(f'MSE = {MSE}')




# как тут делать тест, если парметры рассчитываются для каждой отдельной точки
# Примените LOWESS
#lowess = sm.nonparametric.lowess(y, x, frac=0.3)  # frac - параметр, определяющий ширину окна

# Результат LOWESS - это массив точек (x, y)
"""smoothed_x, smoothed_y = lowess.T
plt.scatter(x,y)
plt.plot(smoothed_x, smoothed_y)
plt.show()"""