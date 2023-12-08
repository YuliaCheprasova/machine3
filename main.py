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
        #if np.random.rand() < outliers_percent / 100:
            #intensity = np.random.uniform(-10.0, 15.0)
            #y[i] += intensity
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


def count_W2(z):
    if abs(z)<1:
        W = (1-z**2)**2
    else:
        W = 0
    return W


def count_W3(z):
    if abs(z)<1:
        W = (1-abs(z)**3)**3
    else:
        W = 0
    return W


def count_w(x1, x2, i, h1, h2):
    w = np.zeros(len(x1))
    for j in range(len(x1)):
        w[j] = count_W3((x1[j]-x1[i])/h1)*count_W3((x2[j]-x2[i])/h2)
    return w


def count_w_robast(E, mead):
    w = np.zeros(len(E))
    for j in range(len(E)):
        w[j] = count_W2(E[j] / (6*mead))
    return w


def Ypredict(start, end, i, kind_of_func, ypredict, params, x1, x2, E, y):
    ypred = np.zeros(len(x1))
    for j in range(start, end, 1):
        if kind_of_func:
            ypred[j] = params[0] + params[1] * x1[j] + params[2] * x2[j]
        else:
            ypred[j] = params[0] + params[1] * x1[j] + params[2] * x2[j] + params[3] * x1[j] ** 2 + params[4] * x2[j] ** 2
        E[j] = (y[j] - ypred[j]) ** 2
    ypredict[i] = ypred[i]


if __name__ == '__main__':
    s = 100
    kind_of_func = True  # True if linear
    if kind_of_func:
        num_param = 3
    else:
        num_param = 5
    m = 2
    x1 = np.linspace(0, 100, s)
    x2 = np.linspace(0, 200, s)
    y = Y(x1, x2)
    ypredict = np.zeros(s)
    E = np.zeros(s)
    ME = np.zeros(s)
    points = np.arange(s)
    """for k in range(1, 10, 1):
        b=k/10
        r = int(math.ceil(b * s))
        step = r
        print(r)
        for i in range(0, s):
            h1, h2 = H(i, r, x1, x2)
            w = count_w(x1, x2, i, h1, h2)
            if i-step<0:
                begin=0
            else:
                begin=i-step
            if i+step>s:
                end=s
            else:
                end=i+step
            x1_param=x1[begin:end]
            x2_param = x2[begin:end]
            y_param = y[begin:end]
            w_param = w[begin:end]
            params = optimize.minimize(weighted_least_squares, x0=np.ones(num_param), args=(x1_param, x2_param, y_param, w_param, kind_of_func)).x
            Ypredict(begin, end, i, kind_of_func, ypredict, params, x1, x2, E, y)
            for iter in range(m):
                E_param = E[begin:end]
                ME_param = ME[begin:end]
                med = statistics.median(E_param)
                for j in range(len(E_param)):
                    ME_param[j] = abs(E_param[j]-med)
                mead = statistics.median(ME_param)
                w = count_w_robast(E, mead)
                w_param = w[begin:end]
                params = optimize.minimize(weighted_least_squares, x0=np.ones(num_param), args=(x1_param, x2_param, y_param, w_param, kind_of_func)).x
                Ypredict(begin, end, i, kind_of_func, ypredict, params, x1, x2, E, y)
        plt.plot(points, ypredict, label=f'b={b}')"""

    b = 0.03
    r = int(math.ceil(b * s))
    step = r
    print(r)
    for i in range(0, s):
        h1, h2 = H(i, r, x1, x2)
        w = count_w(x1, x2, i, h1, h2)
        if i - step < 0:
            begin = 0
        else:
            begin = i - step
        if i + step > s:
            end = s
        else:
            end = i + step
        x1_param = x1[begin:end]
        x2_param = x2[begin:end]
        y_param = y[begin:end]
        w_param = w[begin:end]
        params = optimize.minimize(weighted_least_squares, x0=np.ones(num_param),
                                   args=(x1_param, x2_param, y_param, w_param, kind_of_func)).x
        Ypredict(begin, end, i, kind_of_func, ypredict, params, x1, x2, E, y)
        """for iter in range(m):
            E_param = E[begin:end]
            ME_param = ME[begin:end]
            med = statistics.median(E_param)
            for j in range(len(E_param)):
                ME_param[j] = abs(E_param[j] - med)
            mead = statistics.median(ME_param)
            w = count_w_robast(E, mead)
            w_param = w[begin:end]
            params = optimize.minimize(weighted_least_squares, x0=np.ones(num_param),
                                       args=(x1_param, x2_param, y_param, w_param, kind_of_func)).x
            Ypredict(begin, end, i, kind_of_func, ypredict, params, x1, x2, E, y)"""
    plt.plot(points, ypredict, label=f'b={b}')


    plt.plot(points, y, label='Истинные значения')
    plt.legend()
    plt.show()



