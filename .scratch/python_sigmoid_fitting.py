
import numpy as np
import pylab
from scipy.optimize import curve_fit

def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))

def fsigmoid2(x, a, b, c, d, f, t):
    return( a/(b + c*np.exp(-d*(x-f))) + t )

def rfsigmoid2(F, a, b, c, d, f, t):
    return(-np.log( (a-b*(F-t)) / (c* (F-t)) ) * (1/d) + f)

curv_function = rfsigmoid2
# xdata = np.array([400, 600, 800, 1000, 1200, 1400, 1600])
# ydata = np.array([0, 0, 0.13, 0.35, 0.75, 0.89, 0.91])
# xdata = np.arange(len(total_scores))
# ydata = total_scores

ydata = np.linspace(0, 10, 500)
xdata = np.linspace(0, 10, 500)

popt_ = np.array([1.31, 0.21, 2.96, 1.69, 4.23, 0.04])
# popt, pcov = curve_fit(sigmoid, xdata, ydata)
popt, pcov = curve_fit(curv_function, xdata, ydata, method='dogbox', bounds=(popt_ - 0.002, popt_ + 0.002))


import sympy
!pip install sympy
from sympy import Symbol, S
from sympy.calculus.util import continuous_domain
x = Symbol("x")
f = sin(x)/x
continuous_domain(f, x, S.Reals)

curv_function(0, *popt_)
popt, pcov = curve_fit(fsigmoid2, xdata, ydata, method='dogbox', bounds=([popt_ - 0.2], [popt_ + 0.2]))
popt, pcov = curve_fit(fsigmoid, xdata, ydata, method='dogbox', bounds=([0., 600.],[0.01, 1200.]))

popt_ = popt
# print(popt)
# popt = np.random.rand(len(popt))
# fsigmoid2(5, 1, 3, 4, 2, 5, 6)
# x = np.linspace(0, 10, 100)
# pylab.plot(x, rfsigmoid2(x, *popt_))
x = np.linspace(0, 10, 500)
y = curv_function(x, *popt)
pylab.plot(xdata, ydata, 'o', label='data')
pylab.plot(x, y, label='fit')
# pylab.ylim(0, 1.05)
pylab.legend(loc='best')
