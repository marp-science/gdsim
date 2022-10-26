
from scipy.interpolate import interp1d
from numpy.linalg import norm, pinv
import numpy as np
from math import exp


class SystemRecognition:

    def __init__(self, x, y):

        self.stepResponse = (x, y)

    def detectSystemType(self, verbose=True):
        import numpy as np
        from math import sqrt, pi, log

        x, y = self.stepResponse

        # examining type of system
        y_test = y - y[-1]
        positive = (y_test > 0).sum()
        negative = (y_test < 0).sum()

        if negative > 0 and positive > 0:
            if verbose==True: print('Type 2 :: oscillator')
            return 2
        else:
            if verbose==True: print('Type 1 :: inertial system')
            return 1

        return 0


    def estimateTau(self):
        import numpy as np

        xm, ym = self.stepResponse

        d2c = D2CApprox(xm, ym, 'cubic')
        step = xm.max() / 1000
        x = np.arange(0, xm.max()+step, step)
        y = d2c.h(x)

        # y = a*x + b
        f = lambda x, a, b: a*x + b

        (xA, yA) = (0, 0)

        for (xB, yB) in zip(x[1:], y[1:]):

            a = (yA - yB) / (xA - xB)
            b = 0

            ny = np.array(f(x, a, b))

            count = np.sum(ny >= y)

            if count == x.size:
                break

        intersectX = np.abs(ny - np.ones_like(ny)*y[-1]).argmin()

        return (a, b, intersectX, x[intersectX])



    def estimateFreq(self):
        import numpy as np
        from math import sqrt, pi, log

        x, y = self.stepResponse

        # estimating Tn time
        # find and remove first maximum
        cy = y.copy()
        max1 = cy.max()
        id_max1 = np.where(cy == max1)[0][0]

        cy[0:id_max1] = y[-1]
        # find and remove first minimum
        min1 = cy.min()
        id_min1 = np.where(cy == min1)[0][0]

        cy[0:id_min1] = y[-1]
        # find the second maxiumum
        max2 = cy.max()
        id_max2 = np.where(cy == max2)[0][0]

        # compute Tn as time between first and second maximum
        a = y[-1].real
        dy = y.max().real - y[-1].real
        Tn = x[id_max2] - x[id_max1]

        w0 = 2*pi / (Tn*sqrt(1 - (log(dy/a)**2) / ((log(dy/a)**2) + pi**2)))
        print("Angular freq = {:.3f}, Freq = {:.3f}".format(w0.real, w0.real/(2*pi)))

        return (w0.real, w0.real/(2*pi))



    def validate(self, verbose=True):
        if self.detectSystemType(verbose) == 2:
            return (2, self.estimateFreq()[0])
        else:
            return (1, self.estimateTau()[3])


class D2CApprox:

    def __init__(self, x, y, kind):
        self._INF = 10**8 # discrete infinity
        self.xm = x
        self.ym = y
        self.kind = kind
        self._a = 2.0

        self.func = interp1d(x, y, kind=kind, bounds_error=False, fill_value=(0,y[-1]))

    def h(self, t):

        res = self.func(t)

        return res


    def grid(self, scope):
        import numpy as np

        # simulation scope
        T, step = scope

        phia = lambda t: (self.h(t/self._a) - self.h(0)) / (self.h(self._INF) - self.h(0))
        phi = lambda t: (self.h(t) - self.h(0)) / (self.h(self._INF) - self.h(0))

        X = lambda t: (phi(t) + phia(t)) / 2
        Y = lambda t: (phi(t) - phia(t)) / 2

        ticks = np.arange(0., T+step, step)
        x = np.array([X(t) for t in ticks])
        y = np.array([Y(t) for t in ticks])

        return (x, y)


class RBF:

    def __init__(self, indim, numCenters, outdim, betaMax):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [np.random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.betaMax = betaMax
        self.beta = betaMax
        self.W = np.random.random((self.numCenters, self.outdim))

    def _basisfunc(self, ci, c, d):
        # assert len(d) == self.indim
        return exp(-self.beta[ci] * norm(c-d)**2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                # print("--",type(c),type(x))
                G[xi,ci] = self._basisfunc(ci, c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        # rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        # self.centers = [X[i,] for i in rnd_idx]
        self.centers = [np.random.uniform(0, X.max(), self.indim)[0] for i in range(self.numCenters)] 
        self.beta = [np.random.uniform(0, self.betaMax, self.indim)[0] for i in range(self.numCenters)] 

        # print("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        # print(G)

        # calculate output weights (pseudoinverse)
        self.W = np.dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y



def getGridRange(points, grid_data):

    # range where function is convex and has positive values
    validRange = points[-1]
    for idx, val in enumerate(grid_data[1]):
        if val < 0:
            validRange = points[idx]
            break

    # range from beginning to inflection point (IP)
    validRangeIP = points[-1]
    max_val = grid_data[1][0]
    for idx, val in enumerate(grid_data[1]):
        if val > max_val:
            max_val  = val
            validRangeIP = points[idx]

    return (validRange, validRangeIP)



def removeCollidingParams(coeffs):
    out = []
    for idx, item in enumerate(coeffs):
        (_,_,t1,t2,_,_) = item
        if t1 != t2:
            out.append(item)

    return out
