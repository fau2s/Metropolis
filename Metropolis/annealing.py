import random
import math
from builtins import print

import numpy as np
# from Metropolis import data
import data
from timeit import default_timer as timer


def simulatedAnnealing(beta, sol, e, A, G):
    bestEnergy = e
    bestSol = sol
    k = 1
    reject = 0
    while stop(k, reject):
        newSol = neighbor(sol, 1 / beta)
        newEnergy = energy(newSol, A, G)
        if math.exp(min(-beta * (newEnergy - e), 100)) > random.random():
            sol = newSol
            e = newEnergy
            reject = 0
        else:
            reject = reject + 1
        if newEnergy < bestEnergy:
            bestEnergy = newEnergy
            bestSol = newSol
        k = k + 1
    return bestSol


def stop(k, reject):
    if k > 10000 | reject > 500:
        return False
    return True


def energy(w, A, G):
    t = np.matmul(A, w)
    t = np.matmul(G, t)
    t = np.matmul(A, t)
    return -np.matmul(w, t) / np.matmul(w, np.matmul(G, w))


def neighbor(s0, T):
    # random +/-1 vector
    # deltas = random.sample(range(0, math.floor(100*T)),10)
    deltas = np.random.choice([-1, 1], size=10)
    P = np.random.randint(10, size=5)
    snew = np.zeros(10)
    snew[P] = s0[P] + deltas[P]

    return snew


class anneal:
    def __init__(self, parent, wexh, A, G, L, T=10, ):
        # Check input
        # if not nargin: #user wants default options, give it and stop
        #     minimum = define
        #     return
        # elif nargin<2: #user gave only objective function, throw error
        #     error('MATLAB:anneal:noParent','You need to input a first guess.')
        # elif nargin<3: #user gave no options structure, use default
        #     options=define
        # else: #user gave all input, check if options structure is complete
        #     if ~isstruct(options):
        #         error('MATLAB:anneal:badOptions'
        #             'Input argument ''options'' is not a structure')
        #
        #     fs = {'CoolSched','Generator','InitTemp','MaxConsRej','RevertTries'
        #         'MaxSuccess','MaxTries','StopTemp','StopVal','Verbosity'}
        #     for nm in range(fs.size):
        #         if ~isfield(options,fs{nm}):
        #             options.(fs{nm}) = define.(fs{nm}

        # main settings
        self.Tinit = 10.  # 0.1    # initial temp
        self.minT = 1e-8  # stopping temp
        self.minF = -np.inf
        self.max_consec_rejections = 1000  # 0.3*scipy.misc.comb(A.shape[0], L)
        self.revert_tries = 500
        self.max_try = 300  # 0.3*scipy.misc.comb(A.shape[0], L)
        self.max_success = 20
        self.report = 0
        self.k = 1  # boltzmann constant

        self.L = L

        self.exhdist = np.array([0.])

        # counters etc
        self.itry = 0
        self.success = 0
        self.revert = 0
        self.finished = False
        self.consec = 0
        self.T = self.Tinit
        self.initenergy = self.loss(A, G, parent)
        self.nfwexh = self.normflip(wexh)
        self.exhdist[0] = np.linalg.norm(self.normflip(parent) - self.nfwexh)
        self.oldenergy = self.initenergy
        self.bestenergy = self.initenergy
        self.bestinput = parent
        self.total = 0
        self.totalsuccess = 0
        self.parent = parent
        self.A = A
        self.G = G
        if self.report == 2:
            print(1, '\n  T = #7.5f, loss = #10.5f\n', self.T, self.oldenergy)

    def process(self, isOT):
        self.itry = 0
        self.success = 0
        self.consec = 0
        self.revert = 0
        self.finished = False
        self.totalsuccess = 0
        self.oldenergy = self.initenergy
        self.bestenergy = self.initenergy

        finished = self.finished
        # start = timer()
        runtime = 0

        self.T = self.calculateT(isOT, self.parent)

        while not finished:
            self.itry += 1  # just an iteration counter
            current = self.parent

            # # Stop / decrement T criteria
            if (self.itry >= self.max_try) | (self.success >= self.max_success):
                if (self.T < self.minT) | (self.consec >= self.max_consec_rejections):
                    finished = True
                    self.total += self.itry
                    break
                else:
                    self.T = self.cool()  # decrease T according to cooling schedule
                    if self.report == 2:  # output
                        print(1, '  T = #7.5f, loss = #10.5f\n', self.T, self.oldenergy)
                    self.total += self.itry
                    self.totalsuccess += self.success
                    self.itry = 1
                    self.success = 1
            if isOT:
                newparam = self.neigh(current, self.L, self.T)
                newenergy = self.loss(self.A, self.G, newparam)
            else:
                newparam = self.neighNode(current, self.L, self.T)
                newparam1, newenergy = data.valami(newparam, self.A, self.G)
                x = np.zeros(len(current))
                x[newparam != 0] = newparam1
                newparam = x
            self.exhdist = np.insert(self.exhdist, self.exhdist.size,
                                     np.linalg.norm(self.normflip(newparam) - self.nfwexh))
            # record the best
            if newenergy < self.bestenergy:
                self.bestenergy = newenergy.copy()
                self.bestinput = newparam.copy()
            if newenergy < self.minF:
                self.parent = newparam.copy()
                self.oldenergy = newenergy.copy()
                break
            if self.oldenergy - newenergy > 1e-6:
                self.parent = newparam.copy()
                self.oldenergy = newenergy.copy()
                self.success += 1
                self.consec = 0
            else:
                if random.random() < math.exp((self.oldenergy - newenergy) / (self.k * self.T)):
                    self.parent = newparam.copy()
                    self.oldenergy = newenergy.copy()
                    self.success += 1
                else:
                    self.consec += 1
                    if 0 == np.mod(self.consec, self.revert_tries):
                        # revert to the best solution thus far
                        self.parent = self.bestinput.copy()
                        self.oldenergy = self.bestenergy.copy()
                        self.revert += 1

            # end = timer()
            # print([self.itry,end-start, self.T, self.bestenergy])
            # if end - start > .1:
            #     finished = True

        if self.report:
            print(1, '====================================\n')
            print(1, '  Total final loss:        \t#g\n', self.bestenergy)
            print(1, '  Number of function calls:\t#i\n', self.total)
            print(1, '  Initial temperature:     \t#g\n', self.Tinit)
            print(1, '  Final temperature:       \t#g\n', self.T)
            print(1, '  Consecutive rejections:  \t#i\n', self.consec)
            print(1, '  Successful jumps:        \t#i\n', self.totalsuccess)
            print(1, '  Reverts to best point:   \t#i\n', self.revert)
            print(1, '---------------------------------\n')
            print(1, '  Greedy-Exh distance:     \t#g\n', self.exhdist[0])
            print(1, '  SA-Exh distance:         \t#g\n', np.linalg.norm(self.normflip(self.bestinput) - self.nfwexh))
            print(1, '  Closer than Greedy:      \t#g\n', sum(self.exhdist < self.exhdist[0]))
            print(1, '  Closer than Greedy*1.25: \t#g\n', sum(self.exhdist < 1.25 * self.exhdist[0]))
            print(1, '====================================\n')
            # plot (exhdist)
        return self.bestinput, -self.bestenergy

    def normflip(self, vec):
        # normalize and flip a vector so that the first nonzero element is positive
        # useful for comparing the closeness of two vectors
        nvec = vec / np.linalg.norm(vec)
        nzv = nvec[np.nonzero(nvec)]
        if nzv[0] < 0:
            nvec *= -1
        return nvec

    def generator(self, x):
        return x + (np.random.permutation(x.size) == x.size - 1) * np.random.randn() / 100

    def loss(self, A, G, w):
        # Assume A and G exist as global variables
        # load AG;
        if np.linalg.norm(w) == 0:
            return np.Inf
        else:
            # val=-(w'*A'*G*A*w)/(w'*G*w);
            # val =- (w'*A*G*A'*w)/(w'*G*w); #fixed NF 20120802
            t = np.matmul(np.transpose(A), w)
            t = np.matmul(G, t)
            t = np.matmul(A, t)
            return -np.matmul(w, t) / np.matmul(w, np.matmul(G, w))

        [Vcurr, Dcurr] = la.max_eigvector(A, G)
        return Dcurr

    def cool(self):
        return .8 * self.T

    def neigh(self, w, L, T):
        # ANNEAL_NEIGH - computes a "neighbour" w2 given a valid point w # This funciton is called on each iteration of simulated annealing
        # @Author: Norbert Fogarasi

        # w is an n-dimensional vector with maximum L non-zero elements
        # on each iteration, we make 2 steps:
        # 1) we take the non-zero values and change upto L of them by +/- 1
        # 2) we choose L dimensions randomly and insert the changed values
        n = w.shape

        # step 1) compute the new values
        # compute n random delta values. Size of jump depends on T.
        Tm = max(T, 0.01)
        delta = np.ceil((200 * Tm + 1) * np.random.rand(n[0]) - (100 * Tm) - 1)
        # add changes to get new values
        newvals = np.floor(w + delta)

        # step 2) determine dimensions
        origdims = w != 0
        allidx = range(n[0])
        origidx = np.extract(origdims, allidx)
        nonidx = np.setdiff1d(allidx, origidx)
        numnonzero = origidx.size

        maxnumdimchange = np.ceil(min([numnonzero * T, numnonzero, n[0] - numnonzero]))
        numdimchange = np.floor((maxnumdimchange + 1) * np.random.rand())

        # now swap exactly numdimchange elements between the sets.
        R = np.random.permutation(numnonzero)
        # take out numdimchange random elements
        t = R[range(int(numdimchange))]
        newidx = np.setdiff1d(origidx, origidx[t])
        # add in numdimchange random dims
        Q = np.random.permutation(nonidx.shape[0])
        newidx = np.insert(newidx, newidx.size, nonidx[Q[range(int(numdimchange))]])

        # deal with the case where there are too many zeros in w...
        d = L - newidx.shape[0]
        if d > 0:
            if Q.shape[0] < numdimchange + 1:
                newidx = np.insert(newidx, newidx.size, origidx[R[0]])
            else:
                newidx = np.insert(newidx, newidx.size, nonidx[Q[int(numdimchange)]])

        # take only the newidx values, rest to 0
        foo = np.zeros(n)
        foo[newidx] = 1

        return np.matmul(np.diag(foo), newvals)

    def neighNode(self, w, L, T):
        # neighNode - return a new weight vector which is neighbouring node of the input weight#
        #  This funciton is called on each iteration of simulated annealing
        # @Author: Attila Racz

        # w is an n-dimensional vector with maximum L non-zero elements
        # on each iteration, we make 2 steps:
        # 1) we take the non-zero values and change upto L of them by +/- 1
        # 2) we choose L dimensions randomly and insert the changed values
        newIx = np.random.choice(np.where(w == 0)[0], 1)
        w[np.random.choice(np.nonzero(w)[0], 1)] = 0
        w[newIx] = 1
        return w

    def calcKhi(self, Emax, Emin, T):
        return np.sum(np.exp(-Emax / T)) / np.sum(np.exp(-Emin / T))

    def calculateT(self, isOT, parent):
        current = np.abs(parent)
        p = 1
        epsilon = 1e-4
        khi0 = 0.8
        T = 100
        sizeOfPopulation = 1000
        pop = 0
        Emax = np.zeros(sizeOfPopulation)
        Emin = np.zeros(sizeOfPopulation)

        oldenergy = self.initenergy

        while pop < sizeOfPopulation:
            if isOT:
                current = self.neigh(current, self.L, 1e6)
                newenergy = self.loss(self.A, self.G, current)
            else:
                current = self.neighNode(current, self.L, 1e6)
                current1, newenergy = data.valami(current, self.A, self.G)

            if newenergy - oldenergy > 0:
                Emax[pop] = newenergy
                Emin[pop] = oldenergy
                oldenergy = newenergy
                pop += 1
            else:
                oldenergy = newenergy

        khi = self.calcKhi(Emax, Emin, T)

        while khi - khi0 > epsilon:
            khi = self.calcKhi(Emax, Emin, T)
            T = np.power(T * (np.log(khi) / np.log(khi0)), 1 / p)

        return T


if __name__ == "__main__":
    beta = 1
    sol = np.zeros(10)
    e = 1000
    A = np.ones((10, 10))
    G = np.ones((10, 10))
    simulatedAnnealing(beta, sol, e, A, G)
