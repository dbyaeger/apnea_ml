"""Viterbi"""
# cython: profile=True
import cython
import unittest
import abc

import numpy as np

#print("{} ({})".format(__name__, "compiled" if cython.compiled else "interpreted"))



class Observation(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        # set attributes here needed during calculate()
        pass

    @abc.abstractmethod
    def calculate(self, _t, out):
        """
        input: target (instance at some time t), pool
        output: score / prob
        """
        #out[:] = 1
        #raise NotImplementedError
        pass

class Transition(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        # set attributes here needed during calculate()
        pass

    @abc.abstractmethod
    def calculate(self, t, jindex, i, out):
        """
        input: from nodes jindex @ time t-1, to node i @ time t
        output: score / prob
        """
        raise NotImplementedError

@cython.locals(t=cython.int,
               i=cython.int,
               score_cur=cython.int,
               score_prv=cython.int)
class Viterbi(object):
    def __init__(self):
        pass

    def search(self, T, P, observation, transition, log_domain=True):
        """P is the number of candidates at each time
        T is the number of available time points"""
        assert isinstance(T, int)
        assert T > 1
        assert isinstance(P, int)
        assert P < 65536 # could adjust dtype automatically, currently set at uint16
        assert isinstance(observation, Observation)
        assert isinstance(transition, Transition)
        trans = np.empty(P)
        zeta  = np.empty(P)
        score = np.empty((2, P)) # one row for previous, and one for current score (no tracking for all t, saving memory)
        path  = np.zeros((T, P), np.uint16) - 1 # often this matrix has less than %1 of meaningful entries, could be made sparse
        seq   = np.zeros(T,      np.uint16) - 1 # the - 1 (really: 65535) helps catch bugs
        if log_domain:
            opr = np.add
            thr = -np.inf
        else:
            opr = np.multiply
            thr = 0
        # init
        observation.calculate(0, score[0])
        jindex = np.where(score[0])[0] # active FROM nodes
        assert len(jindex), 'no observations for target[0]'
        # forward
        for t in range(1, T):
            score_cur = t % 2
            score_prv = (t - 1) % 2
            observation.calculate(t, score[score_cur])
            iindex = np.where(score[score_cur] > thr)[0] # possible TO nodes
            assert len(iindex), 'no observation probabilities above pruning threshold for target[%d]' % t
            for i in iindex: # TO this node TODO: can this be parallelized? I think so!
                transition.calculate(t, jindex, i, trans)  # trans is the output
                #zeta[jindex] = score[(t - 1) % 2, jindex] + trans[jindex]
                zeta[jindex] = opr(score[score_prv, jindex], trans[jindex])
                path[t, i] = zindex = jindex[zeta[jindex].argmax()]
                #score[t % 2, i] += zeta[zindex]
                score[score_cur, i] = opr(score[score_cur, i], zeta[zindex])
            assert any(score[score_cur] > thr), 'score/prob[t] must not be all below pruning threshold'
            jindex = iindex # new active FROM nodes
        # backward
        assert score_cur == (T - 1) % 2
        #seq[-1] = score[score_cur].argmax()
        seq[-1] = iindex[score[score_cur,iindex].argmax()] # AK BUGFIX
        for t in range(T - 1, 0, -1):
            seq[t-1] = path[t, seq[t]]
        return seq, score[score_cur, seq[-1]]

    def draw_mpl(self, seq, path, score):
        raise NotImplementedError # this assumes existence of score for all t, which is not currently implemented
        #T, P = score.shape
        #from matplotlib import pylab as pp
        ## show score
        #pp.imshow(score.transpose(), cmap=pp.cm.gray, aspect='auto', interpolation='nearest', origin='lower')
        #if 0: # show best local paths
            #for t in range(1, T):
                #iindex = numpy.where(score[t])[0] # a score exists here
                #for i in iindex: # TO this node
                    #pp.plot([t-1, t], [path[t,i], i], color=[1,0,0])
        ## show best global path
        #for t in range(1, T):
            #pp.plot([t-1, t], [seq[t-1], seq[t]], color=[0,1,0])
        #pp.axis([-.5, T-.5, -0.5 ,P-.5])
        #pp.xlabel('target')
        #pp.ylabel('pool')
        #pp.xticks(numpy.arange(T))
        #pp.yticks(numpy.arange(P))
        #pp.grid(True)
        #pp.show()

    def draw_pg(self, seq, path, score):
        raise NotImplementedError

    draw = draw_mpl


class ObservationUnvectorized(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass
        # set attributes here needed during calculate()

    @abc.abstractmethod
    def calculate(self, t, i):
        """
        input: target (instance at some time t), pool
        output: score / prob
        """
        #raise NotImplementedError
        #return 1
        pass

class TransitionUnvectorized(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        # set attributes here needed during calculate()
        pass

    @abc.abstractmethod
    def calculate(self, t, j, i):
        pass
        #raise NotImplementedError
        #return 1


class ViterbiUnvectorized(object):
    """The purpose of this class is to simulate unvectorized C code"""
    def __init__(self):
        pass

    def search(self, T, P, observation, transition, log_domain=True):
        assert isinstance(observation, ObservationUnvectorized)
        assert isinstance(transition, TransitionUnvectorized)
        path  = np.zeros((T, P), np.uint16) - 1 # often this matrix has less than %1 of meaningful entries, could be made sparse
        seq   = np.zeros(T,      np.uint16) - 1 # the - 1 helps catch bugs
        assert P < 65536 # could adjust dtype automatically
        score = np.empty((2, P)) # one row for previous, and one for current score (no tracking for all t, saving memory)
        if log_domain:
            opr = np.add
        else:
            opr = np.multiply
        # init
        for i in range(P):
            # BEGIN
            score[0, i] = observation.calculate(0, i) # inline
            # END
        # forward
        for t in range(1, T):
            score_cur = t % 2
            score_prv = (t - 1) % 2
            for i in range(P):
                # BEGIN
                score[score_cur, i] = observation.calculate(t, i) # inline
                # END
            for i in range(P):
                if score[score_cur, i]: # possible TO node i
                    zeta_max = zeta_argmax = 0
                    for j in range(P):
                        if score[score_prv, j]: # possible FROM node j
                            # BEGIN
                            trans = transition.calculate(t, j, i) # inline
                            # END
                            zeta = opr(score[score_prv, j], trans)
                            if zeta > zeta_max:
                                zeta_max = zeta
                                zeta_argmax = j
                    score[score_cur, i] = opr(score[score_cur, i], zeta_max)
                    path[t, i] = zeta_argmax
        # backward
        score_cur = (T - 1) % 2
        score_max = 0
        for i in range(P):
            if score[score_cur, i] > score_max:
                score_max = score[score_cur, i]
                seq[-1] = i
        for t in range(T - 1, 0, -1):
            seq[t-1] = path[t, seq[t]]
        return seq, score[score_cur, seq[-1]]


# class Test(unittest.TestCase):
#     def test_viterbi_example1(self):
#         target = np.array([[1, 1], [2, 2], [50, 50]])
#         pool = np.array([[1, 1], [2, 2], [42, 42], [50, 50], [2, 2]])
#         class MyObservation(Observation):
#             def __init__(self, target, pool):
#                 Observation.__init__(self)
#                 self.target = target
#                 self.pool = pool
#             def calculate(self, t, obs):
#                 """return observation probabilities (exact for n-d)"""
#                 obs[:] = np.all((self.target[t] == self.pool), axis=1)
#         class MyTransition(Transition):
#             def __init__(self, pool):
#                 Transition.__init__(self)
#                 self.pool = pool
#             def calculate(self, t, jindex, i, trans):
#                 """return transition probabilities from nodes j (vector) to node i (scalar)"""
#                 trans[jindex] = 0.5
#                 if i > 0:
#                     trans[i-1] = 1.0
#         viterbi = Viterbi()
#         index, _final = viterbi.search(len(target), len(pool), MyObservation(target, pool), MyTransition(pool), log_domain=False)
#         self.assertTrue(np.all(index == [0, 1, 3]))
#         index, _final = viterbi.search(len(target), len(pool), MyObservation(target, pool), MyTransition(pool), log_domain=True)
#         self.assertTrue(np.all(index == [0, 1, 3]))
#
#     def test_viterbi_compare_vectorized_vs_unvectorized(self):
#         from numpy import random
#         P = 50; T = 10
#         target = random.rand(T)
#         pool   = random.rand(P)
#         class MyObservation(Observation):
#             def __init__(self, pool, target):
#                 Observation.__init__(self)
#                 self.pool = pool
#                 self.target = target
#             def calculate(self, t, obs):
#                 """return observation scores (exact for n-d)"""
#                 obs[:] = (1 / (abs(self.target[t] - self.pool) + 1e-6)).flat
#         class MyTransition(Transition):
#             def __init__(self, pool):
#                 Transition.__init__(self)
#                 self.pool = pool
#             def calculate(self, t, jindex, i, trans):
#                 """return transition scores from nodes j (vector) to node i (scalar)"""
#                 trans[jindex] = ((self.pool[i] - self.pool[jindex]) > 0).astype(np.float)
#                 #trans /= max(trans) # normalize
#                 trans *= 100 # tradeoff between best observation and observing transition
#         index1, final1 = Viterbi().search(T, P, MyObservation(pool, target), MyTransition(pool))
#         #assert all(numpy.diff(pool[index]) > 0)
#         class MyObservationUnvectorized(ObservationUnvectorized):
#             def __init__(self, pool, target):
#                 ObservationUnvectorized.__init__(self)
#                 self.pool = pool
#                 self.target = target
#             def calculate(self, t, i):
#                 """return observation scores (exact for n-d)"""
#                 return (1 / (abs(self.target[t] - self.pool[i]) + 1e-6))
#         class MyTransitionUnvectorized(TransitionUnvectorized):
#             def __init__(self, pool):
#                 TransitionUnvectorized.__init__(self)
#                 self.pool = pool
#             def calculate(self, t, j, i):
#                 """return transition scores from nodes j (vector) to node i (scalar)"""
#                 trans = float((self.pool[i] - self.pool[j]) > 0)
#                 #trans /= max(trans) # normalize
#                 trans *= 100 # tradeoff between best observation and observing transition
#                 return trans
#         index2, final2 = ViterbiUnvectorized().search(T, P, MyObservationUnvectorized(pool, target), MyTransitionUnvectorized(pool))
#         print('{}\n\n{}\n'.format(index1, index2))
#         assert (index1 == index2).all()
#         assert np.allclose(final1, final2)
#
#     #def NOtest_viterbi_vizualization(self):
#         #import numpy.random
#         #N = 20
#         #target = numpy.linspace(0, 1, N)
#         #pool = numpy.random.rand(N)
#         #def observe(i, target, pool, obs):
#             #"""return observation scores (exact for n-d)"""
#             #obs[:] = (1 / (abs(target[i] - pool) + 1e-6)).flat
#             #obs /= max(obs) # normalize
#         #def transition(j, i, pool, trans):
#             #"""return transition scores from nodes j (vector) to node i (scalar)"""
#             #trans[j] = (1 / (abs(i - j) + 1e-6)).flat
#             #trans /= max(trans) # normalize
#             #trans *= 1 # tradeoff between best observation and close-by transition
#         ##index, final = viterbi_search(target, pool, observe, transition, log=True, VIZ=True)


class ViterbiUnvectorizedEuclidian(object):
    """The purpose of this class is to simulate unvectorized C code"""
    def __init__(self, X):
        self.X = X # pool feature matrix

    def search(self, S, log_domain=True):
        assert isinstance(S, np.ndarray)
        X = self.X
        assert S.shape[1] == X.shape[1]
        T = S.shape[1] # number of frames in the target feature matrix
        P = X.shape[1] # number of frames in the pool feature matrix
        path  = np.zeros((T, P), np.uint16) - 1 # often this matrix has less than %1 of meaningful entries, could be made sparse
        seq   = np.zeros(T,      np.uint16) - 1 # the - 1 helps catch bugs
        assert P < 65536 # could adjust dtype automatically
        score = np.empty((2, P)) # one row for previous, and one for current score (no tracking for all t, saving memory)
        if log_domain:
            opr = np.add
        else:
            opr = np.multiply
        # init
        for i in range(P):
            # BEGIN
            #score[0, i] = observation.calculate(0, i) # inline
            score[0, i] = -(np.sum((S[0] - X[i]) ** 2)) ** 0.5
            # END
        # forward
        for t in range(1, T):
            score_cur = t % 2
            score_prv = (t - 1) % 2
            for i in range(P):
                # BEGIN
                #score[score_cur, i] = observation.calculate(t, i) # inline
                score[score_cur, i] = -(np.sum((S[t] - X[i]) ** 2)) ** 0.5
                # END
            for i in range(P):
                if score[score_cur, i]: # possible TO node i
                    zeta_max = zeta_argmax = 0
                    for j in range(P):
                        if score[score_prv, j]: # possible FROM node j
                            # BEGIN
                            #trans = transition.calculate(t, j, i) # inline
                            trans = -(np.sum((X[j] - X[i]) ** 2)) ** 0.5
                            # END
                            zeta = opr(score[score_prv, j], trans)
                            if zeta > zeta_max:
                                zeta_max = zeta
                                zeta_argmax = j
                    score[score_cur, i] = opr(score[score_cur, i], zeta_max)
                    path[t, i] = zeta_argmax
        # backward
        score_cur = (T - 1) % 2
        score_max = 0
        for i in range(P):
            if score[score_cur, i] > score_max:
                score_max = score[score_cur, i]
                seq[-1] = i
        for t in range(T - 1, 0, -1):
            seq[t-1] = path[t, seq[t]]
        return seq, score[score_cur, seq[-1]]


def task_image():
    class MyObservation(Observation):
        def __init__(self, R):
            self.R = R
        def calculate(self, t, out):
            out[:] = self.R[t]
    
    class MyTransition(Transition):
        def __init__(self, R, weight):
            self.R = R
            self.weight = weight
        def calculate(self, t, jindex, i, out):
            out[jindex] = -abs(jindex - i) * self.weight
    
    X = np.random.randn(1e3,1e2)
    seq, _ = Viterbi().search(X.shape[0], X.shape[1], MyObservation(X), MyTransition(X, 1.0))


def post_process(posterior: np.ndarray, transition: np.ndarray) -> np.ndarray:
    N, d = posterior.shape  # N is number of sequence steps, d is number of classes
    assert d == transition.shape[0]
    assert transition.shape[1] == d, 'transition matrix must be square'

    class MyObservation(Observation):
        def __init__(self, O):
            self.O = np.log(O + 1e-64)

        def calculate(self, t, out):
            out[:] = self.O[t]

    class MyTransition(Transition):
        def __init__(self, T, weight):
            self.T = np.log(T + 1e-64)
            self.weight = weight

        def calculate(self, t, jindex, i, out):
            out[jindex] = self.T[jindex, i]

    seq, _ = Viterbi().search(N, d, MyObservation(posterior), MyTransition(transition, 1.0))
    return seq


def example_post_process():
    # simple 3-class example, cyclical state transitions (1->1 or 2, 2->2 or 3, 3->3 or 1)
    # ignoring starting probabilities
    transition = np.array([[0.5, 0.5, 0],
                           [0, 0.5, 0.5],
                           [0.5, 0, 0.5]])  # row is "from", col is "to"
    posterior = np.array([[0.9, 0.1, 0],
                          [0.1, 0.9, 0],
                          [0.6, 0.4, 0],
                          [0, 0.9, 0.1],
                          [0, 0.1, 0.9]])

    mls = np.argmax(posterior, axis=1)
    print("maximum likelihood state sequence:")
    print(mls)
    vss = post_process(posterior, transition)
    print("Viterbi smoothed state sequence:")
    print(vss)


if __name__ == '__main__':
    example_post_process()
