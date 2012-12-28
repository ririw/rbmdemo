from unittest import TestCase
from matplotlib import pyplot

__author__ = 'riri'
import numpy as np
epsilon = 0.000001
def sigmoid(v):
    return 1/(1+np.exp(-v))

def hidden_activation_probability(v,W,c):
    # x is size k*v
    # W is size h*v
    #  W.T is size v*h
    # c is length h
    # result is size k*h
    return sigmoid(c + np.dot(v, W.T))

def hidden_activation_probability_naive(v,W,c):
    activations = np.zeros(W.shape[0])
    if v.size != W.shape[1]:
        #print v.size
        #print v.shape
        #print W.shape
        pass
    assert(v.size == W.shape[1])
    for i in range(W.shape[0]):
        activations[i] = sigmoid(c[i] + sum([W[i,j]*v[j] for j in range(W.shape[1])]))
    return activations

def visible_activation_probability(h,W,b):
    # h is length k*h
    # W is size h*v
    # b is length v
    # result is size k*v
    return sigmoid(b + np.dot(h,W))

def visible_activation_probability_naive(h,W,b):
    activations = np.zeros(W.shape[1])
    assert(h.size == W.shape[0])
    for j in range(W.shape[1]):
        activations[j] = sigmoid(b[j] + sum([W[i,j]*h[i] for i in range(W.shape[0])]))
    return activations

def sample_hidden_units(v,W,c):
    hidden_probabilities = hidden_activation_probability(v,W,c)
    return np.random.uniform(size=hidden_probabilities.shape) < hidden_probabilities

def sample_hidden_units_naive(v,W,c):
    hidden_probabilities = hidden_activation_probability_naive(v,W,c)
    return np.random.uniform(size=hidden_probabilities.shape) < hidden_probabilities

def sample_visible_units(h,W,b):
    visible_probabilities = visible_activation_probability(h,W,b)
    return np.random.uniform(size=visible_probabilities.shape) < visible_activation_probability(h,W,b)

def sample_visible_units_naive(h,W,b):
    visible_probabilities = visible_activation_probability_naive(h,W,b)
    return np.random.uniform(size=visible_probabilities.shape) < visible_activation_probability(h,W,b)

def rbmUpdate_naive(x, W, b, c, lr=0.1):
    h1 = sample_hidden_units_naive(x,W,c)
    v2 = sample_visible_units_naive(h1,W,b)
    q_v2 = visible_activation_probability_naive(h1,W,b)
    q_h2 = hidden_activation_probability_naive(v2,W,c)
    new_b = b + lr*(x-v2)
    new_c = c + lr*(h1-q_h2)
    a = np.outer(h1,x)
    b = np.outer(q_h2,v2.T)
    new_W = W + lr*(a-b)
    error = np.sum((x-q_v2)**2)
    return new_W,new_b,new_c,error

def rbmUpdate(x,W,b,c,lr=0.1):
    h1 = sample_hidden_units(x,W,c)
    v2 = sample_visible_units(h1,W,b)
    q_v2 = visible_activation_probability(h1,W,b)
    q_h2 = hidden_activation_probability(v2,W,c)
    new_b = b + lr*(x-v2)
    new_c = c + lr*(h1-q_h2)
    a = np.outer(h1,x)
    b = np.outer(q_h2,v2.T)
    new_W = W + lr*(a-b)
    error = np.sum(np.sum((x-q_v2)**2))
    return new_W,new_b,new_c,error

class RBM(object):
    def __init__(self, visible_units, hidden_units):
        self.v = visible_units
        self.h = hidden_units
        self.W = np.random.random(size=(hidden_units, visible_units))
        self.b = np.random.random(visible_units)
        self.c = np.random.random(hidden_units)

    def train(self, data, lr=0.05, max_iterations=1000, eps=0.1):
        iteration = 0
        last_error = eps+1
        while iteration < max_iterations and last_error > eps:
            for item in data:
                self.W,self.b,self.c,last_error = rbmUpdate(item, self.W,self.b,self.c,lr)
            iteration += 1
            if iteration % 10 == 0:
                print last_error

    def train_naive(self,data,lr=0.05,max_iterations=1000,eps=0.1):
        iteration = 0
        last_error = eps+1
        while iteration < max_iterations and last_error > eps:
            for item in data:
                self.W,self.b,self.c,last_error = rbmUpdate_naive(item, self.W,self.b,self.c,lr)
            iteration += 1
            if iteration % 10 == 0:
                print last_error
class TestAgainstNaive(object):
    def __init__(self, h_size, v_size):
        self.h_size = h_size
        self.v_size = v_size
    def test_hidden(self):
        h_size = self.h_size
        v_size = self.v_size
        ww = np.random.uniform(size=(h_size,v_size))
        bb = np.random.uniform(size=v_size)
        cc = np.random.uniform(size=h_size)
        vv = np.random.uniform(size=v_size)
        h1 = hidden_activation_probability_naive(vv,ww,cc)
        h2 = hidden_activation_probability(vv,ww,cc)
        assert(h1.shape == h2.shape)
        assert(h1.size == h_size)
        assert(all(np.abs(h1 - h2) < epsilon))
    def test_visible(self):
        h_size = self.h_size
        v_size = self.v_size
        ww = np.random.uniform(size=(h_size,v_size))
        bb = np.random.uniform(size=v_size)
        hh = np.random.uniform(size=h_size)
        v1 = visible_activation_probability_naive(hh,ww,bb)
        v2 = visible_activation_probability(hh,ww,bb)
        assert(v1.shape == v2.shape)
        assert(v1.size == v_size)
        assert(all(np.abs(v1 - v2) < epsilon))
    def test_update(self):
        h_size = self.h_size
        v_size = self.v_size
        ww = np.random.uniform(size=(h_size,v_size))
        bb = np.random.uniform(size=v_size)
        cc = np.random.uniform(size=h_size)
        vv = np.random.uniform(size=v_size)
        (nw1,nb1,nc1,e1) = rbmUpdate(vv,ww,bb,cc)
        (nw2,nb2,nc2,e2) = rbmUpdate_naive(vv,ww,bb,cc)

        # The bounds for this are a bit larger, because
        # more goes one, so there are more chances for
        # divergence.
        assert(np.all(np.abs(nw1-nw2) < epsilon*5))
        assert(np.all(np.abs(nb1-nb2) < epsilon*5))
        assert(np.all(np.abs(nc1-nc2) < epsilon*5))
        assert(np.all(np.abs(e1-e2) < epsilon*5))

def chunkfiles(files):
    splitwords = []
    vocab = set()
    for f in files:
        for line in f.readlines():
            splitwords.append(set(line.split()))
            vocab = vocab.union(line.split())
    entries = []
    vocablist = list(vocab)
    for entry in splitwords:
        entries.append([1 if word in entry else 0 for word in vocablist])
    return entries

def splitfiles(files):
    splitwords = []
    vocab = set()
    for f in files:
        for line in f.readlines():
            splitwords.append(set(line.split()))
            vocab = vocab.union(line.split())
    return splitwords


def go():
    tests = 1
    magnitude_mat = np.zeros((7,4*tests))
    with open('./movies.txt') as movies:
        with open('./matrices.txt') as matrices:
            terms = chunkfiles([movies, matrices])
    for i in range(tests):
        training_data = np.array(terms[1:-1])
        print training_data.shape
        r = RBM(399, 2)
        r.train(training_data ,max_iterations=500,lr=0.1)
        movterm = np.array([terms[1]])
        matterm = np.array([terms[-1]])
        print hidden_activation_probability(movterm,r.W,r.c)
        print hidden_activation_probability(matterm,r.W,r.c)
    pyplot.imshow(magnitude_mat, interpolation='nearest')

if __name__=='__main__':
    h_size = np.floor(np.random.rand()*100)
    v_size = np.floor(np.random.rand()*100)
    test = TestAgainstNaive(h_size,v_size)
    test.test_visible()
    test.test_hidden()
    test.test_update()
    go()