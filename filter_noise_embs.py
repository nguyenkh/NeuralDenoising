import time
import numpy as np
import theano
import theano.tensor as T 
from common import read_file, save_file, largest_eigenvalue
import argparse
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

rng = np.random.RandomState(12034)
srng = RandomStreams()

def main():
    """
    Noise filtering from word embeddings
    Usage:
        python filter_noise_embs.py -input <original_embs_file> -output <denoising_embs_file>
                                    -over <over_complete_embs_file> -iter <iteration> -bsize <batch_size>
    <original_embs_file>: the original word embeddings is used to learn denoising
    <denoising_embs_file>: the output name file of word denoising embeddings
    <over_complete_embs_file>: the overcomple word embeddings is used to learn overcomplete word denoising embeddings
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-over', action='store', default=False, dest='file_over')
    parser.add_argument('-iter', type=int)
    parser.add_argument('-bsize', type=int)
    
    args = parser.parse_args()
    
    vocab, vecs_in = read_file(args.input)   
    if args.file_over is False:
        vecs_dict = np.load(args.input + '.dict_comp')
        Q, S = initialize_parameters(vecs_dict)
        model = DeEmbs(vecs_in=vecs_in, batch_size=args.bsize, epochs=args.iter, Q=Q, S=S)
    else:
        vecs_dict = np.load(args.input + '.dict_overcomp')
        Q, S = initialize_parameters(vecs_dict)
        vc, vecs_over = read_file(args.file_over)
        assert vocab == vc
        model = DeEmbs(vecs_in=vecs_in, vecs_over=vecs_over,
                       batch_size=args.bsize, epochs=args.iter, Q=Q, S=S)    
    vecs_out = model.fit()
    save_file(args.output, vocab, vecs_out, binary=True)
    print 'Done........'

class HiddenLayer():
    def __init__(self, x, dim_in, dim_out, Q=None, S=None):
        self.x = x
        
        if Q is None:
            Q_values = np.asarray(rng.uniform(low=-np.sqrt(6./(dim_in + dim_out)), 
                                          high=np.sqrt(6./(dim_in + dim_out)),
                                          size=(dim_in, dim_out)),
                              dtype=theano.config.floatX)
            self.Q = theano.shared(value=Q_values, name='Q', borrow=True)
        else:
            self.Q = theano.shared(value=Q, name='Q', borrow=True)
        
        if S is None:
            S_values = np.asarray(rng.uniform(low=-np.sqrt(6./(dim_in + dim_out)), 
                                          high=np.sqrt(6./(dim_in + dim_out)),
                                          size=(dim_out, dim_out)),
                                  dtype=theano.config.floatX)
            self.S = theano.shared(value=S_values, name='S', borrow=True)
        else:
            self.S = theano.shared(value=S, name='S', borrow=True)
        
        self.params = [self.Q, self.S]
        
        B = T.dot(x, self.Q)
        B = self.dropout(B, p=0.5)
        Y = T.tanh(B)
        for _ in range(3):
            Y = T.tanh(T.dot(Y, self.S) + B)
        self.output = self.dropout(Y, p=0.2)
        
    def error(self):
        """
        Calculate error after one epoch
        """
        B = T.dot(self.x, self.Q)
        Y = T.tanh(B)
        for _ in range(3):
            Y = T.tanh(T.dot(Y, self.S) + B)
        return Y
    
    def dropout(self, X, p=0.):
        if p > 0:
            retain_prob = 1 - p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X
        
class DeEmbs():
    def __init__(self, vecs_in, vecs_over=None, batch_size, epochs, Q=None, S=None):
        self.X_data = vecs_in
        if vecs_over is None:
            self.Z_data = vecs_in
            self.dim_out = len(vecs_in[0])
        else:
            self.Z_data = vecs_over
            self.dim_out = len(vecs_over[0])
        self.Q = Q
        self.S = S
        self.dim_in = len(vecs_in[0])
        self.n_samples = len(vecs_in)
        self.l1_reg = np.float32(0.15)
        self.batch_size = batch_size
        self.epochs = epochs
                    
    def cosine(self, z, zhat):
        return 1.0 - T.sum(zhat * z) / (T.sqrt(T.sum(T.sqr(zhat)) * T.sum(T.sqr(z))))
           
    def shared_dataset(self, X_data, Z_data):
        """Builds a theano shared variable for input data"""
        X_shuffle, Z_shuffle = self.shuffle_data(X_data, Z_data)
        X_train = theano.shared(X_shuffle.astype('float32'))
        Z_train = theano.shared(Z_shuffle.astype('float32'))
        return X_train, Z_train
    
    def shuffle_data(self, X_data, Z_data):
        """Shuffle data for training"""
        per = np.random.permutation(len(X_data))
        X_shuffle = X_data[per]
        Z_shuffle = Z_data[per]
        return X_shuffle, Z_shuffle
    
    def build_shared_zeros(self, shape, name):
        """ Builds a theano shared variable filled with a zeros numpy array """
        return theano.shared(value=np.zeros(shape).astype('float32'), name=name, borrow=True) 

    def adadelta(self, cost, params, rho=0.95, eps=1.e-6):
        """
        Adadelta updates
        """
        gparams = T.grad(cost, params)
        updates = OrderedDict()
        for param, gparam in zip(params, gparams):
            accugrad = self.build_shared_zeros(param.shape.eval(), 'accugrad')
            accudelta = self.build_shared_zeros(param.shape.eval(), 'accudelta')
            agrad = (rho * accugrad + (1 - rho) * gparam * gparam).astype('float32')
            updates[accugrad] = agrad
            dx = (T.sqrt((accudelta + eps) / (agrad + eps)) * gparam).astype('float32')
            updates[param] = param - dx
            adelta = rho * accudelta + (1 - rho) * dx * dx
            updates[accudelta] = (adelta).astype('float32')            
        return updates
    
    def transform(self, X_data):
        X = T.fmatrix()
        def steps(x, Q, S):
            B = T.dot(x, Q)        
            return T.tanh(B)
        vecs, _ = theano.scan(steps, sequences=X,
                              non_sequences=[self.layers.Q, self.layers.S])
        proj = theano.function(inputs=[X], outputs=vecs)
        embs = proj(X_data)
        
        return embs
    
    def fit(self):
        X_train, Z_train = self.shared_dataset(self.X_data, self.Z_data)
        index = T.iscalar()
        X = T.fmatrix()
        Z = T.fmatrix()
        
        self.layers = HiddenLayer(X, self.dim_in, self.dim_out, self.Q, self.S)        
        self.params = self.layers.params

        cost = self.cosine(Z, self.layers.output) + self.l1_reg * T.sum(T.abs_(self.layers.S))
        error = self.cosine(Z, self.layers.error())
        
        batch_start = index * self.batch_size
        batch_stop = T.minimum(batch_start + self.batch_size, self.n_samples)
               
        updates = self.adadelta(cost, self.params)       
        train = theano.function(inputs=[index], outputs=cost, updates=updates,
                                givens={X:X_train[batch_start:batch_stop],
                                        Z:Z_train[batch_start:batch_stop]},
                                allow_input_downcast=True)
            
        calc_total_error = theano.function(inputs=[index], outputs=error,
                                           givens={X:X_train[batch_start:batch_stop],
                                                   Z:Z_train[batch_start:batch_stop]},
                                           allow_input_downcast=True)
        
        n_batches = self.n_samples // self.batch_size
        epoch = 0
        avg_loss = -1.
        print 'Training.......'
        start_time = time.time()
        while epoch < self.epochs:            
            for i in np.random.permutation(range(n_batches)):
                train(np.int32(i))
            epoch += 1
            loss = 0.
            for i in range(n_batches):
                loss += calc_total_error(np.int32(i))
            avg_loss = loss/float(n_batches)            
            print 'epoch: %d, loss: %f' % (epoch, avg_loss)
            
        embs = self.transform(self.X_data)
        t = time.time()-start_time    
        print 'denoising done with training time: %2f secs' %t           
        return embs

def initialize_parameters(vecs_dict):
    D = vecs_dict
    E = largest_eigenvalue(D)
    Q = np.dot(1/E, D)
    S = np.eye(Q.shape[1]) - 1/E * np.dot(D.T, D) 
    
    return Q, S
               
if __name__=='__main__':
    main()
    