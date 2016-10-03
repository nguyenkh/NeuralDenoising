import argparse
import spams
import time
from common import read_file, save_file
import numpy as np

np.random.seed(35363)

def main():
    """
    Builds the dictionaries and the overcomplete word embeddings
    Usage:
        python preprocessing.py -input <original_embs_file> -output <overcomp_file> -factor <factor_overcomplete>
        
    <original_embs_file>: the original word embeddings is used to learn denoising
    <overcomp_file>: the file name of overcomplete word embeddings
    <factor_overcomplete>: a factor of overcomplete embeddings length (=factor * length of original word embeddings)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-factor', type=int, default=10)
    args = parser.parse_args()
    
    vocab, vecs = read_file(args.input, binary=True)
    dict_comp = learn_dict(vecs.T, factor=1)
    dict_overcomp = learn_dict(vecs.T, factor=args.factor)
    dim_over = args.factor * len(vecs[0])
    vecs_overcomp = overcomplete_embs(vecs.T, dim_over)
    np.save(args.input + '.dict_comp', dict_comp)
    np.save(args.input + '.dict_overcomp', dict_overcomp)
    save_file(args.output, vocab, vecs_overcomp.T, binary=True)
    print 'Preprocessing done!'

def overcomplete_embs(vecs, dim_over, lambda1=1.e-6):
    print 'X shape: ' + str(vecs.shape)
    n_features = len(vecs)
    n_components = dim_over
    X = np.asfortranarray(vecs, dtype=myfloat)
    D = np.asfortranarray(np.random.normal(size = (n_features,n_components)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)),dtype=myfloat)
    ind_groups = np.array(xrange(0,X.shape[1],10),dtype=np.int32) #indices of the first signals in each group
    # parameters of the optimization procedure are chosen
    itermax = 1000
    tol = 1e-3
    mode = spams.PENALTY
    lambda1 = lambda1 # squared norm of the residual should be less than 0.1
    numThreads = -1 # number of processors/cores to use the default choice is -1
                # and uses all the cores of the machine
    alpha0 = np.zeros((D.shape[1],X.shape[1]),dtype= myfloat,order="FORTRAN")
    tic = time.time()
    alpha = spams.l1L2BCD(X,D,alpha0,ind_groups,lambda1 = lambda1,mode = mode,
                          itermax = itermax,tol = tol,numThreads = numThreads)
    tac = time.time()
    t = tac - tic
    print 'Z shape: ' + str(alpha.shape)
    print "%f signals processed per second" %(X.shape[1] / t)
    return alpha

def learn_dict(vecs, factor):
    print 'X shape: ' + str(vecs.shape)
    X = vecs
    n_components = len(X)
    X = np.asfortranarray(X / np.tile(np.sqrt((X * X).sum(axis=0)),(X.shape[0],1)),dtype = myfloat)
    param = { 'K' : factor*n_components, # learns a dictionary with K elements
          'lambda1' : 0.15, 'numThreads' : -1, 'batchsize' : 50,
          'iter' : 1000}
    
    D = spams.trainDL(X,**param)
    print 'D shape: ' + str(D.shape)
    return D

if __name__=='__main__':    
    main()
    
    