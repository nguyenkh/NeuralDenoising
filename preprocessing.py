import argparse
from sklearn.decomposition import MiniBatchDictionaryLearning
from common import read_file, save_file
import numpy as np

np.random.seed(35363)

def main():
    """
    Builds the dictionaries and the overcomplete word embeddings
    Usage:
        python preprocessing.py -input <original_embs_file> -output <overcomp_file> -factor <factor_overcomplete>
                                -bin <binary_file>
        
    <original_embs_file>: the original word embeddings is used to learn denoising
    <overcomp_file>: the file name of overcomplete word embeddings
    <factor_overcomplete>: a factor of overcomplete embeddings length (=factor * length of original word embeddings)
    <binary_file>: 1 for binary; 0 for text
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-factor', type=int, default=10)
    parser.add_argument('-bin', type=int, default=1)
    args = parser.parse_args()
    
    vocab, vecs = read_file(args.input, binary=args.bin)
    dict_comp = pre_complete_embs(vecs)
    dict_overcomp, vecs_overcomp = pre_overcomplete_embs(vecs, factor=args.factor)
    np.save(args.input + '.dict_comp', dict_comp)
    np.save(args.input + '.dict_overcomp', dict_overcomp)
    save_file(args.output, vocab, vecs_overcomp, binary=args.bin)
    print 'Preprocessing done!'
    
def pre_complete_embs(vecs, alpha=1.e-2):
    print 'X shape: ' + str(vecs.shape)
    X = vecs
    n_components = len(X)[1]
    print 'n_components: %d' %n_components
    dl = MiniBatchDictionaryLearning(n_components=n_components,alpha=1.e-2,batch_size=10,n_jobs=40)
    dl.fit(X)
    D = dl.components_
    print 'Complete dictionary shape: ' + str(D.shape)
    return D.T

def pre_overcomplete_embs(vecs, factor=10, alpha=1.e-2):
    print 'X shape: ' + str(vecs.shape)
    X = vecs
    n_components = factor * len(X)[1]
    print 'n_components: %d' %n_components
    dl = MiniBatchDictionaryLearning(n_components=n_components,alpha=1.e-2,batch_size=10,n_jobs=40)
    dl.fit(X)
    D = dl.components_
    overvecs = dl.transform(X)
    print 'Overcomplete dictionary shape: ' + str(D.shape)
    print 'Overcomplete vecs shape: ' + str(overvecs.shape)
    
    return D.T, overvecs

if __name__=='__main__':    
    main()
    
    