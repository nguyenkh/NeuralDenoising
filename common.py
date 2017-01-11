import numpy as np
from numpy import fromstring, dtype
from scipy.linalg import eigh as largest_eigh
import struct

def smart_open(fname, mode='rb'):
    if fname.endswith('.gz'):
        import gzip
        return gzip.open(fname, mode)
    elif fname.endswith('.bz2'):
        import bz2
        return bz2.BZ2File(fname, mode)
    else:
        return open(fname, mode)

def read_file(binary_file, binary=1):
    vecs = []
    vocab = []
    if binary==1:
        with smart_open(binary_file, 'rb') as f:
            header = to_unicode(f.readline())
            vocab_size, vector_size = map(int, header.split())
            binary_len = dtype(np.float32).itemsize * vector_size
            for _ in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':
                        word.append(ch)
                word = to_unicode(b''.join(word))
                vocab.append(word)
                vec = fromstring(f.read(binary_len), dtype=np.float32)
                vecs.append(vec)
    else:
        with smart_open(binary_file, 'rb') as f:
            header = to_unicode(f.readline())
            if len(header.split()) == 2: vocab_size, vector_size = map(int, header.split())
            elif len(header.split()) > 2:
                parts = header.rstrip().split(" ")
                word, vec = parts[0], list(map(np.float32, parts[1:]))
                vocab.append(to_unicode(word))
                vecs.append(vec)
            for _, line in enumerate(f):
                parts = to_unicode(line.rstrip()).split(" ")
                word, vec = parts[0], list(map(np.float32, parts[1:]))
                vocab.append(to_unicode(word))
                vecs.append(vec)
                    
    vecs = np.array(vecs, dtype=np.float32)
    
    return vocab, vecs

def save_file(outfile, vocab, vecs, binary=1):
    assert len(vocab) == len(vecs)
    print 'Saving embeddings...'
    dim = len(vecs[0])
    if binary == 1:
        fvec = open(outfile, 'wb')
        fvec.write('%d %d\n' % (len(vecs), dim))
        fvec.write('\n')
        for word, vector in zip(vocab, vecs):
            fvec.write('%s ' % word)
            for s in vector:
                fvec.write(struct.pack('f', s))
            fvec.write('\n')
    elif binary == 0:
        fvec = open(outfile, 'w')
        fvec.write('%d %d\n' % (len(vecs), dim))
        for word, vector in zip(vocab, vecs):
            vector_str = ' '.join([str(s) for s in vector])
            fvec.write('%s %s\n' % (word, vector_str))
    
#     with smart_open(outfile, 'wb') as f:
#         f.write(to_utf8("%s %s\n" % vecs.shape))
#         for i in range(len(vecs)):
#             word = vocab[i]
#             if binary==1:
#                 arr = vecs[i]
#                 f.write(to_utf8(word) + b' ' + arr.tostring())
#             else:
#                 f.write(to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in vecs[i,:]))))

def to_utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    else:
        return unicode(text, encoding, errors=errors).encode('utf8')

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    else:
        return unicode(text, encoding=encoding, errors=errors)

def largest_eigenvalue(mat):
    D = np.dot(mat.T, mat)
    N = len(D)
    k = 1
    eigvalues = largest_eigh(D, eigvals=(N-k,N-1), eigvals_only = True)
    print 'Q shape: ' + str(D.shape)
    return eigvalues[0]
    