import numpy as np
from numpy import fromstring, dtype
from scipy.linalg import eigh as largest_eigh

def smart_open(fname, mode='rb'):
    if fname.endswith('.gz'):
        import gzip
        return gzip.open(fname, mode)
    elif fname.endswith('.bz2'):
        import bz2
        return bz2.BZ2File(fname, mode)
    else:
        return open(fname, mode)
    
def build_vocab(list_vocab):
    vocab = {}
    for index, word in enumerate(list_vocab):
        vocab[word] = index
    return vocab

def read_vocab(vocab_file):
    vocab = {}
    words = []
    with smart_open(vocab_file, 'rb') as f:
        for line in f:
            t = line.strip().split('\n')
            words.append(t[0])
    for index, word in enumerate(words):
        vocab[word] = index
    return vocab

def read_words(infile):
    words = []
    with smart_open(infile, 'rb') as f:
        for line in f:
            t = line.strip().split('\n')
            words.append(t[0])
    return words

def read_file(binary_file, words_file=None, binary=True):
    weights = []
    vc = []
    if binary==True:
        with smart_open(binary_file, 'rb') as f:
            header = to_unicode(f.readline())
            vocab_size, vector_size = map(int, header.split())
            if words_file is None:
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
                    vc.append(word)
                    weight = fromstring(f.read(binary_len), dtype=np.float32)
                    weights.append(weight)
            else:
                words = read_words(words_file)
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
                    if word in words: 
                        vc.append(word)
                        weight = fromstring(f.read(binary_len), dtype=np.float32)
                        weights.append(weight)
                    else:
                        f.read(binary_len)
    elif binary==False:
        with smart_open(binary_file, 'rb') as f:
            if words_file is None:
                header = to_unicode(f.readline())
                if len(header.split()) == 2: vocab_size, vector_size = map(int, header.split())
                elif len(header.split()) > 2:
                    parts = header.rstrip().split(" ")
                    word, weight = parts[0], list(map(np.float32, parts[1:]))
                    vc.append(to_unicode(word))
                    weights.append(weight)
                for _, line in enumerate(f):
                    parts = to_unicode(line.rstrip()).split(" ")
                    word, weight = parts[0], list(map(np.float32, parts[1:]))
                    vc.append(to_unicode(word))
                    weights.append(weight)
            else:
                header = to_unicode(f.readline())
                if len(header.split()) == 2: vocab_size, vector_size = map(int, header.split())
                elif len(header.split()) > 2:
                    parts = header.rstrip().split(" ")
                    word, weight = parts[0], list(map(np.float32, parts[1:]))
                    if word in words:
                        vc.append(to_unicode(word))
                        weights.append(weight)
                for _, line in enumerate(f):
                    parts = to_unicode(line.rstrip()).split(" ")
                    word, weight = parts[0], list(map(np.float32, parts[1:]))
                    if word in words:
                        vc.append(to_unicode(word))
                        weights.append(weight)
                    
    mat = np.array(weights, dtype=np.float32)
    
    return vc, mat

def save_file(outfile, vocab, mat, binary=True):
    assert len(vocab) == len(mat)
    with smart_open(outfile, 'wb') as f:
        f.write(to_utf8("%s %s\n" % mat.shape))
        for i in range(len(mat)):
            word = vocab[i]
            if binary:
                arr = mat[i]
                f.write(to_utf8(word) + b' ' + arr.tostring())
            else:
                f.write(to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in mat[i,:]))))

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

def index2word(vocab, index):
    word = ""
    for w, i in vocab.iteritems():
        if i == index: 
            word = w
            break
    if word == "": return None
    else: return word

def largest_eigenvalue(mat):
    D = np.dot(mat.T, mat)
    N = len(D)
    k = 1
    eigvalues = largest_eigh(D, eigvals=(N-k,N-1), eigvals_only = True)
    print 'Shape D: ' + str(D.shape)
    return eigvalues[0]
    