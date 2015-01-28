import sys, os
sys.path.append("../")
import numpy as np
from feelit import utils
import pickle
from feelit.features import dump
import scipy.io as sio

def gen_idx(y):
    """
    Parameters
    ==========
    y: 1-D np.array or list
        labels in training data
    
    Returns
    ======
    G: dict
        {
            <label>: [positive_ins, negative_ins],
            ...
        }
    """
    from collections import Counter
    import random

    dist = Counter(y)

    G = {}

    for label in dist:
        possitive_samples, negative_candidates = [], []
        for i, _label in enumerate(y):
            if _label == label:
                possitive_samples.append( (i, _label) )
            else:
                negative_candidates.append( (i, _label) )

        negative_samples = random.sample(negative_candidates, len(possitive_samples))

        G[label] = possitive_samples + negative_samples

    return G

def subsample(X, y, idxs):
    """
    subsample a 2-D array by row index
    """
    _X, _y = [], []
    for i in xrange(len(X)):
        if i in idxs:
            _y.append( y[i] )
            _X.append( X[i] )
    return ( np.array(_X), np.array(_y) )

def relabel(y, label): return [float(1) if _y == label else float(-1) for _y in y ]


def save(G, path="random_idx.pkl"): pickle.dump(G, open(path, "wb"), protocol=2)

def load(path="random_idx.pkl"): return pickle.load(open(path))
def usage():
    print 'Usage:'
    print 'python %s [feature_names]' % (__file__)
    print
    print 'e.g., feature_names: `image_rgba_phog`'
    exit(-1)

if __name__ == '__main__':
    
    # feature_name = "image_rgba_gist"
    if len(sys.argv) > 1:
        feature_names = sys.argv[1:]
    else:
        usage()

    ## generate
    # G = gen_idx(y)

    ## load existed
    print 'loading random_idx.pkl'
    G = load(path="random_idx.pkl")

    
    for feature_name in feature_names:

        print '>>> processing', feature_name

        ## load text_TFIDF.Xy.test.npz
        ## load text_TFIDF.Xy.train.npz
        npz_path = "../exp/data/from_mongo/"+feature_name+".Xy.train.npz"

        print ' > loading',npz_path

        data = np.load(npz_path)

        # slice to train/test

        X = data['X']
        if utils.isSparse(X):
            print 'the npz file you check is a sparse matrix'
            print ' > X to Dense'
            X = utils.toDense(X)
        print ' > get X', X.shape

        y = data['y']
        print ' > get y', y.shape

        for i_label, label in enumerate(G):

            print 'processing %d/%d' % ( i_label+1, len(G) )
            print ' > subsampling', label

            idxs = set([i for i,l in G[label] ])
            _X, _y = subsample(X, y, idxs)

            _y = relabel(_y, label)

            path = "../exp/train/"+feature_name+"/800p800n_Xy/"+feature_name+".800p800n_Xy."+label+".train"
            dirs = os.path.dirname(path+".npz")
            if dirs and not os.path.exists( dirs ): os.makedirs( dirs )
            
            print ' > dumping', path+".npz"
            dump(path+".npz", X=_X, y=_y)

            # sio.savemat(path+'.mat', {'X':_X, 'y':_y})
