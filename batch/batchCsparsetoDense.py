import logging, os, sys, pickle
sys.path.append("../")
import numpy as np
from feelit.features import dump
import scipy.io as sio
from feelit import utils


if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        feature_names = sys.argv[1:]
    else:
        print 'usage: python %s <feature_names>' 
        exit(-1)

    for feature_name in feature_names:

        ## load
        npz_path = "../exp/data/from_mongo/"+feature_name+".Xy.test.npz"
        print 'loading from',npz_path
        data = np.load(npz_path)

        X = data['X']
        if utils.isSparse(X):
            print 'the npz file you check is a sparse matrix'
            print ' > X to Dense'
            X = utils.toDense(X)
        print 'get X', X.shape

        y = data['y']
        print 'get y', y.shape
        
        test_path = "../exp/data/from_mongo/"+feature_name+".sp.Xy.test"
        
        sio.savemat(test_path+'.mat', {'X':X, 'y':y})

        # print ' > dumping testing to', test_path
        # dump(test_path, X=X, y=y)