# -*- coding: UTF-8 -*-
import sys, os
sys.path.append("../")
from feelit import utils
from sklearn.decomposition import TruncatedSVD
import numpy as np
import time
from threading import Thread


def sparseTodense(X):
    if utils.isSparse(X):
        print 'The data is a sparse matrix'
        print '>>>data to Dense'
        X = utils.toDense(X)
    return X

def TSVD(i,n_components):
    print(">>>thread %d start! with %d components" % (i,n_components))

    print(">>>thread %d, %d components, loading training and testing files" % (i,n_components))
    tr_data = np.load('../exp/data/from_mongo/pattern_mincount2.Xy.train.npz')
    trX = tr_data['X']
    tr_y = tr_data['y']
    te_data = np.load('../exp/data/from_mongo/pattern_mincount2.Xy.test.npz')
    teX = te_data['X']
    te_y = te_data['y']

    trX = sparseTodense(trX)
    teX = sparseTodense(teX)

    print '>>>thread %d, %d components, fit' % (i,n_components)
    svd = TruncatedSVD(n_components=n_components).fit(trX)
    print '>>>thread %d, %d components, transform training data' % (i,n_components)
    trX = svd.transform(trX)
    print '>>>thread %d, %d components, transform testing data' % (i,n_components)
    teX = svd.transform(teX)

    print 'thread %d, %d components, svd.explained_variance_ : ' % (i,n_components)
    print svd.explained_variance_[0:5]
    print 'thread %d, %d components, svd.explained_variance_ratio_ : ' % (i,n_components)
    print svd.explained_variance_ratio_[0:5]

    path = '../exp/data/from_mongo/pattern_mincount2_TSVD'+str(n_components)

    np.savez_compressed(path+'.Xy.train.npz', X=trX, y=tr_y)
    np.savez_compressed(path+'.Xy.test.npz', X=teX, y=te_y)
    print '>>>thread %d, %d components, FINISH' % (i,n_components)

def main():
    n_components = [150, 300, 500]
    for i,j in enumerate(n_components):
        t = Thread(target=TSVD, args=(i,j))
        time.sleep(10)
        t.start()
 
if __name__ == '__main__':
    main()
