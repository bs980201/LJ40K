## from mongodb to building a kernel

import sys, os
sys.path.append("../")
import numpy as np
from feelit.features import FetchMongo
from feelit.features import dump
from feelit.features import Learning
from feelit import utils
from feelit.kernel import RBF
import pickle
emotions = utils.LJ40K

def help():
    print "usage: python %s [Feature_name][SettingID][TrainingData_range][.mat or not, 1 or 0]" % (__file__)
    print
    print "  e.g: python %s keyword_emotion 538a08a5d4388c142389a032 800 1" % (__file__)
    print "       '1' represent that making another type of files: .mat"    
    exit(-1)

def fetch(feature_name, settingID, data_range):
    fm_tr = FetchMongo(verbose=True)
    fm_te = FetchMongo(verbose=True)
    try:
        fm_tr.fetch_transform(feature_name, settingID, data_range=data_range)
        fm_te.fetch_transform(feature_name, settingID, data_range=">"+str(data_range))      
    except:
        print "Failed to fetch file features.%s in Mongo" % (feature_name)
        exit(-1)
    try:
        fm_tr.dump(path="../exp/data/from_mongo/"+feature_name+".Xy.train", ext=".npz")
        fm_te.dump(path="../exp/data/from_mongo/"+feature_name+".Xy.test", ext=".npz")
    except:
        print "Failed to make .npz file"
        exit(-1)

def Npzto40Emo(feature_name, mat=0):
    print 'loading random160_idx'
    G = load(path="random160_idx.pkl")

    print '>>> processing', feature_name

    ## load text_TFIDF.Xy.test.npz
    ## load text_TFIDF.Xy.train.npz
    npz_path = "../exp/data/from_file/"+feature_name+".Xy.train.npz"

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

        idxs = set([i for i in G[label] ])
        _X, _y = subsample(X, y, idxs)

        _y = relabel(_y, label)

        path = "../exp/train/"+feature_name+"/160_Xy/"+feature_name+".Xy."+label+".train"
        dirs = os.path.dirname(path+".npz")
        if dirs and not os.path.exists( dirs ): os.makedirs( dirs )
        
        print ' > dumping', path+".npz"
        dump(path+".npz", X=_X, y=_y)

        if mat == 1:
            import scipy.io as sio
            sio.savemat(path+'.mat', {'X':_X, 'y':_y})

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

def relabel(y, label): return [1 if _y == label else -1 for _y in y ]

def save(G, path="random160_idx.pkl"): pickle.dump(G, open(path, "wb"), protocol=2)

def load(path="random160_idx.pkl"): return pickle.load(open(path))

def usage():
    print 'Usage:'
    print 'python %s [feature_names]' % (__file__)
    print
    print 'e.g., feature_names: `image_rgba_phog`'
    exit(-1)

def evals(y_te, y_predict, emotion):

    y_te_ = [1 if a == emotion else 0 for a in y_te]
    y_predict_ = [0 if a.startswith('_') else 1 for a in y_predict]

    Y = zip(y_te_, y_predict_)

    FP = len([ 1 for a,b in Y if a == 0 and b == 1 ])
    TN = len([ 1 for a,b in Y if a == 0 and b == 0 ])
    TP = len([ 1 for a,b in Y if a == 1 and b == 1 ])
    FN = len([ 1 for a,b in Y if a == 1 and b == 0 ])
    accu = (TP + TN/39) / float(FP/39 + TN/39 + TP + FN)

    return accu

def training(feature,eid):
    ## init
    emotion = emotions[int(eid)]
    l = Learning(verbose=False)

    ## ================ training ================ ##

    ## load train
    l.load(path="../exp/train/%s/160_Xy/%s.Xy.%s.train.npz" % (feature, feature, emotion))

    ## train
    l.train(classifier="SVM", kernel="rbf", prob=False)

    ## ================= testing ================= ##

    ## load test data
    test_data = np.load('../exp/data/from_mongo/%s.Xy.test.npz' % (feature))
    # y_te
    # array(['accomplished', 'accomplished', 'accomplished', ..., 'tired',
    #        'tired', 'tired'],
    #       dtype='|S13')
    X_te, y_te = test_data['X'], test_data['y']

    ## predict
    # y_predict
    # array([u'_sad', u'_sad', u'sad', ..., u'_sad', u'_sad', u'_sad'],
    #       dtype='<U4')
    y_predict = l.clf.predict(X_te)

    ## eval
    accuracy = evals(y_te, y_predict, emotion)

    print emotion, '\t', accuracy

def buildkernel(root,feature,begin,end):

    in_subdir  = "%s/160_Xy" % (feature)
    out_subdir = "%s/160_Ky" % (feature)

    in_dir = os.path.join(root, in_subdir)
    out_dir = os.path.join(root, out_subdir)

    npzs = filter(lambda x:x.endswith('.npz'), os.listdir(in_dir))
    to_process = sorted(npzs)[begin:end]

    print 'files to be processed:'
    print '\n'.join(['\t'+x for x in to_process])
    print '='*50

    ## load train/dev index files
    ## train_idx: a list containing 1440 index (int)
    ## dev_idx: a list containing 160 index (int)
    '''
    # required work:
    # from feelit.utils import random_idx
    # train_idx, dev_idx = random_idx(160, 140)
    # import pickle
    # pickle.dump(train_idx, open('train160_binary_idx.pkl', 'w'))
    # pickle.dump(dev_idx, open('dev160_binary_idx.pkl', 'w'))
    '''
    try:
        train160_idx, dev160_idx = pickle.load(open('../exp/train/train160_binary_idx.pkl')), pickle.load(open('../exp/train/dev160_binary_idx.pkl'))
    except:
        help()

    for npz_fn in to_process:

        ## npz_fn: rgba_gist+rgba_phog.Xy.happy.train.npz
        print 'processing', npz_fn

        rbf = RBF(verbose=True)
        rbf.load(os.path.join(in_dir, npz_fn))

        ## devide X,y into (X_train, y_train) and (X_dev, y_dev)
        # get dev by deleting the indexes of train
        X_dev, y_dev = utils.RandomSample((rbf.X, rbf.y), delete_index=train160_idx)
        # get train by deleting the indexes of dev
        X_tr, y_tr = utils.RandomSample((rbf.X, rbf.y), delete_index=dev160_idx)

        ## build
        K_tr, K_dev = rbf.build( (X_tr, X_tr), (X_tr, X_dev) )

        ## save
        ## rgba_gist+rgba_phog.Xy.happy.train.npz
        feature, xy, emotion, dtype, ext = npz_fn.split('.')
        out_train_fn = "%s.160_Ky.%s.train.npz" % (feature, emotion)
        out_dev_fn = "%s.160_Ky.%s.dev.npz" % (feature, emotion)
        
        rbf.save(os.path.join(out_dir, out_train_fn), K_tr=K_tr,   y_tr=y_tr   )
        rbf.save(os.path.join(out_dir, out_dev_fn),   K_dev=K_dev, y_dev=y_dev ) 


if __name__ == '__main__':
    
    if len(sys.argv) != 5: help()
    sys.argv[3] = int(sys.argv[3])

    # #fetch files from Mongodb
    # fetch(sys.argv[1],sys.argv[2],sys.argv[3])
    
    #make above files into 40 npz files
    Npzto40Emo(sys.argv[1], mat=int(sys.argv[4]))
    
    # eid = input("input the emotion number you want to train:(0~39)")
    # training(sys.argv[1],eid)
    
    #make sure that you've already use Build_random_Tr+Dev_idx.py to built '../exp/data/train_binary_idx.pkl' and '../exp/data/dev_binary_idx.pkl'
    # buildkernel('../exp/train/',sys.argv[1],0,40)

