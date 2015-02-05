## from mongodb to building a kernel

import sys, os
sys.path.append("../")
import numpy as np
import logging, pickle
from feelit.features import FetchMongo
from feelit.features import dump
from feelit.features import Learning
from feelit.features import LoadFile
from feelit import utils
from feelit.kernel import RBF

emotions = utils.LJ40K

def help():
    print "usage: python %s [Feature_name]" % (__file__)
    print
    print "  e.g: python %s keyword" % (__file__)
    print "     : python %s image_rgba_gist" % (__file__)
    print 
    print " Step: FetchToFile --> FileTo40Emo --> Training --> Buildkernel"
    print
    exit(-1)

def fetch(feature_name, settingID, data_range):
    if data_range != "all":
        data_range = int(data_range)
    fm_tr = FetchMongo(verbose=True)
    # fm_te = FetchMongo(verbose=True)
    try:
        fm_tr.fetch_transform(feature_name, settingID, data_range=data_range)
        # fm_te.fetch_transform(feature_name, settingID, data_range=">"+str(data_range))      
    except:
        print "Failed to fetch file features.%s in Mongo" % (feature_name)
        exit(-1)
    try:
        dirs = os.path.dirname("../exp/data/from_mongo/")
        if dirs and not os.path.exists( dirs ): os.makedirs( dirs )
        fm_tr.dump(path="../exp/data/from_mongo/"+feature_name+".Xy", ext=".npz")
        # fm_te.dump(path="../exp/data/from_mongo/"+feature_name+".Xy.test", ext=".npz")
    except:
        print "Failed to make .npz file"
        exit(-1)

def loadimagefile(feature_name,load_path,data_range):
    if data_range != "all":
        data_range = int(data_range)
        lf_tr = LoadFile(verbose=True)
        lf_te = LoadFile(verbose=True)
        lf_tr.loads(root=load_path, data_range=(None,data_range), amend=True)
        EachEmDataQuanty = 1000
        testdataQuanty = EachEmDataQuanty-data_range
        lf_te.loads(root=load_path, data_range=(-testdataQuanty,None), amend=True)
        lf_tr.dump(path="../exp/data/from_file/"+feature_name+".Xy.train", ext=".npz")
        lf_te.dump(path="../exp/data/from_file/"+feature_name+".Xy.test", ext=".npz")
    if data_range == "all":
        lf_trall = LoadFile(verbose=True)
        lf_trall.loads(root=load_path, amend=True)
        lf_trall.dump(path="../exp/data/from_file/"+feature_name+".Xy", ext=".npz")

def Npzto40Emo(feature_name, datafunction, RandomSetting_Path, mat=0, feature_type="default"):
    # for training
    print 'loading'+RandomSetting_Path

    G = load(path=RandomSetting_Path)
    DataFormat = RandomSetting_Path.split('/')[-1].replace('random','').replace("_idx.pkl","").replace(datafunction,"")

    # #for testing
    # print 'loading random200p200nTest_idx.pkl'
    # G = load(path="random200p200nTest_idx.pkl")
 
    print '>>> processing', feature_name

    ## load text_TFIDF.Xy.test.npz
    ## load text_TFIDF.Xy.train.npz
    if feature_type == "text":
        npz_path = "../exp/data/from_mongo/"+feature_name+".Xy."+datafunction+".npz"
    elif feature_type == "image":
        npz_path = "../exp/data/from_file/"+feature_name+".Xy."+datafunction+".npz"
    else:
        help()

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

        path = "../exp/"+datafunction+"/"+feature_name+"/"+DataFormat+"_Xy/"+feature_name+"."+DataFormat+"_Xy."+label+"."+datafunction
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

def relabel(y, label): return [float(1) if _y == label else float(-1) for _y in y ]

def save(G, path="default"): pickle.dump(G, open(path, "wb"), protocol=2)

def load(path="default"): return pickle.load(open(path))

def evals(y_te, y_predict, emotion,te_format):
    
    # ## for y is 1, -1 format
    # y_te_ = [1 if a == 1 else 0 for a in y_te]
    # y_predict_ = [0 if a == -1 else 1 for a in y_predict]

    # for y is sad, _sad format
    # y_te_ = [1 if a == emotion else 0 for a in y_te]
    # y_predict_ = [0 if a.startswith('_') else 1 for a in y_predict]
    
    if te_format == 'all':
        y_predict_ = [0 if a == -1 else 1 for a in y_predict]
        y_te_ = [1 if a == emotion else 0 for a in y_te]
    else:
        y_te_ = [1 if a == 1 else 0 for a in y_te]
        y_predict_ = [0 if a == -1 else 1 for a in y_predict]

    Y = zip(y_te_, y_predict_)

    FP = len([ 1 for a,b in Y if a == 0 and b == 1 ])
    TN = len([ 1 for a,b in Y if a == 0 and b == 0 ])
    TP = len([ 1 for a,b in Y if a == 1 and b == 1 ])
    FN = len([ 1 for a,b in Y if a == 1 and b == 0 ])
    

    if te_format == 'all':    
        accu = (TP + TN/39) / float(FP/39 + TN/39 + TP + FN)
    else:
        # for "not 8000 testing data"
        accu = (TP + TN) / float(FP + TN + TP + FN)
    
    return accu

def training(feature,begin,end,tr_format,te_format):
    ## init
    Pofemotion = emotions[begin:end]
    nfold = raw_input("nfold?:(y/n)")

    if te_format == 'all':
        ## y = u'_sad', u'sad'
        print 'loading testing data from'+"../exp/test/%s/%s.Xy.test.npz" % (feature, feature)
        test_data = np.load("../exp/test/%s/%s.Xy.test.npz" % (feature, feature))
    else:pass

    for emotion in Pofemotion:

        l = Learning(verbose=False)

        ## ================ training ================ ##
        topC = 1.0
        # load train
        print 'loading training data from'+"../exp/train/%s/%s_Xy/%s.%s_Xy.%s.train.npz" % (feature, tr_format, feature, tr_format, emotion)
        l.load(path="../exp/train/%s/%s_Xy/%s.%s_Xy.%s.train.npz" % (feature, tr_format, feature, tr_format, emotion))     
        
        if nfold == 'y':
            print '>> training n_folds',emotion
            topC = l.kFold(n_folds=10)
        else: print '>> NOT to training n_folds',emotion
        # train
        print '>> training, using topC',emotion
        l.train(classifier="SVM", kernel="rbf", C=topC, prob=False)

        # # ## ================= testing ================= ##
        ## load test data
        if te_format != 'all':
            print 'loading testing data from'+"../exp/test/%s/%s_Xy/%s.%s_Xy.%s.test.npz" % (feature, te_format, feature, te_format, emotion)        
            test_data = np.load("../exp/test/%s/%s_Xy/%s.%s_Xy.%s.test.npz" % (feature, te_format, feature, te_format, emotion))
        else:pass

        X_te, y_te = test_data['X'], test_data['y']
        if utils.isSparse(X_te):
            print 'The testing data is a sparse matrix'
            print ' > X_te to Dense'
            X_te = utils.toDense(X_te)

        y_predict = l.clf.predict(X_te)

        # #print 'y_predict = ', y_predict

        ## eval
        accuracy = evals(y_te, y_predict, emotion,te_format)
        # accuracy = l.clf.score(X_te, y_te)
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
    
    if len(sys.argv) != 2: help()


    Fetchdata_ornot = raw_input("Fetch data?:(y/n)")    
    if Fetchdata_ornot == 'y':
        print "Input required info:[Loading_path(SettingID/file_path)][TrainingData_range(all/number)][Fetchfrom(mongo/file)]"
        print "for text, load data form mongodb","e.g.: 537451d1d4388c7843516ba4 all mongo"
        print "for image, load csv data from file path","e.g.: ../images/data/features/csv/rgba/gist 800 file"        
        while True:
            Pre_Fetchdata = raw_input("Input:")
            if Pre_Fetchdata == 'ex':break
            Pre_Fetchdata = Pre_Fetchdata.split()
            if len(Pre_Fetchdata) == 3:
                if Pre_Fetchdata[2] == "mongo":
                    #fetch data from Mongodb
                    fetch(sys.argv[1],Pre_Fetchdata[0],Pre_Fetchdata[1])
                    break
                elif Pre_Fetchdata[2] == "file":
                    #load data from file path
                    loadimagefile(sys.argv[1],Pre_Fetchdata[0],Pre_Fetchdata[1])
                    break
                else: print "your input format is wrong, try again, or type ex to out"
            else: print "your input format is wrong, try again, or type ex to out"
    else: pass
    

    Into40emtion_ornot = raw_input("Make training or testing data into 40 emotion:(y/n)")    
    if Into40emtion_ornot == 'y':
        print "Input required info:[make .mat or not?(1/0)][data_is(text/image)][train/test][RandomSetting_Path]"
        print "e.g.:1 text test random400test_idx.pkl"
        print "e.g.:1 image train random160_idx.pkl"
        while True:
            Pre_Into40emtion = raw_input("Input:")
            if Pre_Into40emtion == 'ex':break
            Pre_Into40emtion = Pre_Into40emtion.split() 
            if (len(Pre_Into40emtion) == 4) and ((Pre_Into40emtion[2] == 'test') or (Pre_Into40emtion[2] == 'train')):
                #make above files into 40 npz files
                Npzto40Emo(sys.argv[1], Pre_Into40emtion[2], Pre_Into40emtion[3], mat=int(Pre_Into40emtion[0]), feature_type=Pre_Into40emtion[1])
                break
            else: print "your input format is wrong, try again, or type ex to out"
    else: pass
    

    Training_ornot = raw_input("Training?:(y/n)")    
    if Training_ornot == 'y': 
        # eid = input("input the emotion ID you want to train:(0~39)")
        print "Input required info:[Begin emotion ID][End emotion ID][TrainingDataFormat][TestingDataFormat]" 
        print "e.g.:0 40 160 20p20n"
        while True:
            Pre_Training = raw_input("Input:")
            if Pre_Training == 'ex':break
            Pre_Training = Pre_Training.split()
            if len(Pre_Training) == 4:
                training(sys.argv[1],int(Pre_Training[0]),int(Pre_Training[1]),Pre_Training[2],Pre_Training[3])
                break
            else: print "your input format is wrong, try again, or type ex to out"
    else: pass
    

    buildkernel_ornot = raw_input("Build a training kernel?:(y/n)")    
    if buildkernel_ornot == 'y':
        print "Input required info:[Begin emotion ID][End emotion ID]" 
        print "e.g.:0 40"
        Pre_buildkernel = raw_input("Input:")
        Pre_buildkernel = Pre_buildkernel.split() 
        if len(Pre_buildkernel) == 2:
            #make sure that you've already use Build_random_Tr+Dev_idx.py to built '../exp/data/train_binary_idx.pkl' and '../exp/data/dev_binary_idx.pkl'
            buildkernel('../exp/train/',sys.argv[1],int(Pre_buildkernel[0]),int(Pre_buildkernel[1]))
        else: print "your input format is wrong, try next time"
    else: pass
