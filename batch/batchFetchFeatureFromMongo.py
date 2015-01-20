## Fetch features from mongo and make npz file

import sys, os
sys.path.append("../")
import numpy as np
from feelit.features import FetchMongo

def help():
    print "usage: python [feature_name][settingID][data_range][save_path]"
    print
    print "  e.g: python TFIDF 53a1921a3681df411cdf9f38 800 ../exp/data/from_mongo/TFIDF.Xy.train"
    exit(-1)

if __name__ == '__main__':
	
    if len(sys.argv) != 5: help()
    fm = FetchMongo(verbose=True)
    sys.argv[3] = int(sys.argv[3])

    try:
        fm.fetch_transform(sys.argv[1], sys.argv[2], data_range=sys.argv[3])    
    except:
        print "Failed to fetch file features.%s in Mongo" % (sys.argv[1])
        exit(-1)

    try:
        fm.dump(path=sys.argv[4], ext=".npz")
    except:
        print "Failed to make .npz file"
        exit(-1)


