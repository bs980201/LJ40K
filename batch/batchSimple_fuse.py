## fuse two features

import sys, os
sys.path.append("../")
import numpy as np
from feelit.features import Fusion

def help():
    print "usage: python [path_one][path_two][save_name]"
    print
    print "  e.g: python '../exp/data/from_mongo/TFIDF.Xy.train.npz' '../exp/data/from_mongo/keyword.Xy.train.npz' 'from_mongo/TFIDF+keyword_fromMongo.Xy.train'"
    print "  e.g: python '../exp/data/from_file/image_rgba_gist.Xy.train.npz' '../exp/data/from_file/image_rgba_phog.Xy.train.npz' 'from_file/rgba_gist+rgba_phog_fromfile.Xy.train'"
    exit(-1)

if __name__ == '__main__':
	
    if len(sys.argv) != 4: help()
    fu = Fusion(verbose=True)
    print "load files.."
    fu.loads(sys.argv[1], sys.argv[2])
    print "fusing..."
    fu.fuse()
    print "building the file"
    fu.dump(root="../exp/data", path=sys.argv[3] )
