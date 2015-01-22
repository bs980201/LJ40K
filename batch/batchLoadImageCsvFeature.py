# -*- coding: utf-8 -*-
import sys, os
sys.path.append("../")
import logging, pickle
from feelit import utils
from feelit.features import LoadFile
import numpy as np


def help():
    print "usage: python [load_path][feature_numbers][save_path]"
    print
    print " e.g.: python ../images/data/features/csv/rgba/gist 800 ../exp/data/from_file/image_rgba_gist.Xy"
    print " e.g.: python ../images/data/features/csv/rgba/phog 800 ../exp/data/from_file/image_rgba_phog.Xy"
    exit(-1)

if __name__ == '__main__':

    if len(sys.argv) != 4: help()
    lf = LoadFile(verbose=True)
    sys.argv[2] = int(sys.argv[2])
    lf.loads(root=sys.argv[1], data_range=(None,sys.argv[2]), amend=True)
    lf.dump(path=sys.argv[3], ext=".npz")
