import sys, os
sys.path.append("../")
import pickle

def help():
    print "usage: python [file_path]"
    print
    print "  e.g: python ../exp/train/TFIDF+keyword/Ky/TFIDF+keyword.Ky.loved.train.npz"
    exit(-1)

if __name__ == '__main__':
	
    if len(sys.argv) != 2: help()
    pkl_file = pickle.load(open(sys.argv[1]))
    print pkl_file