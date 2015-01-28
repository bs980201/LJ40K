import sys, os
sys.path.append("../")
import pickle

def help():
    print "usage: python [file_path]"
    print
    print "  e.g: python random80_idx.pkl"
    exit(-1)

if __name__ == '__main__':
	
    if len(sys.argv) != 2: help()
    pkl_file = pickle.load(open(sys.argv[1]))
    print pkl_file
    print "type of pkl_file:", type(pkl_file)
    print "len of pkl_file:", len(pkl_file)
    print