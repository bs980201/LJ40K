import sys, os
sys.path.append("../")
from feelit.utils import random_idx
import pickle

train160_idx, dev160_idx = random_idx(160, 144)
pickle.dump(train160_idx, open('../exp/train/train160_binary_idx.pkl', 'w'))
pickle.dump(dev160_idx, open('../exp/train/dev160_binary_idx.pkl', 'w'))