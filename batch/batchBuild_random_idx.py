# -*- coding: utf-8 -*-
import sys, os
import random
import pickle

LJ40K = ['accomplished', 'aggravated', 'amused', 'annoyed', 'anxious', 'awake', 'blah', 'blank', 'bored', 'bouncy', 'busy', 'calm', 'cheerful', 'chipper', 'cold', 'confused', 'contemplative', 'content', 'crappy', 'crazy', 'creative', 'crushed', 'depressed', 'drained', 'ecstatic', 'excited', 'exhausted', 'frustrated', 'good', 'happy', 'hopeful', 'hungry', 'lonely', 'loved', 'okay', 'pissed off', 'sad', 'sick', 'sleepy', 'tired']

def help():
    print
    print "usage: python %s [Data_quantity][Selected_EachEm_Data_quantity][train/test]"  % (__file__)
    print
    print "-----------------------------------------------------------------------------------"
    print "  e.g: python %s 32000 80 train" % (__file__)
    print "  from 32000 training data randomly select 80*2 data each emotion for training"
    print "  e.g: output pkl file: {'accomplished':[3,56,34,78...],'sad':[466,536,423,...],.....}"
    print
    print "-----------------------------------------------------------------------------------"
    print "  e.g: python %s 8000 20 test" % (__file__)
    print "  from 8000 testing data randomly select 20 data each emotion for testing"
    print
    exit(-1)

def random_thislabel_Data(Data_quantity,EachEm_Data_quantity,Selected_EachEm_Data_quantity):  
    selected_Data = []
    #build[0,1,2,....31999]
    feature_indexs = range(Data_quantity)
    #build[0,800,1600,....31200] and make feature_indexs separated 
    for i in range(0,Data_quantity,EachEm_Data_quantity):
        Part_of_feature_index = feature_indexs[i:i+EachEm_Data_quantity]
        random.shuffle(Part_of_feature_index)
        selected_Data.append(Part_of_feature_index[:Selected_EachEm_Data_quantity])
    return selected_Data

def random_Nthislabel_Data(Data_quantity,EachEm_Data_quantity,Selected_EachEm_Data_quantity):    
    selected_Data = []
    for i in range(0,Data_quantity,EachEm_Data_quantity):
        feature_indexs = range(Data_quantity)
        del feature_indexs[i:i+EachEm_Data_quantity]
        random.shuffle(feature_indexs)
        feature_indexs = feature_indexs[:Selected_EachEm_Data_quantity]
        selected_Data.append(feature_indexs)
    # print selected_Data
    return selected_Data

def random_Data(Data_quantity,Emotion_quantity,Selected_EachEm_Data_quantity):     
    selected_Data = []
    #build[0,200,400,....7800] and make feature_indexs separated      
    for i in range(Emotion_quantity):
        #build[0,1,2,....7999]
        feature_indexs = range(Data_quantity)
        random.shuffle(feature_indexs)
        selected_Data.append(feature_indexs[:Selected_EachEm_Data_quantity])
    # print random_thislabel_Data
    return selected_Data

if __name__ == '__main__':

    if len(sys.argv) != 4: help()
    Emotion_quantity = 40
    Data_quantity = int(sys.argv[1])
    #EachEm_Data_quantity = 32000/40 = 800
    EachEm_Data_quantity = Data_quantity/Emotion_quantity
    # e.g: Selected_EachEm_Data_quantity = 80
    Selected_EachEm_Data_quantity = int(sys.argv[2])
    
    if sys.argv[3] == 'train':
        SelectPart1 = random_thislabel_Data(Data_quantity,EachEm_Data_quantity,Selected_EachEm_Data_quantity)
        SelectPart2 = random_Nthislabel_Data(Data_quantity,EachEm_Data_quantity,Selected_EachEm_Data_quantity)

        Total_selecting_Data = [x+y for x, y in zip(SelectPart1, SelectPart2)]
        Total_SETQ = str(Selected_EachEm_Data_quantity*2)
        random_idx = dict(zip(LJ40K,Total_selecting_Data))
        pickle.dump(random_idx, open("random"+Total_SETQ+"Train_idx.pkl", "wb"), protocol=2)
    
    elif sys.argv[3] == 'test':
        Total_selecting_Data = random_Data(Data_quantity,Emotion_quantity,Selected_EachEm_Data_quantity)
        Total_SETQ = str(Selected_EachEm_Data_quantity)
        random_idx = dict(zip(LJ40K,Total_selecting_Data))
        pickle.dump(random_idx, open("random"+Total_SETQ+"Test_idx.pkl", "wb"), protocol=2)
    
    else:
        help()
