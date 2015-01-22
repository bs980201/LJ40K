import random
import pickle

LJ40K = ['accomplished', 'aggravated', 'amused', 'annoyed', 'anxious', 'awake', 'blah', 'blank', 'bored', 'bouncy', 'busy', 'calm', 'cheerful', 'chipper', 'cold', 'confused', 'contemplative', 'content', 'crappy', 'crazy', 'creative', 'crushed', 'depressed', 'drained', 'ecstatic', 'excited', 'exhausted', 'frustrated', 'good', 'happy', 'hopeful', 'hungry', 'lonely', 'loved', 'okay', 'pissed off', 'sad', 'sick', 'sleepy', 'tired']

#build[0,1,2,....32000]
TrainingData_quantity = 32000
feature_indexs = range(TrainingData_quantity)
#build[0,800,1600,....32000] and make feature_indexs separated
random_thislabel_trainingData = []
for i in range(0,TrainingData_quantity,800):
    Part_of_feature_index = feature_indexs[i:i+800]
    random.shuffle(Part_of_feature_index)
    random_thislabel_trainingData.append(Part_of_feature_index[:80])
# print random_thislabel_trainingData
random_Nthislabel_trainingData = []
for i in range(0,TrainingData_quantity,800):
    TrainData = range(TrainingData_quantity)
    del TrainData[i:i+800]
    random.shuffle(TrainData)
    TrainData = TrainData[:80]
    random_Nthislabel_trainingData.append(TrainData)
# print random_Nthislabel_trainingData
Total_selecting_trainingData = [x+y for x, y in zip(random_thislabel_trainingData, random_Nthislabel_trainingData)]
# print Total_selecting_trainingData

random160_idx = dict(zip(LJ40K,Total_selecting_trainingData))
pickle.dump(random160_idx, open("random16_idx.pkl", "wb"), protocol=2)
