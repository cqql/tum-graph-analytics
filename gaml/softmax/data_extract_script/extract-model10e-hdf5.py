from gensim.models import Doc2Vec
import gzip
import json
import numpy as np
from tqdm import tqdm
import h5py

def instance_generator(reviews_path, model_path):
    print "Loading model"
    model = Doc2Vec.load(model_path)
    print "Model loaded"
    with gzip.open(reviews_path, 'rt') as file:
        for index, line in enumerate(file):
            review = json.loads(line)
            yield model.infer_vector(review['reviewText'].split()), review['overall']


reviews_path = '/media/sf_SharedFolder/Work/reviews_Movies_and_TV.json.gz'
model_path = '/media/sf_SharedFolder/Work/reviews_Movies_and_TV_10_iter/reviews_Movies_and_TV_10_iter.model'

totalSize = 4607047
trainSize = int(totalSize * 0.8)
testSize = totalSize - trainSize
counter = 0
trainTotal = 0
testTotal = 0

trainFile = h5py.File("train.hdf5", "w", libver='latest')
trainSet = trainFile.create_dataset("train", (trainSize, 301), dtype='f')

testFile = h5py.File("test.hdf5", "w", libver='latest')
testSet = testFile.create_dataset("test", (testSize, 301), dtype='f')

for data, label in tqdm(instance_generator(reviews_path, model_path), total=totalSize):
	item = np.append(label, data)
	
	if trainTotal < trainSize:
		trainSet[trainTotal] = item
		trainTotal += 1
	elif testTotal < testSize:
		testSet[testTotal] = item
		testTotal += 1
	else:
		print "Overflow!!!"
	
	counter += 1        

print "Flushing and closing files."

trainFile.flush()
trainFile.close()
testFile.flush()
testFile.close()

print "DONE!"
print "Initial total: " + str(totalSize)
print "Final Counter: " + str(counter)
print "Train Total: " + str(trainTotal)
print "Test Total: " + str(testTotal)
