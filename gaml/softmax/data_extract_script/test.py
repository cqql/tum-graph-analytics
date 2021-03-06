
# coding: utf-8

# In[1]:

from gensim.models import Doc2Vec
import gzip
import json
import numpy as np
import sklearn.datasets as svmlight
from tqdm import tqdm
import struct
import time as t


# In[2]:

def instance_generator(reviews_path, model_path):
    start = t.time()
    print "Loading model"
    model = Doc2Vec.load(model_path)
    print "Model loaded " + str(t.time() - start)
    with gzip.open(reviews_path, 'rt') as file:
        for index, line in enumerate(file):
            review = json.loads(line)
            yield model.infer_vector(review['reviewText'].split()), review['overall']


reviews_path = '/media/sf_SharedFolder/Work/reviews_Movies_and_TV.json.gz'
model_path = '/media/sf_SharedFolder/Work/reviews_Movies_and_TV_10_iter/reviews_Movies_and_TV_10_iter.model'

numData = 10 #4607047
trainFile = open("train.bin",'wb')
testFile = open("test.bin",'wb')
splitIndex = int(numData * 0.8)
counter = 0
trainTotal = 0
testTotal = 0

for data, label in tqdm(instance_generator(reviews_path, model_path), total=numData):
	indexes = data.nonzero()[0]
        values = data[indexes]
	
	if counter < splitIndex:
		trainFile.write(struct.pack("i", int(label)))
		for i in xrange(len(indexes)):
			trainFile.write(struct.pack("f", float(values[i])))
		trainTotal += 1
	else:
		testFile.write(struct.pack("i", int(label)))
		for i in xrange(len(indexes)):
			testFile.write(struct.pack("f", float(values[i])))
		testTotal += 1

	print values
	print label
	print "<<<<<<<<<<|||||>>>>>>>>>>"	


	counter += 1
	if counter == numData:
		break	
        


print "DONE!"
print "Split Index: " + str(splitIndex)
print "Final Counter: " + str(counter)
print "Train Total: " + str(trainTotal)
print "Test Total: " + str(testTotal)

