
# coding: utf-8

# In[1]:

from gensim.models import Doc2Vec
import gzip
import json
import numpy as np
import sklearn.datasets as svmlight
from tqdm import tqdm


# In[2]:

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

trainFile = open("train10_movTV.libsvm",'w')
testFile = open("test10_movTV.libsvm",'w')
splitIndex = int(4607047 * 0.8)
counter = 0

for data, label in tqdm(instance_generator(reviews_path, model_path), total=4607047):
	indexes = data.nonzero()[0]
        values = data[indexes]

	lbl = '%i'%label
        pairs = ['%i:%f'%(indexes[i]+1,values[i]) for i in xrange(len(indexes))]

        sep_line = [lbl]
        sep_line.extend(pairs)
        sep_line.append('\n')

        line = ' '.join(sep_line)
	
	if counter < splitIndex:
		trainFile.write(line)
	else:
		testFile.write(line)
	
	counter += 1
        


print "DONE!"

