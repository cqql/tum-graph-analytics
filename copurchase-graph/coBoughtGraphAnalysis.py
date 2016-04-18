import gzip
import itertools
from igraph import *
from scipy import sparse, io
import numpy as np
from tqdm import tqdm

def parseAmazonMetadata(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def getLimitedNumberPairCounts(itemCountLimit, parsedMetadata):
    relatedKey = 'related'
    alsoBoughtKey = 'also_bought'
    pairCounts = dict()
    
    for item in tqdm(parsedMetadata, total=itemCountLimit):
        if relatedKey in item:
            relatedData = item[relatedKey]

            if alsoBoughtKey in relatedData:
                alsoBoughtUsers = relatedData[alsoBoughtKey]
                allPairs = itertools.combinations(alsoBoughtUsers, 2)

                for pair in allPairs:
                    sortedPair = list(pair)
                    sortedPair.sort()
                    pairKey = tuple(sortedPair)

                    if pairKey in pairCounts:
                        pairCounts[pairKey] = pairCounts[pairKey] + 1
                    else:
                        pairCounts[pairKey] = 1

                    if len(pairCounts) >= itemCountLimit:
                        return pairCounts

def get_sparse_adjacency_matrix(G, attr=None):
    if attr:
        source, target, data = zip(*[(e.source, e.target, e[attr]) 
            for e in G.es if not np.isnan(e[attr])]);
    else:
        source, target = zip(*[(e.source, e.target)
           for e in G.es]);
        data = np.ones(len(source)).astype('int').tolist();
    if not G.is_directed():
        # If not directed, also create the other edge
        source, target = source + target, target + source;
        data = data + data;
    L = sparse.coo_matrix((data, (source, target)), shape=[G.vcount(), G.vcount()]);
    return L;

def saveGraphAsSparceMatrix(graph, fileName):
    sparseMat = get_sparse_adjacency_matrix(graph)
    io.savemat( str(fileName) + '.mat', dict(sparseMat=sparseMat))

parsedMetadata = parseAmazonMetadata('meta_Video_Games.json.gz')

#NOTE:For a fast round of iterations the progress monitor does not work correctly
pairCounts = getLimitedNumberPairCounts(10000, parsedMetadata)

coBoughtGraph = Graph()
vertices = list()

for key, value in tqdm(pairCounts.iteritems(), total=len(pairCounts)):
    
    for user in key:
        if not user in vertices:
            coBoughtGraph.add_vertex(user)
            vertices.append(user)
    
    coBoughtGraph.add_edge(key[0], key[1], weight=value )

#link to different layouts: http://igraph.org/python/doc/tutorial/tutorial.html#layouts-and-plotting
# kk (Kamada-Kawai) and fr (Fruchterman-Reingold) seem to be the most informative

visual_style = {}
visual_style["layout"] = coBoughtGraph.layout("fr")
visual_style["vertex_size"] = 5
#visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
#visual_style["vertex_label"] = coBoughtGraph.vs["name"]
visual_style["edge_width"] = [int(val) for val in coBoughtGraph.es["weight"]]
#visual_style["bbox"] = (300, 300)
#visual_style["margin"] = 20

plot(coBoughtGraph, 'coBoughtGraph.png', **visual_style)

saveGraphAsSparceMatrix(coBoughtGraph, 'coBoughtGraphSparseMatrix')

