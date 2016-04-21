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


def get_sparse_adjacency_matrix(graph, attr=None):    
    if attr:
        source, target, data = zip(*[(e.source, e.target, e[attr]) 
            for e in graph.es if not np.isnan(e[attr])]);
    else:
        source, target = zip(*[(e.source, e.target)
           for e in graph.es]);
        data = np.ones(len(source)).astype('int').tolist();
    if not graph.is_directed():
        # If not directed, also create the other edge
        source, target = source + target, target + source;
        data = data + data;
    L = sparse.coo_matrix((data, (source, target)), shape=[graph.vcount(), graph.vcount()]);
    return L.tocsr();


def saveGraphAsSparceMatrix(graph, fileName):
    sparseMat = get_sparse_adjacency_matrix(graph)
    io.savemat( str(fileName) + '.mat', dict(sparseMat=sparseMat))


#http://blog.samuelmh.com/2015/02/pagerank-sparse-matrices-python-ipython.html
#http://michaelnielsen.org/blog/using-your-laptop-to-compute-pagerank-for-millions-of-webpages/
def compute_PageRank(matrix, beta=0.85, epsilon=10**-4):
    '''
    Parameters
    ----------
    G : sparse adjacency matrix, which shows the connections between nodes.
    beta: 1-teleportation probability. The teleportation probability is the chance the we move from one node to another one completely at random.
    epsilon: stop condition used to stop computation when it starts converging

    Returns
    -------
    output : PageRank in a numpy array

    '''
    #Getting the dimensions of the n-by-n matrix
    n = matrix.shape[0]
    #Sum the matrix along the rows to get the degree for each of the nodes
    #and divide by the 1 - teleportation
    deg_out_beta = matrix.sum(axis=0).T/beta #vector
    #Initialize the ranks for each node to be 1/n, where n is the number of nodes
    ranks = np.ones((n,1))/n #vector
    flag = True
    while flag:
        new_ranks = matrix.dot((ranks/deg_out_beta)) #vector
        #Last calculated PageRank
        new_ranks += (1-new_ranks.sum())/n
        #Check if computation has converged
        if np.linalg.norm(ranks-new_ranks,ord=1)<=epsilon:
            flag = False        
        ranks = new_ranks
    return ranks


parsedMetadata = parseAmazonMetadata('meta_Video_Games.json.gz')


#NOTE:For a fast round of iterations the progress monitor does not work correctly
pairCounts = getLimitedNumberPairCounts(10000, parsedMetadata)


coBoughtGraph = Graph()
vertices = list()

for key, value in tqdm(pairCounts.iteritems(), total=len(pairCounts)):
    
    for user in key:
        if not user in vertices:
            coBoughtGraph.add_vertex(name=user)
            vertices.append(user)
    
    coBoughtGraph.add_edge(key[0], key[1], weight=value )


pr = compute_PageRank(get_sparse_adjacency_matrix(coBoughtGraph, attr='weight'), beta=1)


coBoughtMatrix = get_sparse_adjacency_matrix(coBoughtGraph, attr='weight')


#Set special values for the nodes with lowest and highest PageRank
colorList = []
sizeList = []
for vertex in coBoughtGraph.vs:
    if vertex.index == np.argmax(pr):
        colorList.append("blue")
        sizeList.append(10)
    elif vertex.index == np.argmin(pr):
        colorList.append("yellow")
        sizeList.append(10)
    else:
        colorList.append("red")
        sizeList.append(5)

		
#link to different layouts: http://igraph.org/python/doc/tutorial/tutorial.html#layouts-and-plotting
# kk (Kamada-Kawai) and fr (Fruchterman-Reingold) seem to be the most informative

visual_style = {}
visual_style["layout"] = coBoughtGraph.layout("fr")
visual_style["vertex_size"] = sizeList
visual_style["vertex_color"] = colorList
#visual_style["vertex_label"] = coBoughtGraph.vs["name"]
visual_style["edge_width"] = [int(val) for val in coBoughtGraph.es["weight"]]
#visual_style["bbox"] = (300, 300)
#visual_style["margin"] = 20

plot(coBoughtGraph, 'coBoughtGraph.png', **visual_style)


saveGraphAsSparceMatrix(coBoughtGraph, 'coBoughtGraphSparseMatrix')

