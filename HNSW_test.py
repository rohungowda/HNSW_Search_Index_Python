import math
import np


class HNSW:


    def __init__(self, M):
        self.ep, self.L = None, None # implement get ep here
        self.Mmax, self.M, self.Mmax0 = M, M, 2 * M
        self.m_l = 1.0 / math.log(self.M)


    def cosine_similarity(self, query_node, doc_node):
        query_magnitude = np.linalg.norm(query_node)
        doc_magnitude = np.linalg.norm(doc_node)
        dot_product = np.dot(query_node, doc_node)
        similarity_score = dot_product / (query_magnitude * doc_magnitude)
        return similarity_score

    # only use these with the heapq implementations
    class Node:
        def __init__(self, score, node):
            self.score = score
            self.node = node

        # true then self < other    
        def __lt__(self, other):
            return self.score < other.score
        
        def __eq__(self, other):
            return self.score == other.score



    def INSERT(self, insert_node, efConstruction):
        W = []
        ep = [self.ep]

        insert_layer = math.floor((-np.log(np.random.uniform(0,1)) * self.m_l))

        for lc in range(self.L, insert_layer, -1):
            W = self.SEARCH_LAYER(insert_node, ep, ef=1, lc=lc)
            ep = [self.collection.find_one({"node_id": W[0]['node_id'], "layer": lc - 1}, {'_id':0, 'text':0, 'topic':0})]

        for lc in range(min(self.L, insert_layer), -1, -1):
            W = self.SEARCH_LAYER(insert_node, ep, ef=efConstruction, lc=lc)
            neighbors = self.SELECT_NEIGHBORS(insert_node, W, self.M, lc=lc)
            # add bidirectional connections from node to neighbors

            # add connections from neighbors to insert_node and insert insert_node
            insert_node['neighbors'] = [node['node_id'] for node in neighbors]
            insert_node['layer'] = lc

            self.collection.insert_one(insert_node, {'_id':0, 'text':0, 'topic':0})


            for node in neighbors:

                node['neighbors'].append(insert_node['node_id'])
                eConnections = node['neighbors']


                M = None

                if (len(eConnections) > self.Mmax and lc > 0):
                    M = self.Mmax
                elif (len(eConnections) > self.Mmax0 and lc == 0):
                    M = self.Mmax0
                
                if M:
                    eConnections = list(self.collection.find({'node_id':{'$in':eConnections}, 'layer':lc}, {'_id':0, 'text':0, 'topic':0}))
                    eConnections = self.SELECT_NEIGHBORS(node, eConnections, M,lc)
                                
                    node['neighbors'] = [conn['node_id'] for conn in eConnections]
                else:
                    node['neighbors'] = eConnections

                query_filter = {'node_id': node['node_id'], 'layer':lc}
                update_operation = {'$set' : { 'neighbors' : node['neighbors'] }}

                self.collection.update_one(query_filter, update_operation)


            if lc > 0:
                ep = list(self.collection.find({'node_id':{'$in':[w['node_id'] for w in W]}, 'layer':(lc - 1)}, {'_id':0, 'text':0, 'topic':0}))
        
        if insert_layer > self.L:
            for lc in range(self.L + 1, insert_layer + 1, 1):             
                insert_node['layer'] = lc
                insert_node['neighbors'] = []

                self.collection.insert_one(insert_node,{'_id':0, 'text':0, 'topic':0})
            self.L = insert_layer
            self.ep = insert_node