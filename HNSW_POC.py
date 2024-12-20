import math
import numpy as np
import heapq
import pandas as pd
import uuid
import time
import sys

from transformers import pipeline
from connectors import mongo_connector
from collections import deque


# FOR MONGODB OPERATIONS MAKE SURE ALGORITHM IS GOOD BEFORE DOING OPTIMIZATION CHANGES

db = mongo_connector()


class Node:
    def __init__(self, score, document):
        self.score = score
        self.document = document

    # true then self < other    
    def __lt__(self, other):
        return self.score < other.score
    
    def __eq__(self, other):
        return self.score == other.score

class Rough_Draft_HSNW:

    # checked
    # optimized
    def cosine_similarity(self, query_node, doc_node):
        query_magnitude = np.linalg.norm(query_node)
        doc_magnitude = np.linalg.norm(doc_node)
        dot_product = np.dot(query_node, doc_node)
        similarity_score = dot_product / (query_magnitude * doc_magnitude)
        return similarity_score

    # checked
    # optimized O (k log k)
    # closest=False, get the farthest document, so its a min heap [ 0.20, 0.30, 0.90]
    # closest=True, get the closest document, so its a max heap [-0.90, -0.30, -0.20]
    def farthest_closest(self,query_node,document_nodes, closest=True,k=1):


        result = []
        query_array = np.array(query_node['embedding'])

        for node in document_nodes:
            node_array = np.array(node['embedding'])
            
            similarity_score = self.cosine_similarity(query_array, node_array)

            if closest:
                similarity_score = -1.0 * similarity_score

            if len(result) < k:
                heapq.heappush(result, Node(similarity_score,node))
            else:
                heapq.heappushpop(result,Node(similarity_score,node))

        k_nodes = []
        # if max heap will return with -1.0 * score
        while result:
            node = heapq.heappop(result)
            k_nodes.append(node)

        return k_nodes

    # checked
    # optimized
    def __init__(self,topic,M,Mmax,Mmxa0, ep=None,entry_layer=-1):
        self.topic = topic
        self.M = M
        self.Mmax = Mmax
        self.Mmax0 = Mmxa0
        self.ep = ep
        self.entry_layer = entry_layer
        self.m_l = 1.0 / math.log(self.M)
        self.embedder = pipeline('feature-extraction', model='bert-base-uncased', tokenizer='bert-base-uncased')
        self.collection = db[topic]

    # checked
    # optimized
    def create_node(self,node_id, text):
        tokenizer_kwargs = {'truncation':True,'max_length':512, 'truncation_side':'right'}
        vector = np.mean(self.embedder(text, **tokenizer_kwargs)[0], axis=0).tolist()

        node = {
            'node_id':node_id,
            'text': text,
            'topic': self.topic,
            'embedding':vector,
            'neighbors': []
        }
        return node

    # checked
    # optimized O(n log n)
    def search_layer(self,new_node,entry_points,k_candidates,layer,text=False):
        

        visited = set([e['node_id'] for e in entry_points]) # ids


        closest_candidates = self.farthest_closest(new_node,entry_points,closest=True,k=len(entry_points))
        # heapq.heapify(closest_candidates) # max heap - score, document (largest element) *-1) C

        potential_candidates = self.farthest_closest(new_node,entry_points,closest=False,k=len(entry_points))
        # heapq.heapify(potential_candidates) # min heap - score, document (smallest element) W


        while len(closest_candidates) > 0:
            closest_candidate_node = heapq.heappop(closest_candidates)
            similarity_c, node_c = -1.0 * closest_candidate_node.score, closest_candidate_node.document

            farthest_potenital_node = potential_candidates[0]
            similarity_f, node_f = farthest_potenital_node.score, farthest_potenital_node.document

            # cosine similarity so bigger number means closer value
            if similarity_c < similarity_f:
                break

            for neighbor_id in node_c['neighbors']: 

                if neighbor_id in visited:
                    continue

                # get neighbor node
                neighbor = self.collection.find_one({"node_id": neighbor_id, "layer": layer},
                                                    {'_id':0, 'text':0, 'topic':0} if not text else {'_id':0,  'topic':0})
                visited.add(neighbor_id)
                similarity_n = self.cosine_similarity(np.array(new_node['embedding']), np.array(neighbor['embedding']))
                # self.farthest_closest(new_node,[neighbor],closest=False,k=1)[0].score

                farthest_potenital_node = potential_candidates[0]
                similarity_f, node_f = farthest_potenital_node.score, farthest_potenital_node.document

                if similarity_n > similarity_f or len(potential_candidates) < k_candidates:
                    heapq.heappush(potential_candidates, Node(similarity_n, neighbor)) # farthest min heapq W
                    heapq.heappush(closest_candidates, Node(-1.0 * similarity_n, neighbor)) # closest max heapq C
                    

                    if len(potential_candidates) > k_candidates:
                        heapq.heappop(potential_candidates)
        
        result = [node.document for node in heapq.nlargest(len(potential_candidates),potential_candidates)]
        return result

    # checked
    # optimized (r log r)
    # new_node, self.M, local_mins_candidates,lc # REMEMBER nodes in neighbor must be sorted from closest to farthest
    def select_neighbors_heuristic(self,new_node,neighbors, M_connections, layer, extendConnections=False, prunedConnections=True):
        
        results = []

        live_candidates = deque(neighbors)
        dead_candidates = []

        visited = set([n['node_id'] for n in neighbors])

        # we don't want to include the acutal node as its own neighbor
        visited.add(new_node['node_id'])

        if extendConnections:
            for doc_node in neighbors:
                for node_id in doc_node[' neighbors']:
                    if node_id in visited:
                        continue
                    visited.add(node_id)
                    node = self.collection.find_one({"node_id": node_id, "layer": layer},
                                                    {'_id':0, 'text':0, 'topic':0})
                    live_candidates.append(node)

        result_score = float('-inf')

        while len(live_candidates) > 0 and len(results) < M_connections:
            candidate_node = live_candidates.popleft()
            
            candidate_score = self.cosine_similarity(np.array(new_node['embedding']), np.array(candidate_node['embedding'])) 
            #self.farthest_closest(new_node,[candidate_node],closest=False,k=1)[0].score

            if candidate_score > result_score:
                results.append(candidate_node)
                result_score = candidate_score
            else:
                heapq.heappush(dead_candidates, Node(-1.0 * candidate_score,candidate_node))

        if prunedConnections:
            while len(dead_candidates) > 0 and len(results) < M_connections:
                results.append(heapq.heappop(dead_candidates).document)

        return results

    # checked
    def insert(self,new_node,candidate_number=3):

        insert_layer = math.floor((-np.log(np.random.uniform(0,1)) * self.m_l))
        ep = [self.ep] 
        
        for lc in range(self.entry_layer, insert_layer, -1):

            local_mins = self.search_layer(new_node,ep,1,lc) # returns ef documents
            closest = local_mins[0] #self.farthest_closest(new_node,local_mins,closest=False,k=1)[0].document
            ep = [self.collection.find_one({"node_id": closest['node_id'], "layer": lc - 1},
                                           {'_id':0, 'text':0, 'topic':0})] 
            # returns document with the closest embedding to q from local_min
            # not the lower layers

        for lc in range(min(self.entry_layer, insert_layer),-1,-1):
            local_mins_candidates = self.search_layer(new_node, ep, candidate_number,lc) # returns document nodes

            neighbors = self.select_neighbors_heuristic(new_node,local_mins_candidates,self.M ,lc)

            # Add neighbor and layer data
            new_node['neighbors'] = [node['node_id'] for node in neighbors]
            new_node['layer'] = lc

            # Add new_node to collection
            db_response = self.collection.insert_one(new_node)
            # delete autmoatic add _id
            del new_node['_id']


            # for each neighbor check the amount of connections and prune
            for n in neighbors:

                # append new node to n's neighbors
                n['neighbors'].append(new_node['node_id'])

                eNewConnections = []

                # grab node_ids
                for node_id in n['neighbors']:
                    eNewConnections.append(self.collection.find_one({"node_id": node_id, "layer": lc},
                                                                    {'_id':0, 'text':0, 'topic':0}))
                    
                eNewConnections = [element.document for element in self.farthest_closest(n, eNewConnections,closest=True,k=len(eNewConnections))]

                if lc == 0 and len(eNewConnections) > self.Mmax0:
                    eNewConnections = self.select_neighbors_heuristic(n,eNewConnections,self.Mmax0,lc) # returns documents

                elif lc != 0 and len(eNewConnections) > self.Mmax:
                    eNewConnections = self.select_neighbors_heuristic(n,eNewConnections,self.Mmax,lc) # returns documents

                n['neighbors'] = [conn['node_id'] for conn in eNewConnections]

                query_filter = {'node_id': n['node_id'], 'layer':lc}
                update_operation = {'$set' : { 'neighbors' : n['neighbors'] }}

                # change this to bulk write later for optimization?
                db_response = self.collection.update_one(query_filter, update_operation)
                print(f"For node {n['node_id']} neighbors are {n['neighbors']} at LAYER {lc}")

            # find local_min_candidates in the layer down - potential use of $in operator
            if lc > 0:
                for i in range(len(local_mins_candidates)):
                    local_mins_candidates[i] = self.collection.find_one({"node_id": local_mins_candidates[i]['node_id'], "layer": lc - 1}, 
                                                                        {'_id':0, 'text':0, 'topic':0})

            ep = local_mins_candidates # have to pass in the nodes from the next layer
        
        # create new layers and change ep if new layer is greater than old layer
        if insert_layer > self.entry_layer:

            for lc in range(self.entry_layer + 1, insert_layer + 1, 1):             
                new_node['layer'] = lc
                new_node['neighbors'] = []

                # add new node to collection
                db_response = self.collection.insert_one(new_node)
                print(f"Created entry node {new_node['node_id']} at layer {lc}")
                # delete object id
                del new_node['_id']

            self.entry_layer = insert_layer
            new_node['layer'] = insert_layer
            self.ep = new_node

    # checked
    def search(self,query, K, candidate_list=3):
        query_node = {'embedding': np.mean(self.embedder(query)[0], axis=0).tolist()}
        ep = [self.ep]
        for lc in range(self.entry_layer, 0, -1):

            local_mins = self.search_layer(query_node,ep,1,lc) # returns ef documents
            closest = local_mins[0]
            
            ep = [self.collection.find_one({"node_id": closest['node_id'], "layer": lc - 1},
                                           {'_id':0, 'topic':0})]
            # returns document with the closest embedding to q from local_min
            # not the lower layers
        W = self.search_layer(query_node,ep,candidate_list,0,text=True) # returns documents in sorted order

        return [w['text']  for w in W[:K]]

    # checked
    def get_ep(self, id, layer):
        self.ep = self.collection.find_one({"node_id": id, "layer": layer},
                                           {'_id':0, 'text':0, 'topic':0})
        self.entry_layer = layer




graph = Rough_Draft_HSNW("technology",5,5,10) # M, M, M * 2

def search_graph():
    #graph.get_ep('949937ff-edfd-4407-96ca-62b5b3b17b49',1)
    print(graph.ep['node_id'])
    print(graph.entry_layer)

    while True:
        print("Query Knowledge Base, -1 to exit")
        query = input()
        if query == "-1":
            break
        results = graph.search(query, K=5, candidate_list=20) # give more candidates to work with

        for text in results:
            print(text)
            print("\n")


def create_graph():
    
    df = pd.read_excel('old_data/combined.xlsx')
    df = df.dropna(subset=['content'])
    # 194, 202
    df = df[df['topic'] == "technology"]
    start_time = time.time()
    j = 0
    chunks = 0
    chunk_size = 300

    for index, row in df.iterrows():
        lis = row['content'].split()
        for i in range(0, max(len(lis),chunk_size), chunk_size):

            text = lis[i:i + chunk_size]

            if len(text) < (chunk_size - 150):
                continue

            text = ' '.join(text)

            try:
                node = graph.create_node(str(uuid.uuid4()), text)
                print(node['node_id'])
                print(node['text'])
                graph.insert(node, candidate_number=15) # give more candidates to read from
                chunks += 1
                input()
            except KeyboardInterrupt:
                print("\nProcess interrupted by user.")
                exit(0)
            except Exception as e:
                print("\n-----------------------------------------Error Stacktrace-----------------------------------------------")
                print(e)
                print("-----------------------------------------Text-----------------------------------------------")
                print(text)
                print("-----------------------------------------Index-----------------------------------------------")
                print(f"ran into error at index {index}")
                print("-----------------------------------------++++++++++++-----------------------------------------------")
            #sys.stdout.write(f'\r{(j/len(df)) * 100} percent completed')
            #sys.stdout.flush()
        j += 1
    end_time = time.time()
    print(f"Total documents: {chunks}")
    #sys.stdout.write(f'\r{(j/len(df)) * 100} percent completed') 
    print("\n---------------------------------Finished Initializing Knowledge base------------------------\n")

    elapsed_time_seconds = end_time - start_time
    minutes = int(elapsed_time_seconds // 60)
    seconds = elapsed_time_seconds % 60

    print(f"Elapsed time: {minutes} minutes and {seconds:.2f} seconds")

create_graph()
search_graph()


