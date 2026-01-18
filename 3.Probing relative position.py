# -*- coding: utf-8 -*-
import collections
import pickle
import numpy as np


fname = "HIT_all_cws.pkl"

def load_pickle(fname):
    with open(fname, "rb") as f:
        #return pickle.load(f, encoding='iso-8859-1') 
        return pickle.load(f, encoding='utf-8') 
data = load_pickle(fname)


'''
############relative position###############
D1 = dict()
for i in range(-10, 11):
    D1[int(i)] = [0,0,0,0] #gold data, attn data, layer, head


#数gold data

for dist in range(-10, 11):
    for example in data:
        for i in range(len(example["words"])):
            if (example["heads"][i] - int(i+1)) == dist and example["heads"][i] !=0:
                D1[dist][0]+=1 


#To count which layer-head are the most frequent
for dist in range(-10, 11):
    maxm = 0
    for layer in range(12):
        for head in range(12):
            num = 0
            for example in data:
                for i in range(len(example["words"])):
                    if (example["heads"][i] -int (i+1)) != dist:
                        continue
                    attn = np.array(example["attns"][layer][head])
                    attn[range(attn.shape[0]), range(attn.shape[0])] = 0
                    attn = attn[1:-1, 1:-1]
                    if np.argmax(attn, axis=-1)[i] + 1 == example["heads"][i]:
                        num +=1                    
            if num>maxm :
                maxm = num
                D1[dist][1] = num
                D1[dist][2] = layer+1
                D1[dist][3] = head+1

with open('D1.dat', 'wb') as f:
    pickle.dump(D1, f)
'''




############relative position for frequent relations###############
D4 = dict()
for i in range(-10, 11):
    D4[int(i)] = [0,0,0,0] #gold data, attn data, layer, head


# count gold data
arc = 'nmod'

for dist in range(-10, 11):
    for example in data:
        for i in range(len(example["words"])):
            if example["relns"][i] != arc:     # add this sentence
                continue
            if (example["heads"][i] - int(i+1)) == dist and example["heads"][i] !=0:
                D4[dist][0]+=1 


#To count which layer-head are the most frequent
for dist in range(-10, 11):
    maxm = 0
    for layer in range(12):
        for head in range(12):
            num = 0
            for example in data:
                for i in range(len(example["words"])):
                    if (example["heads"][i] -int (i+1)) != dist:
                        continue
                    if example["relns"][i] != arc:   # add this sentence
                        continue
                    attn = np.array(example["attns"][layer][head])
                    attn[range(attn.shape[0]), range(attn.shape[0])] = 0
                    attn = attn[1:-1, 1:-1]
                    if np.argmax(attn, axis=-1)[i] + 1 == example["heads"][i]:
                        num +=1                    
            if num>maxm :
                maxm = num
                D4[dist][1] = num
                D4[dist][2] = layer+1
                D4[dist][3] = head+1

with open('D4.dat', 'wb') as f:
    pickle.dump(D4, f)



