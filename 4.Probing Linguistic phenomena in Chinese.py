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




############Linguistic phenomena in Chinese###############
D2 = dict()
for token in ['被','把','着','了','过']:
    D2[token] = [0,0,0,0] #gold data, attn data, layer, head


#数gold data
for example in data:
    for i in range(len(example["words"])):
        if example["words"][i] == '被' and example["relns"][i] in ['aux:pass','case','case:loc','dep','mark','ADV', 'ATT']: 
            D2['被'][0] +=1
        elif example["words"][i] == '把' and example["relns"][i] in ['case','aux','aux:pass','mark', 'ADV', 'SBV']: 
            D2['把'][0] +=1
        elif example["words"][i] == '着' and example["relns"][i] in ['aux','RAD']: 
            D2['着'][0] +=1
        elif example["words"][i] == '了' and example["relns"][i] in ['discourse:sp','aux','discourse','parataxis', 'RAD', 'CMP', 'VOB', 'POB', 'IOB']: 
            D2['了'][0] +=1
        elif example["words"][i] == '过' and example["relns"][i] in ['aux', 'RAD', 'CMP']: 
            D2['过'][0] +=1


#To count which layer-head are the most frequent
for char in D2.keys():
    maxm = 0
    for layer in range(12):
        for head in range(12):
            num = 0
            for example in data:
                for i in range(len(example["words"])):
                    if example["words"][i] != char:
                        continue
                    if char == '被' and example["relns"][i] not in ['aux:pass','case','case:loc','dep','mark','ADV', 'ATT']:
                        continue
                    if char == '把' and example["relns"][i] not in ['case','aux','aux:pass','mark', 'ADV', 'SBV']:
                        continue
                    if char == '着' and example["relns"][i] not in ['aux','RAD']:
                        continue
                    if char == '了' and example["relns"][i] not in ['discourse:sp','aux','discourse','parataxis', 'RAD', 'CMP', 'VOB', 'POB', 'IOB']:
                        continue
                    if char == '过' and example["relns"][i] not in ['aux', 'RAD', 'CMP']:
                        continue
                    attn = np.array(example["attns"][layer][head])
                    attn[range(attn.shape[0]), range(attn.shape[0])] = 0
                    attn = attn[1:-1, 1:-1]
                    if np.argmax(attn, axis=-1)[i] + 1 == example["heads"][i]:
                        num +=1                    
            if num>maxm :
                maxm = num
                D2[char][1] = num
                D2[char][2] = layer+1
                D2[char][3] = head+1

with open('D2.dat', 'wb') as f:
    pickle.dump(D2, f)







