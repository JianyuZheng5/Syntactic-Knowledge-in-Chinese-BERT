"""Preprocesses dependency parsing data and writes the result as JSON."""
# -*- coding: utf-8 -*-

import os
import re
import utils
import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel, BertConfig

config = BertConfig.from_pretrained('./../bert-base-chinese-random', output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('./../bert-base-chinese-random')
model = BertModel(config=config)
model.eval()



def get_attention(text, seg_ids):
    sent = '[CLS]' + text + '[SEP]'
    str_tokenized_sents = tokenizer.tokenize(sent)
    if len(str_tokenized_sents) != seg_ids[-1]:
        return 'none'
    else:
        indexed_tokens = tokenizer.convert_tokens_to_ids(str_tokenized_sents)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = model(tokens_tensor)
            attmap = outputs[-1]

        new_attmap = np.zeros((12,12, len(seg_ids)-1, len(seg_ids)-1))
        for layer in range(12):
            for head in range(12):
                attention = attmap[layer][0,head,:,:].numpy()
                re_attention = np.zeros((len(seg_ids)-1, len(seg_ids)-1))
                for i in range(len(seg_ids)-1):
                    for j in range(len(seg_ids)-1):
                        re_attention[i][j] = np.sum(attention[seg_ids[i]:seg_ids[i+1], seg_ids[j]:seg_ids[j+1]])/(seg_ids[i+1]-seg_ids[i])
                new_attmap[layer][head] = re_attention
        return new_attmap



def preprocess_depparse_data1(raw_data_file):
  examples = []
  with open(raw_data_file, encoding="utf-8") as f:
      current_example = {"words": [], "relns": [], "heads": []}

      for line in f:
          line = line.strip()
          if "# text = " in line:
              text = line.replace('# text = ',"").strip()
              seg_ids = [0,1]
              id = 1

          if "# sent_id =" in line or "# text = " in line or "# newdoc id =" in line or "# text_en" in line:
              continue
          if line:
              word = line.split()[1]
              head= line.split()[6]
              reln= line.split()[7]

              id += len(tokenizer.tokenize(word))
              seg_ids.append(id)
              current_example["words"].append(word)
              current_example["relns"].append(reln)
              current_example["heads"].append(int(head))
          else:
              if current_example["words"] == []:
                  continue
              seg_ids.append(id+1)
              if get_attention(text, seg_ids) == 'none':
                  current_example = {"words": [], "relns": [], "heads": []}
                  continue
              else:
                  current_example["attns"] = get_attention(text, seg_ids)
                  examples.append(current_example)
                  current_example = {"words": [], "relns": [], "heads": []}

  #utils.write_json(examples, raw_data_file.replace(".txt", ".json"))
  with open(raw_data_file.replace(".txt", "_random.pkl"), 'wb') as f:
    pickle.dump(examples ,f)


def preprocess_depparse_data2(raw_data_file):
  examples = []
  with open(raw_data_file, encoding="utf-8") as f:
      current_example = {"words": [], "relns": [], "heads": []}


      seg_ids = [0,1]
      id = 1
      text = ''
      for line in f:
          line = line.strip()
          if line:
              word = line.split()[1]
              head= line.split()[6]
              reln= line.split()[7]

              text += word
              id += len(tokenizer.tokenize(word))
              seg_ids.append(id)
              current_example["words"].append(word)
              current_example["relns"].append(reln)
              current_example["heads"].append(int(head))
          else:
              if current_example["words"] == []:
                  continue
              seg_ids.append(id+1)
              if get_attention(text, seg_ids) == 'none':
                  current_example = {"words": [], "relns": [], "heads": []}
                  continue
              else:
                  current_example["attns"] = get_attention(text, seg_ids)
                  examples.append(current_example)
                  current_example = {"words": [], "relns": [], "heads": []}
              seg_ids = [0,1]
              id = 1
              text = ''


  #utils.write_json(examples, raw_data_file.replace(".txt", ".json"))
  with open(raw_data_file.replace(".txt", ".pkl"), 'wb') as f:
      pickle.dump(examples ,f)




file = "UD_all.txt"
def main():
  if 'UD' in file:
      preprocess_depparse_data1(file)
  elif 'HIT' in file:
      preprocess_depparse_data2(file)
  print("Done!")


if __name__ == "__main__":
  main()
