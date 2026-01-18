from run_Bert_model import model_train_validate_test
import pandas as pd
from utils import Metric
import os

lcqmc_path = "LCQMC/"
train_df = pd.read_csv(os.path.join(lcqmc_path, "data/train.tsv"),sep='\t',header=None, names=['s1','s2','label'])
dev_df = pd.read_csv(os.path.join(lcqmc_path, "data/dev.tsv"),sep='\t',header=None, names=['s1','s2','label'])
test_df = pd.read_csv(os.path.join(lcqmc_path, "data/test.tsv"),sep='\t',header=None, names=['s1','s2','label'])

target_dir = os.path.join(lcqmc_path, "output/Bert/")

model_train_validate_test(train_df, dev_df, test_df, target_dir, 
         max_seq_len=128, 
         num_labels=2,             
         epochs=100,
         batch_size=128,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0,
         if_save_model=True,
         checkpoint=None)
  
test_result = pd.read_csv(os.path.join(target_dir, 'test_prediction.csv'))
Metric(test_df.label, test_result.prediction) 
