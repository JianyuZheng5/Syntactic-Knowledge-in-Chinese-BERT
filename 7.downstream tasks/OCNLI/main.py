from run_Bert_model import model_train_validate_test
import pandas as pd
from utils import Metric
import os

ocnli_path = "./OCNLI/"
train_df = pd.read_csv(os.path.join(ocnli_path, "data/train.csv"),header=None, names=['s1','s2','label','genre'])
dev_df = pd.read_csv(os.path.join(ocnli_path, "data/dev.csv"),header=None, names=['s1','s2','label','genre'])
test_df = pd.read_csv(os.path.join(ocnli_path, "data/test.csv"),header=None, names=['s1','s2','label','genre'])

target_dir = os.path.join(ocnli_path, "output/Bert/")

model_train_validate_test(train_df, dev_df, test_df, target_dir , 
         max_seq_len=128,
         num_labels=3,
         epochs=100,
         batch_size=128,
         lr=2e-02,
         patience=1,
         max_grad_norm=10.0,
         if_save_model=True,
         checkpoint=None)

test_result = pd.read_csv(os.path.join(target_dir, 'test_prediction.csv'))
Metric(test_df.label, test_result.prediction)
# torch.cuda.empty_cache() #释放cuda的显存
