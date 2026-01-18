from framework import Framework, set_seed
from data_loader import NERDataset
from model import BERTforNER_CRF
from transformers import BertConfig, BertTokenizer
import argparse
import torch
import json
import os

def main(args):
    labels = ['O', 'B', 'M', 'E']  #ancient
    args.num_labels = len(labels)
    
    tokenizer = None
    word2id = None

     # use 'bert-base-chinese' model
    pretrained_model_name = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    config = BertConfig.from_pretrained(pretrained_model_name, num_labels=args.num_labels, hidden_dropout_prob=args.hidden_dropout_prob)
    model = BERTforNER_CRF.from_pretrained(pretrained_model_name, config=config, use_crf=args.crf)
    model.to(args.device)


    framework = Framework(args)


    if args.mode == "train":
        print("loading training dataset...")
        train_dataset = NERDataset(
            file_path=args.train_file, 
            labels=labels,
            tokenizer=tokenizer, 
            max_len=args.max_len)

        framework.train(train_dataset, model, labels)


    print("\Testing ...")
    print("loading dev datasets...")
    test_dataset = NERDataset(
            file_path=args.test_file, 
            labels=labels, 
            word2id=word2id, 
            tokenizer=tokenizer, 
            max_len=args.max_len, 
            is_BERT=is_BERT)

    model.load_state_dict(torch.load(args.save_model))
    framework.test(test_dataset, model, labels)

if __name__ == "__main__":
    
    set_seed(2020)

    parser = argparse.ArgumentParser()

    # task setting
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--crf', action='store_true')

    # train setting
    parser.add_argument('--max_len', type=int, default=100)

    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-5)


    # file path
    parser.add_argument('--train_file', type=str, default='./data/train.txt')
    parser.add_argument('--test_file', type=str, default='./data/test.txt')

    parser.add_argument('--save_model', type=str, default='./save_model/')
    parser.add_argument('--output_dir', type=str, default='./output/')

    # others
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.15)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    args = parser.parse_args()

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for file_dir in [args.save_model, args.output_dir]:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

    if args.crf:
        save_name = args.model + "_crf"
    else:
        save_name = args.model

    args.save_model = os.path.join(args.save_model, save_name + ".pt")
    args.output_dir = os.path.join(args.output_dir, save_name + ".result")

    print(args)
    main(args)
