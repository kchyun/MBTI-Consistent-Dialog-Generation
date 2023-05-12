from argparse import ArgumentParser
import json 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from xlibs import EncoderDecoderModel 
from xlibs import BartTokenizer
from xlibs import AdamW 
from kobart import get_kobart_tokenizer
#from kobert_tokenizer import KoBERTTokenizer
from datasets.dataset import AIHubDataset, NLIDataset
from datasets.dataset import MBTIDataset 

import os 
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def init(tokenizer, model):
    # special tokens? 
    # more required options?
    
    model.config.max_length = 32 
    model.config.min_length = 3
    model.config.early_stopping = True 
    model.config.length_penalty = 1.0 
    
    return tokenizer, model 

def train(args):
    print("\nInitialized Model...\n")
    if os.path.exists(args.checkpoint):
        model = EncoderDecoderModel.from_pretrained(args.checkpoint)
    else:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            args.encoder_model, args.decoder_model, args.decoder2_model
        )
    
    model.to(device)
    model.train()
    
    print("Load tokenized data..\n")
    tokenizer = get_kobart_tokenizer()
    tokenizer = tokenizer.KoBERTTokenizer.from_pretrained(args.encoder_model)
    
    tokenizer, model = init(tokenizer, model)
    # tokenize 필요

    # tokenize
    if args.dumped_token is None:
        print('Pre-tokenized files must be provided.')
        raise (ValueError)
    else:
        path = args.dumped_token
        try:
            print(f"Load tokenized dataset from {args.dumped_token}.")

            # Loading training/validation set
            if args.dataset_type == 'aihub':
                with open(path + 'train_qact.json', 'w') as train_qacts:
                    print("Load train_qact")
                    tmp = train_qacts.readline()
                    train_qact = json.loads(tmp)
                with open(path + 'train_ract.json', 'w') as train_racts:
                    print("Load train_ract")
                    tmp = train_racts.readline()
                    train_ract = json.loads(tmp)
                with open(path + 'train_query.json', 'w') as train_query:
                    print("Load train_query")
                    tmp = train_query.readline()
                    train_query_tokenized = json.loads(tmp)
                with open(path + 'train_response.json', 'w') as train_response:
                    print("Load train_response")
                    tmp = train_response.readline()
                    train_response_tokenized = json.loads(tmp)
                with open(path + 'val_qact.json', 'w') as val_qacts:
                    print("Load train_qact")
                    tmp = val_qacts.readline()
                    val_qact = json.loads(tmp)
                with open(path + 'val_ract.json', 'w') as val_racts:
                    print("Load val_ract")
                    tmp = val_racts.readline()
                    val_ract = json.loads(tmp)
                with open(path + 'val_query.json', 'w') as val_query:
                    print("Load val_query")
                    tmp = val_query.readline()
                    val_query_tokenized = json.loads(tmp)
                with open(path + 'val_response.json', 'w') as val_response:
                    print("Load val_response")
                    tmp = val_response.readline()
                    val_response_tokenized = json.loads(tmp)
            elif args.dataset_type == 'mbti':
                with open(path + 'train_mbti.json', 'w') as train_mbti:
                    print("Load train_mbti")
                    tmp = train_mbti.readline()
                    train_mbti_tokenized = json.loads(tmp)
                with open(path + 'train_persona.json', 'w') as train_persona:
                    print("Load train_persona")
                    tmp = train_persona.readline()
                    train_persona_tokenized = json.loads(tmp)
                with open(path + 'train_query.json', 'w') as train_query:
                    print("Load train_query")
                    tmp = train_query.readline()
                    train_query_tokenized = json.loads(tmp)
                with open(path + 'train_response.json', 'w') as train_response:
                    print("Load train_response")
                    tmp = train_response.readline()
                    train_response_tokenized = json.loads(tmp)

                with open(path + 'val_mbti.json', 'w') as val_mbti:
                    print("Load val_mbti")
                    tmp = val_mbti.readline()
                    val_mbti_tokenized = json.loads(tmp)
                with open(path + 'val_persona.json', 'w') as val_persona:
                    print("Load val_persona")
                    tmp = val_persona.readline()
                    val_persona_tokenized = json.loads(tmp)
                with open(path + 'val_query.json', 'w') as val_query:
                    print("Load val_query")
                    tmp = val_query.readline()
                    val_query_tokenized = json.loads(tmp)
                with open(path + 'val_response.json', 'w') as val_response:
                    print("Load val_response")
                    tmp = val_response.readline()
                    val_response_tokenized = json.loads(tmp)
            elif args.dataset_type == 'nli':
                # NLI
                with open(path + 'neutral_pre.json', 'w') as neutral_pre:
                    print("Load neutral_pre")
                    tmp = neutral_pre.readline()
                    neutral_pre_tokenized = json.loads(tmp)
                with open(path + 'neutral_hyp.json', 'w') as neutral_hyp:
                    print("Load neutral_hyp")
                    tmp = neutral_hyp.readline()
                    neutral_hyp_tokenized = json.loads(tmp)
                
                with open(path + 'contradiction_pre.json', 'w') as contradiction_pre:
                    print("Load contradiction_pre")
                    tmp = contradiction_pre.readline()
                    contradiction_pre_tokenized = json.loads(tmp)
                with open(path + 'contradiction_hyp.json', 'w') as contradiction_hyp:
                    print("Load contradiction_hyp")
                    tmp = contradiction_hyp.readline()
                    contradiction_hyp_tokenized = json.loads(tmp)

                with open(path + 'entailment_pre.json', 'w') as entailment_pre:
                    print("Load entailment_pre")
                    tmp = entailment_pre.readline()
                    entailment_pre_tokenized = json.loads(tmp)
                with open(path + 'entailment_hyp.json', 'w') as entailment_hyp:
                    print("Load entailment_hyp")
                    tmp = entailment_hyp.readline()
                    entailment_hyp_tokenized = json.loads(tmp)
            
        except FileNotFoundError:
            print(f"Sorry! The files in {args.dumped_token} can't be found.")
            raise ValueError
        
    # prepare dataset
    if args.dataset_type == 'aihub':
        train_dataset = AIHubDataset(train_qact,
                                        train_ract,
                                        train_query_tokenized,
                                        train_response_tokenized,
                                        device)
        train_dataset = AIHubDataset(val_qact,
                                        val_ract,
                                        val_query_tokenized,
                                        val_response_tokenized,
                                        device)
    elif args.dataset_type == 'mbti':
        train_dataset = MBTIDataset(train_mbti,
                                        train_persona_tokenized,
                                        train_query_tokenized,
                                        train_response_tokenized,
                                        device)
        val_dataset = MBTIDataset(val_mbti,
                                    val_persona_tokenized,
                                    val_query_tokenized,
                                    val_response_tokenized,
                                    device)
    elif args.dataset_type == 'nli':
        neutral_nli_dataset = NLIDataset(neutral_pre_tokenized,
                                            neutral_hyp_tokenized,
                                            device)
        contradiction_nli_dataset = NLIDataset(contradiction_pre_tokenized,
                                                contradiction_hyp_tokenized,
                                                device)
        entailment_nli_dataset = NLIDataset(entailment_pre_tokenized,
                                                entailment_hyp_tokenized,
                                                device)
    
    # training
    print("\nStart Training...")
    if args.dataset_type == 'aihub' | 'mbti':
        train_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True)
        val_loader = DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)
    elif args.dataset_type == 'nli':
        neutral_ul_loader = DataLoader(neutral_nli_dataset,
                                       batch_Size=args.batch_size,
                                       shuffle=True)
        contradiction_ul_loader = DataLoader(contradiction_nli_dataset,
                                       batch_Size=args.batch_size,
                                       shuffle=True)
        entailment_ul_loader = DataLoader(entailment_nli_dataset,
                                       batch_Size=args.batch_size,
                                       shuffle=True)
    
    n_ul_iterator = enumerate(neutral_ul_loader)
    n_ul_len = neutral_ul_loader.__len__()
    n_global_step = 0

    c_ul_iterator = enumerate(contradiction_ul_loader)
    c_ul_len = contradiction_ul_loader.__len__()
    c_global_step = 0

    e_ul_iterator = enumerate(entailment_ul_loader)
    e_ul_len = entailment_ul_loader.__len__()
    e_global_step = 0

    optim_warmup = AdamW(model.parameters(), lr=args.warm_up_learning_rate)
    optim = AdamW(model.parameters(), lr=args.learning_rate)

    
    
    #############################
    # prediction
            with open(path + 'test_query.json', 'w') as test_query:
                print("Load test_query")
                tmp = test_query.readline()
                test_query_tokenized = json.loads(tmp)
            with open(path + 'test_response.json', 'w') as test_response:
                print("Load test_response")
                tmp = test_response.readline()
                test_response_tokenized = json.loads(tmp)
    
    
    # evaluation
    
    
if __name__ == '__main__':
    parser = ArgumentParser("EncoderDecoder Model")
    parser.add_argument("--device", default = "cuda:0" , type = str)
    #Training
    parser.add_argument("--batch_size", default = 32, type = int)
    parser.add_argument("--checkpoint", type = str, 
                        help = "path to checkpoint")
    #Model
    parser.add_argument("--encoder_model", type = str, default = "./pretrained_models/kobart-base-v2")
    parser.add_argument("--decoder_model", type = str, default = "./pretrained_models/kobart-base-v2")
    parser.add_argument("--decoder2_model", type = str, default = "./pretrained_models/kobart-based-v2")
    
    
    #Data
    parser.add_argument("--dataset_type", type = str, default = "aihub")
    