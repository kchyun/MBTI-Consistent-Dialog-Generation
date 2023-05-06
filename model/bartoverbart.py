from argparse import ArgumentParser 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from xlibs import EncoderDecoderModel 
from xlibs import BartTokenizer
from xlibs import AdamW 
from kobart import get_kobart_tokenizer
#from kobert_tokenizer import KoBERTTokenizer
from datasets.dataset import AIHubDataset
from datasets.dataset import MBTIDataset 

import os 
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def init(tokenizer, model):
    # special tokens? 
    
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
    