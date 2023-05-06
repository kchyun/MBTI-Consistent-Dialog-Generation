import torch
import json
import torch.utils as utils
from torch.utils.data import Dataset
import os
""" 
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=543
"""
# 
class AIHubDataset():
    def __init__(self, speaker1, speaker2, queries, responses, device):
        self.speaker1 = speaker1
        self.speaker2 = speaker2 
        self.queries = queries 
        self.responses = responses 
        self.device = device 
    
    def __getitem__(self, idx):
        
        speaker1 = {
            key : torch.tensor(val[idx]).to(self.device)
            for key, val in self.speaker1.items()
        }
        speaker2 = {
            key : torch.tensor(val[idx]).to(self.device)
            for key, val in self.speaker2.items()
        }
        query = {
            key : torch.tensor(val[idx]).to(self.device)
            for key, val in self.queries.items()
        }
        response = {
            key : torch.tensor(val[idx]).to(self.device)
            for key, val in self.responses.items()
        }
        return {'persona1': speaker1, 'person2' : speaker2, 'query' : query, 'response' : response}

    def __len__(self):
        return len(self.responses['input_ids'])

class MBTIDataset(Dataset):
    def __init__(self, a_mbti, questions, answers, device):
        self.a_mbti = a_mbti 
        self.queries = questions
        self.responses = answers 
        self.device = device 
        
    def __getitem__(self, idx):
        a_mbti = {
            key : torch.tensor(val[idx]).to(self.device)
            for key, val in self.a_mbti.items()
        }
        query = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.queries.items()
        }
        response = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.labels.items()
        }
        return {'a_persona' : a_mbti, 'query': query, 'response': response}

    def __len__(self):
        return len(self.responses['input_ids'])

class NLIDataset(Dataset):
    def __init__(self, pre, hyp, device):
        self.pre = pre
        self.hyp = hyp
        self.device = device

    def __getitem__(self, idx):
        pre = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.pre.items()
        }
        hyp = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.hyp.items()
        }
        return {'pre': pre, 'hyp': hyp}

    def __len__(self):
        return len(self.pre['input_ids'])

def read_nli_split(split_dir):
    neutral_pre_list = []
    neutral_hyp_list = []
    cont_pre_list = []
    cont_hyp_list = []
    entail_pre_list = []
    entail_hyp_list = []
    #label_list = []
    try:
        with open(split_dir, "r", encoding = "utf-8") as src:
            for line in src:
                line = line.strip()
                splited = line.split('\t')
                sent1, sent2, gold_label = splited[0], splited[1], splited[2]
                if len(sent1.split(' ')) > len(sent2.split(' ')):
                    pre, hyp = sent1, sent2 
                else:
                    pre, hyp = sent2, sent1
                if splited[2] in ['neutral', 'contradiction', 'entailment']:
                    if splited[2] == 'neutral':
                        neutral_pre_list.append(pre)
                        neutral_hyp_list.append(hyp)
                    elif splited[2] == 'contradiction':
                        cont_pre_list.append(pre)
                        cont_hyp_list.append(hyp)
                    elif splited[2] == 'entailment':
                        entail_pre_list.append(pre)
                        entail_hyp_list.append(hyp)
                    else: pass 
                    #label_list.append(splited[2])
    except FileNotFoundError:
        print(f"Sorry! file not found!")
    return neutral_pre_list, neutral_hyp_list, cont_pre_list, cont_hyp_list, entail_pre_list, entail_hyp_list

def read_mbti_split(split_dir):
    mbti = []
    persona = []
    query = []
    answer = []

    try:
        with open(split_dir, "r", encoding="utf-8") as src:
            for line in src:
                _, _, _, q, a, _, a_mbti = line.split("\t")

                q = list(q.replace("\n", " ").split("[SEP]")[:-1])
                a = list(a.replace("\n", " ").split("[SEP]")[:-1])

                # TBD
                if "t" in a_mbti:
                    persona.append("T_sentence")
                elif "f" in a_mbti:
                    persona.append("F_sentence")
                else:
                    raise (ValueError)

                mbti.append(a_mbti.replace("\n", ""))
                query.append(q)
                answer.append(a)

    except FileNotFoundError:
        print(f"Sorry! The file {split_dir} can't be found.")

    return mbti, persona, query, answer
def create_encoder_input():
    pass

def create_decoder_input():
    pass

if __name__ == '__main__':
    print(os.path.exists('data/kor-nlu-datasets/KorNLI/xnli.dev.ko.tsv'))
    n_pre, n_hyp, cont_pre, cont_hyp, en_pre, en_hyp = read_nli_split('data/kor-nlu-datasets/KorNLI/xnli.test.ko.tsv')
    print(len(n_pre), len(n_hyp), len(cont_pre), len(cont_hyp), len(en_pre), len(en_hyp))
    
    #multinli.train.ko.tsv : 392702
    #snli_1.0_train.ko.tsv : 550152
    #xnli.dev.ko.tsv : 2490
    #xnli.test.ko.tsv : 5010
    
    #print(pre, hyp, label)