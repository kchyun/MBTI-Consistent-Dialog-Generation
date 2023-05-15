import torch
import json
import torch.utils as utils
from torch.utils.data import Dataset
import os
""" 
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=543
"""
# Aihub : 다자 대화 제외해야 함. 
class AIHubDataset():
    def __init__(self, q_act, r_act, queries, responses, device):
        self.q_act = q_act
        self.r_act = r_act # unused?
        self.queries = queries 
        self.responses = responses 
        self.device = device 
    
    def __getitem__(self, idx):
        q_act = {
            key : torch.tensor(val[idx]).to(self.device)
            for key, val in self.q_act.items()
        }
        query = {
            key : torch.tensor(val[idx]).to(self.device)
            for key, val in self.queries.items()
        }
        response = {
            key : torch.tensor(val[idx]).to(self.device)
            for key, val in self.responses.items()
        }

        return {'q_act' : q_act, 'query' : query, 'response' : response}

    def __len__(self):
        return len(self.responses['input_ids'])
class MBTIDataset(Dataset):
    def __init__(self, a_mbti, personas, queries, responses, device):
        self.a_mbti = a_mbti 
        self.queries = torch.concat(personas, queries)
        self.responses = responses 
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
            for key, val in self.responses.items()
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
def read_aihub_split(split_dir):
    query = []
    response = []
    q_act = []
    r_act = []
    try:
        with open(split_dir, 'r', encoding="utf-8") as file:
            data = json.load(file)
            info = data["info"][0]
            lines = info["annotations"]["lines"]
            print(len(lines))
            for i in range(len(lines)):
                if i == len(lines) - 1: break
                q = lines[i]["norm_text"]
                qs = lines[i]["speechAct"].find('(') + 1
                qe = lines[i]["speechAct"].find(')')
                qa = lines[i]["speechAct"][qs:qe]
                
                if qa == "단언": qa = 0
                elif qa == "지시" : qa = 1
                elif qa == "언약" : qa = 2
                elif qa == "표현" : qa = 3
                 
                r = lines[i+1]["norm_text"]
                rs = lines[i+1]["speechAct"].find('(') + 1
                re = lines[i+1]["speechAct"].find(')')
                ra = lines[i+1]["speechAct"][rs:re]
                
                if ra == "단언": ra = 0
                elif ra == "지시" : ra = 1
                elif ra == "언약" : ra = 2
                elif ra == "표현" : ra = 3
                
                query.append(q)
                response.append(r)
                q_act.append(qa)
                r_act.append(ra)
                
    except FileNotFoundError:
        print("Sorry! File not found!\n")
    
    print("Complete!")
    return q_act, query, r_act, response
    
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