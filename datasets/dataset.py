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

        return {'persona' : q_act, 'query' : query, 'response' : response}

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
        return {'persona' : a_mbti, 'query': query, 'response': response}

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
            # print(len(lines))
            for i in range(len(lines)):
                if i == len(lines) - 1: break
                q = lines[i]["norm_text"]
                qa = lines[i]["speechAct"].replace("(", " ").replace(")", " ")

                r = lines[i+1]["norm_text"]
                ra = lines[i+1]["speechAct"].replace("(", " ").replace(")", " ")
                
                query.append(q)
                response.append(r)
                q_act.append(qa)
                r_act.append(ra)
                
    except FileNotFoundError:
        print("Sorry! File not found!\n")
    
    # print("Complete!")
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
                _, _, _, _, q, a, _, a_mbti = line.split("\t")

                q = q.replace("\n", " ").replace("[SEP]", " ").strip()
                a = a.replace("\n", " ").replace("[SEP]", " ").strip()

                persona_sentence = ''
                
                
                if "t" in a_mbti:
                    persona_sentence += "상황의 이유와 결과가 궁금하며, 해결책을 제시합니다. 사실을 바탕으로 이성적이고 논리적으로 이야기합니다."
                elif "f" in a_mbti:
                    persona_sentence += "상대방의 기분이 어떤지 공감, 축하 또는 위로합니다. 유연하고 융통성 있게 대처합니다." 
                else:
                    raise (ValueError)
                
                if "n" in a_mbti:
                    persona_sentence += "상상력이 풍부하여 비유적이고 추상적이다. 직관적이고 직감을 중요하게 생각한다."
                elif "s" in a_mbti:
                    persona_sentence += "감각을 통해 느낀 경험과 사실이 중요하다. 규칙과 원리를 중요하게 생각한다."
                else:
                    raise (ValueError)
                
                mbti.append(a_mbti.strip())
                persona.append(persona_sentence)
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