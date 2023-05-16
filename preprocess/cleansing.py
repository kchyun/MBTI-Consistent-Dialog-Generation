import re
import emoji
from soynlp.normalizer import repeat_normalize
from os.path import join
import json
from glob import glob
import pandas as pd

def is_proper_length(sentence):
    return 8 <= len(sentence) and len(sentence) <= 512

def clean(sentence): 
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    sentence = pattern.sub(' ', sentence)
    sentence = emoji.replace_emoji(sentence, replace='') #emoji 삭제
    sentence = url_pattern.sub('', sentence)
    sentence = sentence.strip()
    sentence = repeat_normalize(sentence, num_repeats=2)

    return sentence

def clean_aihub_dataset():
    print("Cleansing Aihub dataset...\n")
    dir = '.\\data\\aihub\\kakao'
    output_path = './data/aihub/kakao/kakao_cleaned.jsonl'

    aihub_files = list(glob(join(dir, "*\\*.*")))

    conversations = []

    for file in aihub_files:
        print(f"{file} preprocessing...")
        try:
            with open(file, 'r', encoding="utf-8") as f:
                data = json.load(f)
                info = data["info"][0]
                lines = info["annotations"]["lines"]

                for i in range(len(lines)):
                    if i == len(lines) - 1: break
                    q = lines[i]["norm_text"]
                    r = lines[i+1]["norm_text"]

                    conversation = {'query': q, 'response': r}
                    conversations.append(conversation)
            print("done")

        except FileNotFoundError:
            print("Sorry! File not found!\n")
    
    with open(output_path, 'w', encoding='utf-8') as kakao_cleaned:
        json.dump(conversations, kakao_cleaned, indent = 4 ,ensure_ascii=False)
    
    print("Completed Cleansing Aihub dataset queries and reponses...\n")

def clean_mbti_dataset(file_name):
    print(f"Cleansing {file_name} dataset...\n")

    if file_name == "qna.tsv":
        data = pd.read_csv(f'./data/mbti/{file_name}', sep='\t')

    elif file_name == "multiple_qna.tsv":
        data = pd.read_csv(f'./data/mbti/{file_name}', sep='\t', names=['id', 'article_id', 'menu_id', 'question', 'answer', 'q_mbti', 'a_mbti'])   

    for idx in range(len(data)):
        data.loc[idx, 'question'] = data.loc[idx, 'question']
        data.loc[idx, 'answer'] = data.loc[idx, 'answer']

    if file_name == "qna.tsv":
        data = data.drop(data.columns[0], axis=1)
        data.to_csv('./data/mbti/qna_cleaned.tsv', sep='\t')
    elif file_name == "multiple_qna.tsv":
        data.to_csv('./data/mbti/multiple_qna_cleaned.tsv', sep='\t', header=['id', 'article_id', 'menu_id', 'question', 'answer', 'q_mbti', 'a_mbti'])
    
    print(f"Completed Cleansing {file_name} dataset queries and reponses...\n")

if __name__ == '__main__':
    clean_mbti_dataset('qna.tsv')