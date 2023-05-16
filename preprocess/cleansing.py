import re
import emoji
from soynlp.normalizer import repeat_normalize
from os.path import join
import json
from glob import glob

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
    print("Preprocessing...\n")
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
    
    print("Completed Cleansing queries and reponses...\n")

if __name__ == '__main__':
    clean_aihub_dataset()