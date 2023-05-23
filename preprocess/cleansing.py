import re
from soynlp.normalizer import repeat_normalize
from os.path import join
import json
from glob import glob
import pandas as pd
from hanspell import spell_checker


def is_proper_length(sentence):
    return 8 <= len(sentence) and len(sentence) <= 512


def get_mbti_keywords():
    prefixes = {
        'en': ['엔', '엥'],
        'es': ['엣'],
        'in': ['인', '잉'],
        'is': ['잇']
    }
    suffixes = {
        'fp': ['프피', '뿌피', '뿌삐', '픞', '픕'],
        'fj': ['프제', '픚'],
        'tp': ['팁', '팊', '티피', '티삐'],
        'tj': ['티제', '팆']
    }

    korean_mbti_dic = {}

    for p_eng, p_kors in prefixes.items():
        for s_eng, s_kors in suffixes.items():
            for p in p_kors:
                for s in s_kors:
                    korean_mbti_dic[p + s] = p_eng + s_eng

    return korean_mbti_dic


def cleansing_mbti_keywords(sentence):
    # E - I | N - S | F - T | J - P (alphabet order)
    MBTI_UPPER_LIST = ["ENFJ", "ENFP", "ENTJ", "ENTP", "ESFJ", "ESFP", "ESTJ",
                       "ESTP", "INFJ", "INFP", "INTJ", "INTP", "ISFJ", "ISFP", "ISTJ", "ISTP"]
    MBTI_KEYWORDS = get_mbti_keywords()

    # searching English mbti substring
    for mbti in MBTI_UPPER_LIST:
        sentence = sentence.replace(mbti, mbti.lower())

    # if not, searching Korean mbti keyword
    for keyword in MBTI_KEYWORDS.keys():
        sentence = sentence.replace(keyword, MBTI_KEYWORDS[keyword])

    # if there is no keywords, return None
    return sentence


def pattern_cleansing(sentence):
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    sentence = pattern.sub(' ', sentence)
    sentence = url_pattern.sub('', sentence)
    sentence = sentence.strip()
    sentence = repeat_normalize(sentence, num_repeats=2)

    return sentence


def clean_aihub_dataset():
    print("Cleansing Aihub dataset...\n")
    dir = '.\\data\\aihub\\instagram'
    output_path = './data/aihub/instagram_cleaned.jsonl'

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
                    if i == len(lines) - 1:
                        break
                    q = pattern_cleansing(lines[i]["norm_text"].replace(
                        "키키 ", "").replace(" 키키", ""))
                    qa = lines[i]["speechAct"].replace(
                        "(", "").replace(")", "")

                    r = pattern_cleansing(
                        lines[i + 1]["norm_text"].replace("키키 ", "").replace(" 키키", ""))
                    ra = lines[i +
                               1]["speechAct"].replace("(", "").replace(")", "")

                    conversation = {'query': q, 'q_act': qa,
                                    'response': r, 'r_act': ra}
                    conversations.append(conversation)
            print("done")

        except FileNotFoundError:
            print("Sorry! File not found!\n")

    with open(output_path, 'w', encoding='utf-8') as kakao_cleaned:
        json.dump(conversations, kakao_cleaned, indent=4, ensure_ascii=False)

    print("Completed Cleansing Aihub dataset queries and reponses...\n")


def clean_mbti_dataset(file_name):
    print(f"Cleansing {file_name} dataset...\n")

    if file_name == "qna.tsv":
        data = pd.read_csv(f'./data/mbti/{file_name}', sep='\t')

    elif file_name == "multiple_qna.tsv":
        data = pd.read_csv(f'./data/mbti/{file_name}', sep='\t', names=[
                           'id', 'article_id', 'menu_id', 'question', 'answer', 'q_mbti', 'a_mbti'])

    for idx in range(len(data)):
        q = data.loc[idx, 'question']
        a = data.loc[idx, 'answer']

        if is_proper_length(q) and is_proper_length(a):
            try:
                q = spell_checker.check(q).checked
                a = spell_checker.check(a).checked

                q = cleansing_mbti_keywords(q)
                a = cleansing_mbti_keywords(a)

                data.loc[idx, 'question'] = pattern_cleansing(q)
                data.loc[idx, 'answer'] = pattern_cleansing(a)
            except:
                data.drop(index=idx, inplace=True)
                continue
        else:
            data.drop(index=idx, inplace=True)

        if idx % 1000 == 0:
            print(f"{idx}'s rows cleaning done")

    if file_name == "qna.tsv":
        data = data.drop(data.columns[0], axis=1)
        data.to_csv('./data/mbti/qna_cleaned.tsv', sep='\t')
    elif file_name == "multiple_qna.tsv":
        data.to_csv('./data/mbti/multiple_qna_cleaned.tsv', sep='\t', header=[
                    'id', 'article_id', 'menu_id', 'question', 'answer', 'q_mbti', 'a_mbti'])

    print(f"Completed Cleansing {file_name} dataset queries and reponses...\n")


if __name__ == '__main__':
    # clean_aihub_dataset()
    # clean_mbti_dataset('qna.tsv')
    # clean_mbti_dataset('multiple_qna.tsv')
