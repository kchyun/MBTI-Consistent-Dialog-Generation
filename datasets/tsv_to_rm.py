import json
import pandas as pd
import random

tsv_path = "./data/mbti/multiple_qna.tsv"
output_path = "./data/mbti/mbti_rm_ESTJ.jsonl"
chosen_keyword = "estj"
rejected_keyword = 'f'

df = pd.read_csv(tsv_path, sep='\t', names=[
                 'id', 'article_id', 'menu_id', 'content', 'comment', 'content_mbti', 'comment_mbti'])

output = []
output_idx = 0

current_article_id = df.loc[0]['article_id']

for idx, conversation in df.iterrows():
    # 초항만 예외로 처리
    if idx == 0:
        output.append(
            {'prompt': df.loc[0]['content'], 'chosen': [], 'rejected': []})

        if conversation['comment_mbti'].find(chosen_keyword) != -1:
            output[0]['chosen'].append(conversation['comment'])
        elif conversation['comment_mbti'].find(rejected_keyword) != -1:
            output[0]['rejected'].append(conversation['comment'])

        continue

    if conversation['article_id'] != current_article_id:
        # 해당 article이 끝났다는 의미이므로, chosen 중 랜덤으로 택 1, rejected 중 랜덤으로 택 1
        # 만약 chosen, rejected 중 하나에 아무것도 들어 있지 않은 경우, 그 행은 그냥 전부 삭제
        if len(output[output_idx]['chosen']) > 0 and len(output[output_idx]['rejected']) > 0:
            output[output_idx]['chosen'] = random.choice(
                output[output_idx]['chosen'])
            output[output_idx]['rejected'] = random.choice(
                output[output_idx]['rejected'])
        else:
            del output[output_idx]
            output_idx -= 1

        # 새 article을 저장하기 위해 초기화
        output_idx += 1
        current_article_id = conversation['article_id']
        output.append(
            {'prompt': conversation['content'], 'chosen': [], 'rejected': []})

    # comment_mbti에 t가 들어가 있으면 chosen, 아니면 rejected
    if conversation['comment_mbti'].find(chosen_keyword) != -1:
        output[output_idx]['chosen'].append(conversation['comment'])
    elif conversation['comment_mbti'].find(rejected_keyword) != -1:
        output[output_idx]['rejected'].append(conversation['comment'])

if len(output[output_idx]['chosen']) > 0 and len(output[output_idx]['rejected']) > 0:
    output[output_idx]['chosen'] = random.choice(output[output_idx]['chosen'])
    output[output_idx]['rejected'] = random.choice(
        output[output_idx]['rejected'])
else:
    del output[output_idx]
    output_idx -= 1

with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(output, outfile, indent=4, ensure_ascii=False)
