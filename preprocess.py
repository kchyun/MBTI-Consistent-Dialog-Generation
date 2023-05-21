import json
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForMaskedLM

from sklearn.model_selection import train_test_split
from datasets.dataset import read_nli_split
from datasets.dataset import read_mbti_split
from datasets.dataset import read_aihub_split
from glob import glob
import os
from os.path import dirname, join, basename


# Settings
model_path = "./beomi/kcbert-base"
train_data_path = "./data/aihub/TL_01. KAKAO(4)"
test_data_path = "./data/aihub/TL_01. KAKAO(4)"
nli_data_path = "./data/kor-nlu-datasets/KorNLI/xnli.dev.ko.tsv"
dataset_type = "aihub"


def preprocess_aihub_dataset(args):
    print("Preprocessing...\n")
    dir = args.train.split('/')[-1]
    aihub_files = list(glob(join(args.train, "*.*")))

    train_query_tokenized = {"input_ids": [],
                             "token_type_ids": [], "attention_mask": []}
    val_query_tokenized = {"input_ids": [],
                           "token_type_ids": [], "attention_mask": []}
    test_query_tokenized = {"input_ids": [],
                            "token_type_ids": [], "attention_mask": []}

    train_response_tokenized = {"input_ids": [],
                                "token_type_ids": [], "attention_mask": []}
    val_response_tokenized = {"input_ids": [],
                              "token_type_ids": [], "attention_mask": []}
    test_response_tokenized = {"input_ids": [],
                               "token_type_ids": [], "attention_mask": []}

    train_qact_tokenized = {"input_ids": [],
                            "token_type_ids": [], "attention_mask": []}
    val_qact_tokenized = {"input_ids": [],
                          "token_type_ids": [], "attention_mask": []}
    test_qact_tokenized = {"input_ids": [],
                           "token_type_ids": [], "attention_mask": []}

    train_ract_tokenized = {"input_ids": [],
                            "token_type_ids": [], "attention_mask": []}
    val_ract_tokenized = {"input_ids": [],
                          "token_type_ids": [], "attention_mask": []}
    test_ract_tokenized = {"input_ids": [],
                           "token_type_ids": [], "attention_mask": []}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    for file in aihub_files:

        train_qact, train_query, train_ract, train_response = read_aihub_split(
            file)

        if len(train_qact) <= 1:
            continue
        test_qact, test_query, test_ract, test_response = read_aihub_split(
            file)
        assert len(train_qact) == len(train_query) == len(
            train_ract) == len(train_response)

        train_qact, val_qact, train_query, val_query, train_ract, val_ract, train_response, val_response = train_test_split(
            train_qact,
            train_query,
            train_ract,
            train_response,
            test_size=args.split_rate
        )

        for k, v in tokenizer(
            train_qact,
            truncation=True,
            padding="max_length",
            max_length=args.max_srclen
        ).items():
            train_qact_tokenized[k] += v

        for k, v in tokenizer(
            val_qact,
            truncation=True,
            padding="max_length",
            max_length=args.max_srclen
        ).items():
            val_qact_tokenized[k] += v

        for k, v in tokenizer(
            test_qact,
            truncation=True,
            padding="max_length",
            max_length=args.max_srclen
        ).items():
            test_qact_tokenized[k] += v

        for k, v in tokenizer(
            train_ract,
            truncation=True,
            padding="max_length",
            max_length=args.max_srclen
        ).items():
            train_ract_tokenized[k] += v

        for k, v in tokenizer(
            val_ract,
            truncation=True,
            padding="max_length",
            max_length=args.max_srclen
        ).items():
            val_ract_tokenized[k] += v

        for k, v in tokenizer(
            test_ract,
            truncation=True,
            padding="max_length",
            max_length=args.max_srclen
        ).items():
            test_ract_tokenized[k] += v

        for k, v in tokenizer(
            train_query,
            truncation=True,
            padding="max_length",
            max_length=args.max_srclen
        ).items():
            train_query_tokenized[k] += v

        for k, v in tokenizer(
            val_query,
            truncation=True,
            padding="max_length",
            max_length=args.max_srclen
        ).items():
            val_query_tokenized[k] += v

        for k, v in tokenizer(
            test_query,
            truncation=True,
            padding="max_length",
            max_length=args.max_srclen
        ).items():
            test_query_tokenized[k] += v

        for k, v in tokenizer(
            train_response,
            truncation=True,
            padding="max_length",
            max_length=args.max_tgtlen
        ).items():

            train_response_tokenized[k] += v

        for k, v in tokenizer(
            val_response,
            truncation=True,
            padding="max_length",
            max_length=args.max_tgtlen
        ).items():
            val_response_tokenized[k] += v

        for k, v in tokenizer(
            test_response,
            truncation=True,
            padding="max_length",
            max_length=args.max_tgtlen
        ).items():
            test_response_tokenized[k] += v

        print(file)

    print("Saving Tokenized dict\n")

    path = './data/aihub/aihub_tokenized/'
    if not os.path.exists(os.path.join(path, dir)):
        os.makedirs(os.path.join(path, dir))
    path = path + dir + '/'

    with open(path + 'train_query.json', 'w') as train_query:
        json.dump(train_query_tokenized, train_query)
    with open(path + 'train_response.json', 'w') as train_response:
        json.dump(train_response_tokenized, train_response)

    with open(path + 'val_query.json', 'w') as val_query:
        json.dump(val_query_tokenized, val_query)
    with open(path + 'val_response.json', 'w') as val_response:
        json.dump(val_response_tokenized, val_response)

    with open(path + 'test_query.json', 'w') as test_query:
        json.dump(test_query_tokenized, test_query)
    with open(path + 'test_response.json', 'w') as test_response:
        json.dump(test_response_tokenized, test_response)

    with open(path + 'train_qact.json', 'w') as train_qact:
        json.dump(train_qact_tokenized, train_qact)
    with open(path + 'train_ract.json', 'w') as train_ract:
        json.dump(train_ract_tokenized, train_ract)

    with open(path + 'val_qact.json', 'w') as val_qact:
        json.dump(val_qact_tokenized, val_qact)
    with open(path + 'val_ract.json', 'w') as val_ract:
        json.dump(val_ract_tokenized, val_ract)

    with open(path + 'test_qact.json', 'w') as test_qact:
        json.dump(test_qact_tokenized, test_qact)
    with open(path + 'test_ract.json', 'w') as test_ract:
        json.dump(test_ract_tokenized, test_ract)

    print("Completed Dumping personas, queries and reponses...\n")
    tokenize_nli_dataset(args, path)
    print("Complted Dumping nli dataset...\n")


def tokenize_nli_dataset(args, path):
    print("Tokenize nli data...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    n_pre, n_hyp, c_pre, c_hyp, e_pre, e_hyp = read_nli_split(args.nli)

    n_pre_tokenized = {
        k: v for k, v in tokenizer(
            n_pre,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    n_hyp_tokenized = {
        k: v for k, v in tokenizer(
            n_hyp,
            truncation=True,
            padding=True,
            max_length=args.max_tgtlen
        ).items()
    }

    c_pre_tokenized = {
        k: v for k, v in tokenizer(
            c_pre,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    c_hyp_tokenized = {
        k: v for k, v in tokenizer(
            c_hyp,
            truncation=True,
            padding=True,
            max_length=args.max_tgtlen
        ).items()
    }

    e_pre_tokenized = {
        k: v for k, v in tokenizer(
            e_pre,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    e_hyp_tokenized = {
        k: v for k, v in tokenizer(
            e_hyp,
            truncation=True,
            padding=True,
            max_length=args.max_tgtlen
        ).items()
    }

    with open(path + 'neutral_pre.json', 'w') as n_pre:
        json.dump(n_pre_tokenized, n_pre)
    with open(path + 'neutral_hyp.json', 'w') as n_hyp:
        json.dump(n_hyp_tokenized, n_hyp)

    with open(path + 'contradiction_pre.json', 'w') as c_pre:
        json.dump(c_pre_tokenized, c_pre)
    with open(path + 'contradiction_hyp.json', 'w') as c_hyp:
        json.dump(c_hyp_tokenized, c_hyp)

    with open(path + 'entailment_pre.json', 'w') as e_pre:
        json.dump(e_pre_tokenized, e_pre)
    with open(path + 'entailment_hyp.json', 'w') as e_hyp:
        json.dump(e_hyp_tokenized, e_hyp)


def preprocess_mbti_dataset(args):
    print("Preprocessing...\n")
    # split
    train_mbti, train_persona, train_query, train_response = read_mbti_split(
        args.train, args)  # split
    test_mbti, test_persona, test_query, test_response = read_mbti_split(
        args.test, args)  # split
    assert len(train_persona) == len(train_mbti) == len(
        train_query) == len(train_response)

    train_mbti, val_mbti, train_persona, val_persona, train_query, val_query, train_response, val_response = train_test_split(
        train_mbti,
        train_persona,
        train_query,
        train_response,
        test_size=args.split_rate
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    train_persona_tokenized = {
        k: v for k, v in tokenizer(
            train_persona,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    val_persona_tokenized = {
        k: v for k, v in tokenizer(
            val_persona,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    test_persona_tokenized = {
        k: v for k, v in tokenizer(
            test_persona,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    train_mbti_tokenized = {
        k: v for k, v in tokenizer(
            train_mbti,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    val_mbti_tokenized = {
        k: v for k, v in tokenizer(
            val_mbti,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    test_mbti_tokenized = {
        k: v for k, v in tokenizer(
            test_mbti,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    train_query_tokenized = {
        k: v for k, v in tokenizer(
            train_query,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    val_query_tokenized = {
        k: v for k, v in tokenizer(
            val_query,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    test_query_tokenized = {
        k: v for k, v in tokenizer(
            test_query,
            truncation=True,
            padding=True,
            max_length=args.max_srclen
        ).items()
    }

    train_response_tokenized = {
        k: v for k, v in tokenizer(
            train_response,
            truncation=True,
            padding=True,
            max_length=args.max_tgtlen
        ).items()
    }

    val_response_tokenized = {
        k: v for k, v in tokenizer(
            val_response,
            truncation=True,
            padding=True,
            max_length=args.max_tgtlen
        ).items()
    }

    test_response_tokenized = {
        k: v for k, v in tokenizer(
            test_response,
            truncation=True,
            padding=True,
            max_length=args.max_tgtlen
        ).items()
    }

    print("Saving Tokenized dict\n")
    path = './data/mbti/mbti_tokenized/'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'train_mbti.json', 'w') as train_mbti:
        json.dump(train_mbti_tokenized, train_mbti)
    with open(path + 'train_persona.json', 'w') as train_persona:
        json.dump(train_persona_tokenized, train_persona)
    with open(path + 'train_query.json', 'w') as train_query:
        json.dump(train_query_tokenized, train_query)
    with open(path + 'train_response.json', 'w') as train_response:
        json.dump(train_response_tokenized, train_response)

    with open(path + 'val_mbti.json', 'w') as val_mbti:
        json.dump(val_mbti_tokenized, val_mbti)
    with open(path + 'val_persona.json', 'w') as val_persona:
        json.dump(val_persona_tokenized, val_persona)
    with open(path + 'val_query.json', 'w') as val_query:
        json.dump(val_query_tokenized, val_query)
    with open(path + 'val_response.json', 'w') as val_response:
        json.dump(val_response_tokenized, val_response)

    with open(path + 'test_mbti.json', 'w') as test_mbti:
        json.dump(test_mbti_tokenized, test_mbti)
    with open(path + 'test_persona.json', 'w') as test_persona:
        json.dump(test_persona_tokenized, test_persona)
    with open(path + 'test_query.json', 'w') as test_query:
        json.dump(test_query_tokenized, test_query)
    with open(path + 'test_response.json', 'w') as test_response:
        json.dump(test_response_tokenized, test_response)

    print("Completed Dumping personas, queries and reponses...\n")
    tokenize_nli_dataset(args, path)
    print("Complted Dumping nli dataset...\n")


if __name__ == '__main__':
    parser = ArgumentParser("Preprocessing Dataset")
    parser.add_argument("--split_rate", type=float, default=0.1)
    parser.add_argument("--model_path", type=str, default=model_path)
    parser.add_argument("--train", type=str,
                        default=train_data_path,
                        help="path to train dataset")
    parser.add_argument("--test", type=str,
                        default=test_data_path,
                        help="path to test dataset (valid)")
    parser.add_argument("--nli", type=str,
                        default=nli_data_path,
                        help="path to nli dataset")
    parser.add_argument("--dataset_type", type=str,
                        default=dataset_type)

    parser.add_argument("--max_srclen", type=int,
                        default=64,
                        help="max length of source mbti data")
    parser.add_argument("--max_tgtlen", type=int,
                        default=64,
                        help="max length of target mbti data")

    parser.add_argument("--mbti_4", action="store_true")
    parser.add_argument("--mbti_sent", action="store_true")
    parser.add_argument("--sent_type", action="store_true")

    args = parser.parse_args()

    if args.dataset_type == "mbti":
        preprocess_mbti_dataset(args)
    elif args.dataset_type == "aihub":
        preprocess_aihub_dataset(args)
