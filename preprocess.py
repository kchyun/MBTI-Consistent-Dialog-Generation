import json
from argparse import ArgumentParser 
#from xlibs import BartTokenizer, RobertaTokenizer 
#from kobert_tokenizer import KoBERTTokenizer
from kobart import get_kobart_tokenizer
from sklearn.model_selection import train_test_split
from datasets.dataset import read_nli_split 
from datasets.dataset import read_mbti_split

#tokenize해줘야 함 반드시!
def preprocess_aihub_dataset(args):
    print("Preprocessing...\n")
    train_qact, train_query, train_ract, train_response = read_aihub_split()
    
    test_qact, test_query, test_ract, test_response = read_aihub_split()
    assert len(train_qact) == len(train_query) == len(train_ract) == len(train_response)
    
    train_qact, val_qact, train_query, val_query, train_ract, val_ract, train_response, val_response = train_test_split(
        train_qact,
        train_query,
        train_ract,
        train_response,
        test_size = args.split_rate
    )
    
    tokenizer = get_kobart_tokenizer()
    tokenizer = tokenizer.from_pretrained('./pretrained_models/kobart-base-v2')

    #act는 tokenize할 필요X. 단순 레이블링
    train_query_tokenized = {
        k : v for k, v in tokenizer(
            train_query,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    val_query_tokenized = {
        k : v for k, v in tokenizer(
            val_query,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    test_query_tokenized ={
        k : v for k, v in tokenizer(
            test_query,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    train_response_tokenized = {
        k : v for k, v in tokenizer(
            train_response,
            truncation = True,
            padding = True,
            max_length = args.max_tgtlen
        ).items()
    }
    
    val_response_tokenized = {
        k : v for k, v in tokenizer(
            val_response,
            truncation = True,
            padding = True,
            max_length = args.max_tgtlen
        ).items()
    }
    
    test_response_tokenized = {
        k : v for k, v in tokenizer(
            test_response,
            truncation = True,
            padding = True,
            max_length = args.max_tgtlen
        ).items()
    }
    print("Saving Tokenized dict\n")
    path = './data/aihub/aihub_tokenized/'
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
    
    print("Completed Dumping personas, queries and reponses...\n")
    
def tokenize_nli_dataset(args, tokenizer, path):
    print("Tokenize nli data...")
    
    n_pre, n_hyp, c_pre, c_hyp, e_pre, e_hyp = read_nli_split(args.nli)
    
    n_pre_tokenized = {
        k : v for k, v in tokenizer(
            n_pre,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    n_hyp_tokenized = {
        k : v for k, v in tokenizer(
            n_hyp,
            truncation = True,
            padding = True,
            max_length = args.max_tgtlen
        ).items()
    }
    
    
    c_pre_tokenized = {
        k : v for k, v in tokenizer(
            c_pre,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    c_hyp_tokenized = {
        k : v for k, v in tokenizer(
            c_hyp,
            truncation = True,
            padding = True,
            max_length = args.max_tgtlen
        ).items()
    }
    
    e_pre_tokenized = {
        k : v for k, v in tokenizer(
            e_pre,
            truncation = True, 
            padding = True,
            max_length =args.max_srclen
        ).items()
    }
    
    e_hyp_tokenized = {
        k : v for k, v in tokenizer(
            e_hyp,
            truncation = True,
            padding = True,
            max_length = args.max_tgtlen
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
    train_mbti, train_persona, train_query, train_response = read_mbti_split('data/mbti/qna1.tsv') # split
    test_mbti, test_persona, test_query, test_response = read_mbti_split('data/mbti/qna1.tsv') # split 
    assert len(train_persona) == len(train_mbti) == len(train_query) == len(train_response)
    
    train_mbti, val_mbti, train_persona, val_persona, train_query, val_query, train_response, val_response = train_test_split(
        train_mbti,
        train_persona,
        train_query,
        train_response,
        test_size=args.split_rate
    )
    
    tokenizer = get_kobart_tokenizer()
    tokenizer = tokenizer.from_pretrained("./pretrained_models/kobart-base-v2/")
    
    train_persona_tokenized = {
        k : v for k, v in tokenizer(
            train_persona,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    val_persona_tokenized = {
        k : v for k, v in tokenizer(
            val_persona, 
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    test_persona_tokenized = {
        k : v for k, v in tokenizer(
            test_persona,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    train_mbti_tokenized = {
        k : v for k, v in tokenizer(
            train_mbti,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }

    val_mbti_tokenized = {
        k : v for k, v in tokenizer(
            val_mbti,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    test_mbti_tokenized = {
        k : v for k, v in tokenizer(
            test_mbti,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }

    train_query_tokenized = {
        k : v for k, v in tokenizer(
            train_query,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }

    val_query_tokenized = {
        k : v for k, v in tokenizer(
            val_query,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    
    
    test_query_tokenized = {
        k : v for k, v in tokenizer(
            test_query,
            truncation = True,
            padding = True,
            max_length = args.max_srclen
        ).items()
    }
    

    train_response_tokenized = {
        k : v for k, v in tokenizer(
            train_response,
            truncation =  True,
            padding = True,
            max_length = args.max_tgtlen
        ).items()
    }

    
    val_response_tokenized = {
        k :v for k, v in tokenizer(
            val_response,
            truncation = True,
            padding = True,
            max_length = args.max_tgtlen
        ).items()
    }
    
    test_response_tokenized = {
        k : v for k, v in tokenizer(
            test_response,
            truncation = True,
            padding = True,
            max_length = args.max_tgtlen
        ).items()
    }
    
    print("Saving Tokenized dict\n")
    path = './data/mbti/mbti_tokenized/'
        
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
    tokenize_nli_dataset(args, tokenizer, path)
    print("Complted Dumping nli dataset...\n")
    
if __name__ == '__main__':
    parser = ArgumentParser("Preprocessing Dataset")
    parser.add_argument("--split_rate", type = float , default = 0.1)
    parser.add_argument("--train", type = str, 
                        default = "",
                        help = "path to train dataset")
    parser.add_argument("--test", type = str, 
                        default = "",
                        help = "path to test dataset (valid)")
    parser.add_argument("--nli", type = str,
                        default = "data/kor-nlu-datasets/KorNLI/xnli.dev.ko.tsv",
                        help = "path to nli dataset")
    parser.add_argument("--dataset_type", type = str,
                        default = "mbti",
                        required = True)
    parser.add_argument("--max_srclen", type = int,
                        default = 32,
                        help = "max length of source")
    parser.add_argument("--max_tgtlen", type = int,
                        default = 32,
                        help = "max length of target")
    args = parser.parse_args()
    
    if args.dataset_type == "mbti":
        preprocess_mbti_dataset(args)
    elif args.dataset_type == "aihub":
        preprocess_aihub_dataset(args)
    
    