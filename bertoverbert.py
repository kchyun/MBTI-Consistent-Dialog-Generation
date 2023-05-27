
import os 
import torch
import random as rd
import json 

from argparse import ArgumentParser
from torch.utils.data import DataLoader 
from tqdm import tqdm
from evaluations import eval_distinct 

from xlibs import EncoderDecoderModel 
from xlibs import AdamW 

from transformers import AutoTokenizer, AutoModelForMaskedLM

from datasets.dataset import AIHubDataset, NLIDataset
from datasets.dataset import MBTIDataset 


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Settings
dataset_name = "aihub"
model_path = "beomi/kcbert-base"
checkpoint_path = "./checkpoints/" + dataset_name + "/bertoverbert_10000"
save_model_path = "./checkpoints/" + dataset_name + "/bertoverbert"
dumped_token_path = "./data/" + dataset_name + "/" + dataset_name + "_tokenized/"
total_epoch = 10
print_freq = 200

def init(tokenizer, model):
    # special tokens? 
    # more required options?
    
    model.config.max_length = 32 
    model.config.min_length = 3
    model.config.early_stopping = True 
    model.config.length_penalty = 1.0 
    model.config.decoder_start_token_id=2
    model.config.force_bos_token_to_be_generated=True
    
    return tokenizer, model 

def prepare_aihub_data_batch(batch):
    persona_input_ids = batch['persona']['input_ids']
    persona_attention_mask = batch['persona']['attention_mask']
    persona_type_ids = batch['persona']['token_type_ids'] * 0 + 1

    query_input_ids = batch['query']['input_ids']
    query_attention_mask = batch['query']['attention_mask']
    query_type_ids = batch['query']['token_type_ids'] * 0

    input_ids = torch.cat([persona_input_ids, query_input_ids], -1)
    attention_mask = torch.cat([persona_attention_mask, query_attention_mask], -1)
    type_ids = torch.cat([persona_type_ids, query_type_ids], -1)

    decoder_input_ids = batch['response']['input_ids']
    decoder_attention_mask = batch['response']['attention_mask']
    mask_flag = torch.Tensor.bool(1 - decoder_attention_mask)
    lables = decoder_input_ids.masked_fill(mask_flag, -100)

    return input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids

def prepare_mbti_data_batch(batch):
    persona_input_ids = batch['persona']['input_ids']
    persona_attention_mask = batch['persona']['attention_mask']
    persona_type_ids = batch['persona']['token_type_ids'] * 0 + 1

    query_input_ids = batch['query']['input_ids']
    query_attention_mask = batch['query']['attention_mask']
    query_type_ids = batch['query']['token_type_ids'] * 0

    input_ids = torch.cat([persona_input_ids, query_input_ids], -1)
    attention_mask = torch.cat([persona_attention_mask, query_attention_mask], -1)
    type_ids = torch.cat([persona_type_ids, query_type_ids], -1)

    decoder_input_ids = batch['response']['input_ids']
    decoder_attention_mask = batch['response']['attention_mask']
    mask_flag = torch.Tensor.bool(1 - decoder_attention_mask)
    lables = decoder_input_ids.masked_fill(mask_flag, -100)

    return input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids

def prepare_inference_batch(pos_batch, neg_batch):
    pos_pre_input_ids = pos_batch['pre']['input_ids']
    pos_pre_attention_mask = pos_batch['pre']['attention_mask']
    pos_pre_type_ids = pos_batch['pre']['token_type_ids'] * 0 + 1

    pos_hyp_input_ids = pos_batch['hyp']['input_ids']
    pos_hyp_attention_mask = pos_batch['hyp']['attention_mask']
    pos_hyp_type_ids = pos_batch['hyp']['token_type_ids'] * 0

    neg_pre_input_ids = neg_batch['pre']['input_ids']
    neg_pre_attention_mask = neg_batch['pre']['attention_mask']
    neg_pre_type_ids = neg_batch['pre']['token_type_ids'] * 0 + 1

    neg_hyp_input_ids = neg_batch['hyp']['input_ids']
    neg_hyp_attention_mask = neg_batch['hyp']['attention_mask']
    neg_hyp_type_ids = neg_batch['hyp']['token_type_ids'] * 0

    return pos_pre_input_ids, pos_pre_attention_mask, pos_pre_type_ids, pos_hyp_input_ids, pos_hyp_attention_mask, pos_hyp_type_ids, neg_pre_input_ids, neg_pre_attention_mask, neg_pre_type_ids, neg_hyp_input_ids, neg_hyp_attention_mask, neg_hyp_type_ids

def prepare_inference_dict(pos_batch, neg_batch):
    pos_pre_input_ids, pos_pre_attention_mask, pos_pre_type_ids, pos_hyp_input_ids, pos_hyp_attention_mask, pos_hyp_type_ids, neg_pre_input_ids, neg_pre_attention_mask, neg_pre_type_ids, neg_hyp_input_ids, neg_hyp_attention_mask, neg_hyp_type_ids = prepare_inference_batch(
        pos_batch, neg_batch)
    return {'pos_pre_input_ids': pos_pre_input_ids, 'pos_pre_attention_mask': pos_pre_attention_mask,
            'pos_pre_type_ids': pos_pre_type_ids, 'pos_hyp_input_ids': pos_hyp_input_ids,
            'pos_hyp_attention_mask': pos_hyp_attention_mask, 'pos_hyp_type_ids': pos_hyp_type_ids,
            'neg_pre_input_ids': neg_pre_input_ids, 'neg_pre_attention_mask': neg_pre_attention_mask,
            'neg_pre_type_ids': neg_pre_type_ids, 'neg_hyp_input_ids': neg_hyp_input_ids,
            'neg_hyp_attention_mask': neg_hyp_attention_mask, 'neg_hyp_attention_mask': neg_hyp_attention_mask,
            'neg_hyp_type_ids': neg_hyp_type_ids}

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
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    tokenizer, model = init(tokenizer, model)
    # tokenize 필요

    # tokenize
    if args.dumped_token is None:
        print('Pre-tokenized files must be provided.')
        raise (ValueError)
    else:
        path = args.dumped_token
        try:
            print(f"Load tokenized dataset from {args.dumped_token}.")

            # Loading training/validation set
            if args.dataset_type == 'aihub':
                with open(path + 'train_qact.json', 'r') as train_qacts:
                    print("Load train_qact")
                    tmp = train_qacts.readline()
                    train_qact = json.loads(tmp)
                with open(path + 'train_ract.json', 'r') as train_racts:
                    print("Load train_ract")
                    tmp = train_racts.readline()
                    train_ract = json.loads(tmp)
                with open(path + 'train_query.json', 'r') as train_query:
                    print("Load train_query")
                    tmp = train_query.readline()
                    train_query_tokenized = json.loads(tmp)
                with open(path + 'train_response.json', 'r') as train_response:
                    print("Load train_response")
                    tmp = train_response.readline()
                    train_response_tokenized = json.loads(tmp)
                    
                with open(path + 'val_qact.json', 'r') as val_qacts:
                    print("Load train_qact")
                    tmp = val_qacts.readline()
                    val_qact = json.loads(tmp)
                with open(path + 'val_ract.json', 'r') as val_racts:
                    print("Load val_ract")
                    tmp = val_racts.readline()
                    val_ract = json.loads(tmp)
                with open(path + 'val_query.json', 'r') as val_query:
                    print("Load val_query")
                    tmp = val_query.readline()
                    val_query_tokenized = json.loads(tmp)
                with open(path + 'val_response.json', 'r') as val_response:
                    print("Load val_response")
                    tmp = val_response.readline()
                    val_response_tokenized = json.loads(tmp)

            elif args.dataset_type == 'mbti':
                with open(path + 'train_mbti.json', 'r') as train_mbti:
                    print("Load train_mbti")
                    tmp = train_mbti.readline()
                    train_mbti_tokenized = json.loads(tmp)
                with open(path + 'train_persona.json', 'r') as train_persona:
                    print("Load train_persona")
                    tmp = train_persona.readline()
                    train_persona_tokenized = json.loads(tmp)
                with open(path + 'train_query.json', 'r') as train_query:
                    print("Load train_query")
                    tmp = train_query.readline()
                    train_query_tokenized = json.loads(tmp)
                with open(path + 'train_response.json', 'r') as train_response:
                    print("Load train_response")
                    tmp = train_response.readline()
                    train_response_tokenized = json.loads(tmp)
                    
                with open(path + 'val_mbti.json', 'r') as val_mbti:
                    print("Load val_mbti")
                    tmp = val_mbti.readline()
                    val_mbti_tokenized = json.loads(tmp)
                with open(path + 'val_persona.json', 'r') as val_persona:
                    print("Load val_persona")
                    tmp = val_persona.readline()
                    val_persona_tokenized = json.loads(tmp)
                with open(path + 'val_query.json', 'r') as val_query:
                    print("Load val_query")
                    tmp = val_query.readline()
                    val_query_tokenized = json.loads(tmp)
                with open(path + 'val_response.json', 'r') as val_response:
                    print("Load val_response")
                    tmp = val_response.readline()
                    val_response_tokenized = json.loads(tmp)
            # NLI
            
            # neutral unused
            with open(path + 'contradiction_pre.json', 'r') as contradiction_pre:
                print("Load contradiction_pre")
                tmp = contradiction_pre.readline()
                contradiction_pre_tokenized = json.loads(tmp)
            with open(path + 'contradiction_hyp.json', 'r') as contradiction_hyp:
                print("Load contradiction_hyp")
                tmp = contradiction_hyp.readline()
                contradiction_hyp_tokenized = json.loads(tmp)

            with open(path + 'entailment_pre.json', 'r') as entailment_pre:
                print("Load entailment_pre")
                tmp = entailment_pre.readline()
                entailment_pre_tokenized = json.loads(tmp)
            with open(path + 'entailment_hyp.json', 'r') as entailment_hyp:
                print("Load entailment_hyp")
                tmp = entailment_hyp.readline()
                entailment_hyp_tokenized = json.loads(tmp)
            
        except FileNotFoundError:
            print(f"Sorry! The files in {args.dumped_token} can't be found.")
            raise ValueError
        
    # prepare dataset
    if args.dataset_type == 'aihub':
        train_dataset = AIHubDataset(train_qact,
                                        train_ract,
                                        train_query_tokenized,
                                        train_response_tokenized,
                                        device)
        val_dataset = AIHubDataset(val_qact,
                                        val_ract,
                                        val_query_tokenized,
                                        val_response_tokenized,
                                        device)
    elif args.dataset_type == 'mbti':
        train_dataset = MBTIDataset(train_mbti,
                                        train_persona_tokenized,
                                        train_query_tokenized,
                                        train_response_tokenized,
                                        device)
        val_dataset = MBTIDataset(val_mbti,
                                    val_persona_tokenized,
                                    val_query_tokenized,
                                    val_response_tokenized,
                                    device)
        
    # neutral unused
    contradiction_nli_dataset = NLIDataset(contradiction_pre_tokenized,
                                            contradiction_hyp_tokenized,
                                            device)
    entailment_nli_dataset = NLIDataset(entailment_pre_tokenized,
                                            entailment_hyp_tokenized,
                                            device)
    
    # training
    print("\nStart Training...")
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
    val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
        
    negative_ul_loader = DataLoader(contradiction_nli_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)
    positive_ul_loader = DataLoader(entailment_nli_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)
    
    p_ul_iterator = enumerate(positive_ul_loader)
    p_ul_len = positive_ul_loader.__len__()
    p_global_step = 0

    n_ul_iterator = enumerate(negative_ul_loader)
    n_ul_len = negative_ul_loader.__len__()
    n_global_step = 0

    optim_warmup = AdamW(model.parameters(), lr=args.warm_up_learning_rate)
    optim = AdamW(model.parameters(), lr=args.learning_rate)

    step = 0
    start_epoch = 0
    for epoch in range(start_epoch, args.total_epochs):
        print('\nTRAINING EPOCH %d' % epoch)
        batch_n = 0

        for batch in train_loader:
            batch_n += 1
            step += 1
            optim_warmup.zero_grad()
            optim.zero_grad()

            if p_global_step >= p_ul_len - 1:
                p_ul_iterator = enumerate(positive_ul_loader)
            if n_global_step >= n_ul_len - 1:
                n_ul_iterator = enumerate(negative_ul_loader)

            p_global_step, pos_batch = next(p_ul_iterator)
            n_global_step, neg_batch = next(n_ul_iterator)

            inference_data_dict = prepare_inference_dict(pos_batch, neg_batch)

            input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_aihub_data_batch(batch) if args.dataset_type == 'aihub' else prepare_mbti_data_batch(batch)

            outputs, outputs_2, ul_outputs = model(input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   decoder_input_ids=decoder_input_ids,
                                                   decoder_attention_mask=decoder_attention_mask,
                                                   labels=lables,
                                                   token_type_ids=type_ids,
                                                   training=True,
                                                   return_dict=True,
                                                   per_input_ids=persona_input_ids,
                                                   ul_training=True,
                                                   inference_dict=inference_data_dict,
                                                   )
            loss = outputs.loss
            loss_2 = outputs_2.loss
            ul_loss = ul_outputs.loss

            loss_prt = loss.cpu().detach().numpy() if torch.cuda.is_available() else loss.detach().numpy()
            loss_2_prt = loss_2.cpu().detach().numpy() if torch.cuda.is_available() else loss_2.detach().numpy()
            ul_loss_prt = ul_loss.cpu().detach().numpy() if torch.cuda.is_available() else ul_loss.detach().numpy()
            loss_prt, loss_2_prt, ul_loss_prt = round(float(loss_prt),3), round(float(loss_2_prt),3), round(float(ul_loss_prt),3)

            if step <= args.warm_up_steps:
                if step % 500 == 0:
                    print(f"warm up step {step}\tLoss: {loss_prt}")
                loss.backward()
                optim_warmup.step()
            else:
                if step % 500 == 0:
                    print(f"train step {step}\tL_nll_d1: {loss_prt}, L_nll_d2: {loss_2_prt} and L_ul: {ul_loss_prt}")
                (loss + 0.01 * loss_2 + 0.01 * ul_loss).backward()
                optim.step()

            if step % args.print_frequency == 0 and not step <= args.warm_up_steps and not args.print_frequency == -1:
                print('Sampling (not final results) ...')
                model.eval()
                for val_batch in val_loader:

                    input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_aihub_data_batch(val_batch) if args.dataset_type == 'aihub' else prepare_mbti_data_batch(val_batch)

                    generated = model.generate(input_ids,
                                               token_type_ids=type_ids,
                                               attention_mask=attention_mask,
                                               per_input_ids=persona_input_ids)
                    generated_2 = model.generate(input_ids,
                                                 token_type_ids=type_ids,
                                                 attention_mask=attention_mask,
                                                 use_decoder2=True,
                                                 per_input_ids=persona_input_ids)
                    generated_token = tokenizer.batch_decode(
                        generated, skip_special_tokens=True)[-5:]
                    generated_token_2 = tokenizer.batch_decode(
                        generated_2, skip_special_tokens=True)[-5:]
                    query_token = tokenizer.batch_decode(
                        query_input_ids, skip_special_tokens=True)[-5:]
                    gold_token = tokenizer.batch_decode(decoder_input_ids,
                                                        skip_special_tokens=True)[-5:]
                    persona_token = tokenizer.batch_decode(
                        persona_input_ids, skip_special_tokens=True)[-5:]
                    if rd.random() < 0.6:
                        for p, q, g, j, k in zip(persona_token, query_token, gold_token, generated_token,
                                                 generated_token_2):
                            print(
                                f"persona: {p[:150]}\nquery: {q[:100]}\ngold: {g[:100]}\nresponse from D1: {j[:100]}\nresponse from D2: {k[:100]}\n")
                        break
                print('\nTRAINING EPOCH %d\n' % epoch)
                model.train()

            if not step <= args.warm_up_steps and step%5000 == 0:
                print(f'Saving model at epoch {epoch} step {step}')
                model.save_pretrained(f"{args.save_model_path}_%d" % step)

def predict(args):
    print("Load tokenized data...\n")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if args.dumped_token is None:
        print('Pre-tokenized files must be provided.')
        raise (ValueError)
    else:
        path = args.dumped_token
        try:
            print(f"Load tokenized dataset from {args.dumped_token}.")
            if args.dataset_type == 'aihub':
                with open(path + 'test_qact.json', 'r') as test_qacts:
                    print("Load test_qact")
                    tmp = test_qacts.readline()
                    test_qact = json.loads(tmp)
                with open(path + 'test_ract.json', 'r') as test_racts:
                    print("Load train_ract")
                    tmp = test_racts.readline()
                    test_ract = json.loads(tmp)
                with open(path + 'test_query.json', 'r') as test_query:
                    print("Load test_query")
                    tmp = test_query.readline()
                    test_query_tokenized = json.loads(tmp)
                with open(path + 'test_response.json', 'r') as test_response:
                    print("Load test_response")
                    tmp = test_response.readline()
                    test_response_tokenized = json.loads(tmp)

            elif args.dataset_type == 'mbti':
                with open(path + 'test_mbti.json', 'r') as test_mbti:
                    print("Load test_mbti")
                    tmp = test_mbti.readline()
                    test_mbti_tokenized = json.loads(tmp)
                with open(path + 'test_persona.json', 'r') as test_persona:
                    print("Load test_persona")
                    tmp = test_persona.readline()
                    test_persona_tokenized = json.loads(tmp)
                with open(path + 'test_query.json', 'r') as test_query:
                    print("Load test_query")
                    tmp = test_query.readline()
                    test_query_tokenized = json.loads(tmp)
                with open(path + 'test_response.json', 'r') as test_response:
                    print("Load test_response")
                    tmp = test_response.readline()
                    test_response_tokenized = json.loads(tmp)
                    
        except FileNotFoundError:
            print(f"Sorry! The files in {args.dumped_token} can't be found.")

    if args.dataset_type == 'aihub':
        test_dataset = AIHubDataset(test_qact,
                                        test_ract,
                                        test_query_tokenized,
                                        test_response_tokenized,
                                        device)
    elif args.dataset_type == 'mbti':
        test_dataset = MBTIDataset(test_mbti,
                                        test_persona_tokenized,
                                        test_query_tokenized,
                                        test_response_tokenized,
                                        device)
    
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Loading Model
    if args.dataset_type == 'aihub':
        model_path = f"./checkpoints/aihub/bertoverbert_{args.eval_epoch}"
    elif args.dataset_type == 'mbti':
        model_path = f"./checkpoints/mbti/bertoverbert_{args.eval_epoch}"
    else:
        print(f"Invalid dataset_type {args.dataset_type}")
        raise (ValueError)
    print("Loading Model from %s" % model_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    tokenizer, model = init(tokenizer, model)

    print(f"Writing generated results to {args.save_result_path}...")

    with open(args.save_result_path, "w", encoding="utf-8") as outf:
        for test_batch in tqdm(test_loader):

            input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_aihub_data_batch(test_batch) if args.dataset_type == 'aihub' else prepare_mbti_data_batch(test_batch)

            generated = model.generate(input_ids,
                                       token_type_ids=type_ids,
                                       attention_mask=attention_mask,
                                       num_beams=args.beam_size,
                                       length_penalty=args.length_penalty,
                                       min_length=args.min_length,
                                       no_repeat_ngram_size=args.no_repeat_ngram_size,
                                       per_input_ids=persona_input_ids)
            generated_2 = model.generate(input_ids,
                                         token_type_ids=type_ids,
                                         attention_mask=attention_mask,
                                         num_beams=args.beam_size,
                                         length_penalty=args.length_penalty,
                                         min_length=args.min_length,
                                         no_repeat_ngram_size=args.no_repeat_ngram_size,
                                         use_decoder2=True,
                                         per_input_ids=persona_input_ids)
            generated_token = tokenizer.batch_decode(
                generated, skip_special_tokens=True)
            generated_token_2 = tokenizer.batch_decode(
                generated_2, skip_special_tokens=True)
            query_token = tokenizer.batch_decode(
                query_input_ids, skip_special_tokens=True)
            gold_token = tokenizer.batch_decode(decoder_input_ids,
                                                skip_special_tokens=True)
            persona_token = tokenizer.batch_decode(
                persona_input_ids, skip_special_tokens=True)
            for p, q, g, r, r2 in zip(persona_token, query_token, gold_token, generated_token, generated_token_2):
                outf.write(f"persona:{p}\tquery:{q}\tgold:{g}\tresponse_from_d1:{r}\tresponse_from_d2:{r2}\n")

def chat(args):
    print("Load tokenized data...\n")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Loading Model
    print("Loading Model from %s" % args.checkpoint)
    model = EncoderDecoderModel.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    tokenizer, model = init(tokenizer, model)

    while(True):
        mbti = input("mbti: ")
        query = input("query: ")
        persona = ''
        response = input("expected response: ")
        
        f_type = ['감사 ', '주장 ', '진술 ']
        t_type = ['단언 ', '주장 ', '지시 ']
        
        n = rd.randint(0,2)
        if "t" in mbti:
            persona += t_type[n]
        elif "f" in mbti:
            persona += f_type[n]
        else:
            continue
        
      #  if(args.mbti_4):
      #      persona += mbti
        
        T_persona = "상황의 이유와 결과가 궁금하며, 해결책을 제시한다. 사실을 바탕으로 이성적이고 논리적으로 이야기한다."
        F_persona = "상대방의 기분이 어떤지 공감, 축하 또는 위로한다. 유연하고 융통성 있게 대처한다." 
        
        if "t" in mbti:
            persona += T_persona
        elif "f" in mbti:
            persona += F_persona
        else:
            continue

        mbti = [mbti]
        persona = [persona]
        query = [query]
        response = [response]

        persona_tokenized = {
            k: v for k, v in tokenizer(
                persona,
                truncation=True,
                padding=True,
                max_length=64
            ).items()
        }
        
        query_tokenized = {
            k: v for k, v in tokenizer(
                query,
                truncation=True,
                padding=True,
                max_length=64
            ).items()
        }

        response_tokenized = {
            k: v for k, v in tokenizer(
                response,
                truncation=True,
                padding=True,
                max_length=64
            ).items()
        }

        test_dataset = MBTIDataset(mbti,
                                    persona_tokenized,
                                    query_tokenized,
                                    response_tokenized,
                                    device)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        print(f"Writing generated results to {args.save_result_path}...")

        with open(args.save_result_path, "w", encoding="utf-8") as outf:
            for test_batch in tqdm(test_loader):

                input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_aihub_data_batch(test_batch) if args.dataset_type == 'aihub' else prepare_mbti_data_batch(test_batch)

                generated = model.generate(input_ids,
                                        token_type_ids=type_ids,
                                        attention_mask=attention_mask,
                                        num_beams=args.beam_size,
                                        length_penalty=args.length_penalty,
                                        min_length=args.min_length,
                                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                                        per_input_ids=persona_input_ids)
                generated_2 = model.generate(input_ids,
                                            token_type_ids=type_ids,
                                            attention_mask=attention_mask,
                                            num_beams=args.beam_size,
                                            length_penalty=args.length_penalty,
                                            min_length=args.min_length,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                                            use_decoder2=True,
                                            per_input_ids=persona_input_ids)
                generated_token = tokenizer.batch_decode(
                    generated, skip_special_tokens=True)
                generated_token_2 = tokenizer.batch_decode(
                    generated_2, skip_special_tokens=True)
                query_token = tokenizer.batch_decode(
                    query_input_ids, skip_special_tokens=True)
                gold_token = tokenizer.batch_decode(decoder_input_ids,
                                                    skip_special_tokens=True)
                persona_token = tokenizer.batch_decode(
                    persona_input_ids, skip_special_tokens=True)
                for p, q, g, r, r2 in zip(persona_token, query_token, gold_token, generated_token, generated_token_2):
                    outf.write(f"persona:{p}\tquery:{q}\tgold:{g}\tresponse_from_d1:{r}\tresponse_from_d2:{r2}\n")

def evaluation(args):
    print("Load tokenized data...\n")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if args.dumped_token is None:
        print('Pre-tokenized files must be provided.')
        raise (ValueError)
    else:
        path = args.dumped_token
        try:
            print(f"Load tokenized dataset from {args.dumped_token}.")
            if args.dataset_type == 'aihub':
                with open(path + 'test_qact.json', 'r') as test_qacts:
                    print("Load test_qact")
                    tmp = test_qacts.readline()
                    test_qact = json.loads(tmp)
                with open(path + 'test_ract.json', 'r') as test_racts:
                    print("Load train_ract")
                    tmp = test_racts.readline()
                    test_ract = json.loads(tmp)
                with open(path + 'test_query.json', 'r') as test_query:
                    print("Load test_query")
                    tmp = test_query.readline()
                    test_query_tokenized = json.loads(tmp)
                with open(path + 'test_response.json', 'r') as test_response:
                    print("Load test_response")
                    tmp = test_response.readline()
                    test_response_tokenized = json.loads(tmp)

            elif args.dataset_type == 'mbti':
                with open(path + 'test_mbti.json', 'r') as test_mbti:
                    print("Load test_mbti")
                    tmp = test_mbti.readline()
                    test_mbti_tokenized = json.loads(tmp)
                with open(path + 'test_persona.json', 'r') as test_persona:
                    print("Load test_persona")
                    tmp = test_persona.readline()
                    test_persona_tokenized = json.loads(tmp)
                with open(path + 'test_query.json', 'r') as test_query:
                    print("Load test_query")
                    tmp = test_query.readline()
                    test_query_tokenized = json.loads(tmp)
                with open(path + 'test_response.json', 'r') as test_response:
                    print("Load test_response")
                    tmp = test_response.readline()
                    test_response_tokenized = json.loads(tmp)

        except FileNotFoundError:
            print(f"Sorry! The file {args.dumped_token} can't be found.")

    if args.dataset_type == 'aihub':
        test_dataset = AIHubDataset(test_qact,
                                        test_ract,
                                        test_query_tokenized,
                                        test_response_tokenized,
                                        device)
    elif args.dataset_type == 'mbti':
        test_dataset = MBTIDataset(test_mbti,
                                        test_persona_tokenized,
                                        test_query_tokenized,
                                        test_response_tokenized,
                                        device)
    
    ppl_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Loading Model
    if args.dataset_type == 'aihub':
        model_path = f"./checkpoints/AIHub/bertoverbert_{args.eval_epoch}"
    elif args.dataset_type == 'mbti':
        model_path = f"./checkpoints/MBTI/bertoverbert_{args.eval_epoch}"
    else:
        print(f"Invalid dataset_type {args.dataset_type}")
        raise (ValueError)

    print("Loading Model from %s" % model_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    tokenizer, model = init(tokenizer, model)
    model.to(device)
    model.eval()

    print('Evaluate perplexity...')
    loss_1 = []
    loss_2 = []
    ntokens = []
    ntokens_2 = []
    n_samples = 0
    for ppl_batch in tqdm(ppl_test_loader):
        input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_aihub_data_batch(ppl_batch) if args.dataset_type == 'aihub' else prepare_mbti_data_batch(ppl_batch)

        with torch.no_grad():

            outputs_1, outputs_2, _ = model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            decoder_input_ids=decoder_input_ids,
                                            decoder_attention_mask=decoder_attention_mask,
                                            labels=lables,
                                            token_type_ids=type_ids,
                                            training=True,
                                            return_dict=True,
                                            per_input_ids=persona_input_ids,
                                            ul_training=False,
                                            inference_dict=None,
                                            )

        if args.ppl_type == 'tokens':
            trg_len = decoder_attention_mask.sum()
            trg_len_2 = decoder_attention_mask.sum()
            log_likelihood_1 = outputs_1.loss * trg_len
            log_likelihood_2 = outputs_2.loss * trg_len_2
            ntokens.append(trg_len)
            ntokens_2.append(trg_len_2)
            loss_1.append(log_likelihood_1)
            loss_2.append(log_likelihood_2)

        elif args.ppl_type == 'sents':
            n_samples += 1
            loss_1.append(torch.exp(outputs_1.loss))
            loss_2.append(torch.exp(outputs_2.loss))
        else:
            print(f"Invalid ppl type {args.ppl_type}")
            raise (ValueError)

    if args.ppl_type == 'tokens':
        ppl_1 = torch.exp(torch.stack(loss_1).sum() / torch.stack(ntokens).sum())
        ppl_2 = torch.exp(torch.stack(loss_2).sum() / torch.stack(ntokens_2).sum())
    elif args.ppl_type == 'sents':
        ppl_1 = torch.stack(loss_1).sum() / n_samples
        ppl_2 = torch.stack(loss_2).sum() / n_samples
    else:
        raise (ValueError)

    print(f"Perplexity on test set is {round(float(ppl_1.cpu().numpy()),3)} and {round(float(ppl_2.cpu().numpy()),3)}."
          ) if torch.cuda.is_available() else (
        f"Perplexity on test set is {round(float(ppl_1.numpy()),3)} and {round(float(ppl_2.numpy()),3)}.")

    if args.word_stat:
        print('Generating...')
        generated_token = []
        generated2_token = []
        gold_token = []
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        with open('evaluations/hyp.txt', 'w') as hyp, open('evaluations/hyp2.txt', 'w') as hyp2, open(
                'evaluations/ref.txt', 'w') as ref:
            for test_batch in tqdm(test_loader):
                input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_aihub_data_batch(test_batch) if args.dataset_type == 'aihub' else prepare_mbti_data_batch(test_batch)
                generated = model.generate(input_ids,
                                           token_type_ids=type_ids,
                                           attention_mask=attention_mask,
                                           num_beams=args.beam_size,
                                           length_penalty=args.length_penalty,
                                           min_length=args.min_length,
                                           no_repeat_ngram_size=args.no_repeat_ngram_size,
                                           use_decoder2=False,
                                           per_input_ids=persona_input_ids)
                generated_2 = model.generate(input_ids,
                                             token_type_ids=type_ids,
                                             attention_mask=attention_mask,
                                             num_beams=args.beam_size,
                                             length_penalty=args.length_penalty,
                                             min_length=args.min_length,
                                             no_repeat_ngram_size=args.no_repeat_ngram_size,
                                             use_decoder2=True,
                                             per_input_ids=persona_input_ids)
                generated_token += tokenizer.batch_decode(generated,
                                                          skip_special_tokens=True)
                generated2_token += tokenizer.batch_decode(generated_2,
                                                           skip_special_tokens=True)
                gold_token += tokenizer.batch_decode(decoder_input_ids,
                                                     skip_special_tokens=True)
            for g, r, r2 in zip(gold_token, generated_token, generated2_token):
                ref.write(f"{g}\n")
                hyp.write(f"{r}\n")
                hyp2.write(f"{r2}\n")

        hyp_d1, hyp_d2 = eval_distinct(generated_token)
        hyp2_d1, hyp2_d2 = eval_distinct(generated2_token)
        ref_d1, ref_d2 = eval_distinct(gold_token)
        print(f"Distinct-1 (hypothesis, hypothesis_2, reference): {round(hyp_d1,4)}, {round(hyp2_d1,4)}, {round(ref_d1,4)}")
        print(f"Distinct-2 (hypothesis, hypothesis_2, reference): {round(hyp_d2,4)}, {round(hyp2_d2,4)}, {round(ref_d2,4)}")
    
    
if __name__ == '__main__':
    parser = ArgumentParser("Transformers EncoderDecoderModel")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--do_chat", action="store_true")
    parser.add_argument("--word_stat", action="store_true")
    parser.add_argument("--use_decoder2", action="store_true")

    parser.add_argument("--train_valid_split", type=float, default=0.1)

    parser.add_argument("--encoder_model", type = str, default = model_path)
    parser.add_argument("--decoder_model", type = str, default = model_path)
    parser.add_argument("--decoder2_model", type = str, default = model_path)
    
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=checkpoint_path)

    parser.add_argument("--max_source_length", type=int, default=64)
    parser.add_argument("--max_target_length", type=int, default=64)

    parser.add_argument("--total_epochs", type=int, default=total_epoch)
    parser.add_argument("--eval_epoch", type=int, default=7)
    parser.add_argument("--print_frequency", type=int, default=print_freq)
    parser.add_argument("--warm_up_steps", type=int, default=6000)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--min_length", type=int, default=3)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warm_up_learning_rate", type=float, default=3e-5)

    parser.add_argument("--save_model_path",
                        type=str,
                        default=save_model_path)
    parser.add_argument("--save_result_path",
                        type=str,
                        default="test_result.tsv")
    parser.add_argument("--dataset_type",
                        type=str,
                        default=dataset_name)
    parser.add_argument("--ppl_type",
                        type=str,
                        default='sents')
    '''
    dumped_token
        aihub:    ./data/aihub/aihub_tokenized/
        mbti:   ./data/mbti/mbti_tokenized/
    '''
    parser.add_argument("--dumped_token",
                        type=str,
                        default=dumped_token_path,
                        required=True)
    args = parser.parse_args()
    
    if args.do_train:
        train(args)
    if args.do_predict:
        predict(args)
    if args.do_chat:
        chat(args)
    if args.do_evaluation:
        evaluation(args)