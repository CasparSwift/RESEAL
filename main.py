import argparse
import os
import torch
import torch.nn as nn
import json
import sys
import random
from pathlib import Path
from data import make_dataloader
from trainers import trainer_factory
from testers import tester_factory
import pdb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig


def set_random_seeds(args):
    seed = float(args.random_seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model_name(args):
    if args.dataset in ['depparse', 'webedit_rel']:
        return 'lstm'
    elif args.pretrain_model in ['lstm', 'gru', 'dual']:
        return args.pretrain_model
    elif 'bert' in args.pretrain_model:
        return 'bert'
    elif 'bart' in args.pretrain_model:
        return 'bart'
    elif 'T5' in args.pretrain_model or 't5' in args.pretrain_model:
        return 't5'
    elif 'gpt' in args.pretrain_model:
        return 'gpt'
    else:
        raise NotImplementedError


def parse_train_args():
    parser = argparse.ArgumentParser("Constrained Decoding")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    
    # model setting
    parser.add_argument("--pretrain_model", type=str, default="facebook/bart-large")
    parser.add_argument("--hidden_size", type=int, default=1024)
    
    # training and optimizer setting
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--random_seed", type=float, default=1024)

    # dataset setting
    parser.add_argument("--dataset", type=str, choices=['commongen', 'gigaword', 
        'english-ewt', 'SR_en_ewt', 'depparse', 'Giga_Dep', 'Giga_Dep_Pred', 'webnlg', 'webedit', 
        'webedit_lstm', 'webedit_rel', 'rotoedit_lstm', 'webnlg_rel'])
    parser.add_argument("--root", type=str, default='./data')
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--max_src_length", type=int, default=128)
    parser.add_argument("--max_word_length", type=int, default=100)
    parser.add_argument("--max_tgt_length", type=int, default=64)
    parser.add_argument("--dep_giga_path", type=str)
    parser.add_argument("--giga_test_set", type=str, default='origin')

    # logging setting
    parser.add_argument("--output_dir", type=str, default="output")

    # tester
    parser.add_argument("--tester", type=str, default="base")
    parser.add_argument("--length_penalty", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--lamb", type=float, default=5.0, help="reward for dep_dba")
    parser.add_argument("--rho", type=float, default=0.5, help="prob threshold for dep_dba")
    parser.add_argument("--alpha_func", type=str, default='1')
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--bank_count", type=str, choices=['word+dep', 'dep', 'word'])
    parser.add_argument("--force_complete", action="store_true")
    parser.add_argument("--reset_avail_states", type=bool, default=True)
    parser.add_argument("--denoise", type=bool, default=False)
    parser.add_argument("--completion", type=bool, default=True)
    parser.add_argument("--unconditional", type=bool, default=False)
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--test_when_training", action="store_true")

    parser.add_argument("--prune_size", type=int, default=20)
    parser.add_argument("--parser_type", type=str, choices=['stanza', 'l2r', 'rel_checker'], default='stanza')
    parser.add_argument("--scorer", type=str, default='1')
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--deppred_threshold", type=float, default=0.5)

    args = parser.parse_args()
    return args


def set_up_model(args, model_name, device):
    if args.dataset == 'depparse':
        from parsers.bilstm import TransitionBasedDependencyParsing
        model = TransitionBasedDependencyParsing(training=not args.test)
    elif args.dataset in ['webedit_lstm', 'rotoedit_lstm']:
        from dependency_prediction.fact_editor import FactBaseDualModel
        if model_name == 'dual':
            if args.dataset == 'webedit_lstm':
                model = FactBaseDualModel(vocab_dim=4176, embed_dim=300, enc_hid_dim=300, dec_hid_dim=600, attention_hid_dim=300)
            else:
                model = FactBaseDualModel(vocab_dim=7337, embed_dim=100, enc_hid_dim=100, dec_hid_dim=200, attention_hid_dim=200)
        else:
            raise NotImplementedError
    elif args.dataset == 'webedit_rel':
        from dependency_prediction.fact_editor import FactRelModel
        model = FactRelModel()
    elif args.dataset == 'webnlg_rel':
        from dependency_prediction.rel_checker import WebNlgRelModel
        model = WebNlgRelModel()
    elif args.dataset == 'Giga_Dep_Pred':
        from dependency_prediction.models import DepPredBertModel
        model = DepPredBertModel(args)
    # use pretrain model
    else:
        if not args.test:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrain_model)
        else:
            tester_name = args.tester.split('_')[0]
            config = AutoConfig.from_pretrained(args.pretrain_model)
            model = AutoModelForSeq2SeqLM.from_config(config)
    
    if args.dataset == 'webnlg':
        from t5_wrapper import T5Wrapper
        model = T5Wrapper(model)
    # if args.dataset == 'gigaword' or args.dataset == 'Giga_Dep':
    #     from t5_wrapper import BARTWrapper
    #     model = BARTWrapper(model)
    return model


def set_up_tokenizer(args, model):
    if args.dataset == 'depparse':
        from tokenizers.tokenizer import GloveTokenizer
        tokenizer = GloveTokenizer('parsers/vocab.txt')
    elif args.dataset in ['webedit_lstm', 'webedit_rel', 'rotoedit_lstm']:
        from tokenizers.tokenizer import SimpleTokenizer
        if 'webedit' in args.dataset:
            tokenizer = SimpleTokenizer('tokenizers/webedit_vocab.txt')
        else:
            tokenizer = SimpleTokenizer('tokenizers/rotoedit_vocab.txt')
    elif args.dataset == 'webnlg_rel':
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
        if args.dataset == 'webnlg' or args.dataset == 'webedit':
            new_tokens = ['<H>', '<R>', '<T>']
            new_tokens_vocab = {}
            new_tokens_vocab['additional_special_tokens'] = []
            for idx, t in enumerate(new_tokens):
                new_tokens_vocab['additional_special_tokens'].append(t)
            num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)
            model.resize_token_embeddings(len(tokenizer))
            print(len(tokenizer))
            print('We have added %s tokens', num_added_toks)
    return tokenizer


def model_to_device(model, device):
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    else:
        print('no parallel')

    print('model to device...')
    model = model.to(device)
    return model


def main():
    args = parse_train_args()
    print(args)
    set_random_seeds(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device = torch.device("cpu" if args.use_cpu else "cuda")
    model_name = get_model_name(args)
    tester_name = args.tester.split('_')[0]
    if args.dataset == 'Giga_Dep':
        model_dir = Path(args.output_dir) / f'{model_name}_gigaword'
    else:
        model_dir = Path(args.output_dir) / f'{model_name}_{args.dataset}'

    # Training
    if not args.test:
        model = set_up_model(args, model_name, device)
        tokenizer = set_up_tokenizer(args, model)
        model = model_to_device(model, device)
        train_loader, test_loader, train_set, test_set = make_dataloader(args, tokenizer)

        trainer = trainer_factory[args.dataset](args, model, train_loader)
        tester = tester_factory[tester_name](args, model, test_loader, test_set, tokenizer)
        for epoch in range(1, args.epoch_num + 1):
            trainer.train_one_epoch(device, epoch)
            output_infer_file_path = f'{model_dir}_e{epoch}_infer_{args.tester}.out'
            if args.test_when_training:
                tester.test_one_epoch(device, output_infer_file_path)
                print(f'test output {output_infer_file_path}')
            torch.save(model.state_dict(), f'{model_dir}_e{epoch}.pt')
            print(f'save to {model_dir}_e{epoch}.pt !')
            if args.debug:
                break

    # Testing
    else:
        epoch = args.epoch_num
        output_infer_file_path = f'{model_dir}_e{epoch}_infer_{args.tester}.out'
        if args.ckpt_path:
            ckpt_path = args.ckpt_path
        else:
            ckpt_path = f'{model_dir}_e{epoch}.pt'
        
        # set up model
        if os.path.exists(output_infer_file_path):
            print('infer file exists! remove ...')
            os.remove(output_infer_file_path)
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path)
            state_dict = {k.replace('bart.', '').replace('module.', ''): v for k, v in state_dict.items()}
            model = set_up_model(args, model_name, device)
            tokenizer = set_up_tokenizer(args, model)
            t5_missing_key = 'decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight'
            if t5_missing_key in state_dict and t5_missing_key not in model.state_dict():
                state_dict.pop(t5_missing_key)
            if 'model.' + t5_missing_key in model.state_dict():
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict)
        else:
            print('ckpt_path not exist!')
            exit()

        model = model_to_device(model, device)
        test_loader, test_set = make_dataloader(args, tokenizer, mode='test')

        tester = tester_factory[tester_name](args, model, test_loader, test_set, tokenizer)
        tester.test_one_epoch(device, output_infer_file_path)


if __name__ == '__main__':
    main()


