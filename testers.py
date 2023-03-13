import torch
import torch.nn.functional as F
import sys
import os
import torch
from my_utils import AverageMeter, Constraint, get_prefix_allowed_tokens_fn
import copy
from tqdm import tqdm
from data import dataset_factory
from metrics.metric_utils import get_predict_and_all_deps
from parser_api import LeftToRightPointerParser, StanzaParser, RelChecker
from dep_scorer import scorer_factory
import json
import pdb
import re
from utils import get_constraints_seq_labeling
from t5_wrapper import BARTWrapper
from generation_utils import generate
from generation_utils_lstm import my_beam_search
from parser_api import RelCheckerForWebnlg
from utils import get_constraints


def convert_text(text):
    #return text
    text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text

def prepare_constraints(args, dataset, tokenizer):
    all_constraints = []
    for obj in dataset.metadata:
        constraints = []
        for triplet in obj["dependencies"]:
            if args.dataset == 'webnlg':
                if triplet[1] not in constraints:
                    constraints.append(triplet[1])
            else:
                if triplet[0] == triplet[-1]:
                    if triplet[0] not in constraints:
                        constraints.append(triplet[0])
                        constraints.append(triplet[-1])
                else:
                    if triplet[0] not in constraints:
                        constraints.append(triplet[0])
                    if triplet[-1] not in constraints:
                        constraints.append(triplet[-1])          
        if "keywords" in obj:
            for w in obj['keywords']:
                if w not in ['<', '>', 'unk', ',', '$'] and w not in constraints:
                    constraints.append(w)
        
        if constraints:
            if args.dataset == 'webnlg':
                word_constraints = []
                for w in constraints:
                    word_constraints += w.split()
                if 't5' in args.pretrain_model.lower():
                    all_constraints.append(
                        tokenizer(word_constraints, add_special_tokens=False)["input_ids"]
                    )
                else:
                    all_constraints.append(
                        tokenizer(word_constraints, add_special_tokens=False, add_prefix_space=True)["input_ids"]
                    )
            else:
                all_constraints.append(
                    tokenizer(constraints, add_special_tokens=False, add_prefix_space=True)["input_ids"]
                )
        else:
            all_constraints.append([])
    return all_constraints


class BaseTester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer 

    def test_one_epoch(self, device, output_path):
        # loss_meter, nll_loss_meter = AverageMeter(), AverageMeter()
        self.model.eval()
        with open(output_path, 'w') as f:
            for i, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = {k: inputs[k].to(device) for k in inputs}
                # pdb.set_trace()
                output_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs["attention_mask"],
                    num_beams=self.args.beam_size,
                    min_length=0 if self.args.giga_test_set != 'duc' else self.args.max_tgt_length,
                    max_length=self.args.max_tgt_length, 
                    early_stopping=not self.args.no_early_stopping, 
                    length_penalty=self.args.length_penalty,
                    repetition_penalty=self.args.repetition_penalty,
                    # num_return_sequences=self.args.beam_size,
                )
                output = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
                    for g in output_ids]
                for sent in output:
                    if self.args.dataset == 'webnlg':
                        f.write(convert_text(sent) + '\n')
                    else:
                        f.write(sent + '\n')
                # if i % self.args.logging_steps == 0:
                    # print(f'epoch: [1] | batch: [{i}]', file=sys.stderr)
                if self.args.debug:
                    break


class DBATester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.dataset = test_set

    def test_one_epoch(self, device, output_path):
        # get constraints 
        all_lexical_constraints = prepare_constraints(self.args, self.dataset, self.tokenizer)

        # loss_meter, nll_loss_meter = AverageMeter(), AverageMeter()
        self.model.eval()
        with open(output_path, 'w') as f:
            cur_pos = 0
            for i, inputs in tqdm(enumerate(self.test_loader)):
                inputs = {k: inputs[k].to(device) for k in inputs}
                batch_size = inputs['input_ids'].shape[0]

                cur_constraints = all_lexical_constraints[cur_pos: cur_pos + batch_size]

                model_kwargs = {
                    'scorer': None
                }

                output_ids = generate(self.model,
                    input_ids=inputs['input_ids'], 
                    num_beams=self.args.beam_size, 
                    min_length=0 if self.args.giga_test_set != 'duc' else self.args.max_tgt_length,
                    max_length=self.args.max_tgt_length, 
                    early_stopping=self.args.early_stopping, 
                    length_penalty=self.args.length_penalty,
                    repetition_penalty=self.args.repetition_penalty,
                    constraints=cur_constraints,
                    **model_kwargs
                )

                output = [self.tokenizer.decode(g, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False) for g in output_ids]
                
                for j in range(cur_pos, cur_pos + batch_size):
                    sent, cons = output[j - cur_pos], all_lexical_constraints[j]
                    if self.args.dataset == 'webnlg':
                        f.write(convert_text(sent) + '\n')
                    else:
                        f.write(sent + '\n')

                # if i % self.args.logging_steps == 0:
                #     print(f'epoch: [1] | batch: [{i}]', file=sys.stderr)
                if self.args.debug:
                    break

                cur_pos += batch_size


class DDBATester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.dataset = test_set

    def test_one_epoch(self, device, output_path):
        # get constraints 
        all_lexical_constraints = prepare_constraints(self.args, self.dataset, self.tokenizer)

        self.model.eval()
        with open(output_path, 'w') as f:
            cur_pos = 0
            for i, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = {k: inputs[k].to(device) for k in inputs}
                batch_size = inputs['input_ids'].shape[0]

                cur_constraints = all_lexical_constraints[cur_pos: cur_pos + batch_size]

                model_kwargs = {
                    'scorer': None,
                    'partial': True,
                    'partial_top_k': 5,
                    'partial_top_p': None,
                    'partial_min_score': None,
                    'partial_score_mode': 0
                }

                output_ids = generate(self.model,
                    input_ids=inputs['input_ids'], 
                    num_beams=self.args.beam_size, 
                    max_length=self.args.max_tgt_length, 
                    early_stopping=self.args.early_stopping, 
                    length_penalty=self.args.length_penalty,
                    repetition_penalty=self.args.repetition_penalty,
                    constraints=cur_constraints,
                    **model_kwargs
                )

                output = [self.tokenizer.decode(g, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False) for g in output_ids]
                
                for sent, cons in zip(output, cur_constraints):
                    print(cons)
                    print(sent)
                    if self.args.dataset == 'webnlg':
                        f.write(convert_text(sent) + '\n')
                    else:
                        f.write(sent + '\n')

                # if i % self.args.logging_steps == 0:
                #     print(f'epoch: [1] | batch: [{i}]', file=sys.stderr)
                if self.args.debug:
                    break

                cur_pos += batch_size


class DependencyConstrainedTester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.dataset = test_set

        if self.args.parser_type == 'stanza':
            self.parser = StanzaParser()
        elif self.args.parser_type == 'l2r':
            self.parser = LeftToRightPointerParser()
        else:
            exit(f'{self.args.parser_type} not exist!')

    def test_one_epoch(self, device, output_path):
        # get constraints 
        all_lexical_constraints = []
        all_dep_constraints = []
        all_src_deps = []
        for obj in self.dataset.metadata:
            constraints = []
            dep_constraints = []
            for triplet in obj["dependencies"]:
                if '_ROOT' in triplet:
                    continue
                if triplet[0] == triplet[-1]:
                    constraints.append(triplet[0])
                    constraints.append(triplet[-1])
                else:
                    if triplet[0] not in constraints:
                        constraints.append(triplet[0])
                    if triplet[-1] not in constraints:
                        constraints.append(triplet[-1])
                dep_constraints.append(triplet)
            if "keyphrases" in obj:
                for w in obj['keyphrases']:
                    constraints.append(w)
            elif "keywords" in obj:
                for w in obj['keywords']:
                    if w not in ['<', '>', 'unk', ',', '$'] and w not in constraints:
                        constraints.append(w)
            all_lexical_constraints.append(constraints)
            all_dep_constraints.append(dep_constraints)
            all_src_deps.append(obj.get('src_deps', []))

        self.model.eval()
        with open(output_path, 'w') as f:
            cur_pos = 0
            for i, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = {k: inputs[k].to(device) for k in inputs}
                batch_size = inputs['input_ids'].shape[0]

                if not isinstance(self.model, BARTWrapper):
                    cur_constraints = []
                    indice1, indice2 = [], []
                    for j in range(cur_pos, cur_pos + batch_size):
                        # For Bart/Roberta tokenizers which use byte-level BPE, add an additional space 
                        if all_lexical_constraints[j]:
                            # print([' ' + s for s in all_lexical_constraints[j]])
                            cur_constraints.append(
                                self.tokenizer([' ' + s for s in all_lexical_constraints[j]], 
                                    add_special_tokens=False)["input_ids"]
                            )
                            indice1.append(j - cur_pos)
                        else:
                            print(1)
                            cur_constraints.append([])
                            indice2.append(j - cur_pos)
                    cur_dep_constraints = all_dep_constraints[cur_pos: cur_pos + batch_size]
                else:
                    cons_prob = self.model.get_constraints_words(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        decoder_input_ids=inputs['decoder_input_ids']
                    )
                    cons_label = (cons_prob > 0.8).long()
                    cur_raw_words, cur_constraints, cur_dep_constraints = get_constraints_seq_labeling(
                        inputs, cons_label, self.tokenizer, all_src_deps[cur_pos: cur_pos + batch_size])
                    indice1, indice2 = [], []
                    for j in range(batch_size):
                        if cur_constraints[j]:
                            indice1.append(j)
                        else:
                            indice2.append(j)

                if indice1:
                    model_kwargs = {
                        'prune_size': self.args.prune_size,
                        'parser': self.parser,
                        'tokenizer': self.tokenizer,
                        'lamb': self.args.lamb,
                        'scorer': self.args.scorer,
                        'rho': self.args.rho,
                        'alpha_func': self.args.alpha_func,
                        'normalize': self.args.normalize,
                        'bank_count': self.args.bank_count,
                        'force_complete': self.args.force_complete,
                        'reset_avail_states': self.args.reset_avail_states,
                        'denoise': self.args.denoise,
                        'completion': self.args.completion,
                        # 'partial': True,
                        # 'partial_top_k': 5
                    }
                
                    output_ids_1 = generate(self.model,
                        input_ids=inputs['input_ids'][indice1],
                        attention_mask=inputs["attention_mask"][indice1],
                        num_beams=self.args.beam_size, 
                        min_length=1 if self.args.giga_test_set != 'duc' else self.args.max_tgt_length,
                        max_length=self.args.max_tgt_length, 
                        early_stopping=not self.args.no_early_stopping, 
                        length_penalty=self.args.length_penalty,
                        repetition_penalty=self.args.repetition_penalty,
                        constraints=[cur_constraints[idx] for idx in indice1],
                        dep_constraints=[cur_dep_constraints[idx] for idx in indice1],
                        **model_kwargs
                    )

                if indice2:
                    output_ids_2 = self.model.generate(
                        input_ids=inputs['input_ids'][indice2], 
                        attention_mask=inputs["attention_mask"][indice2],
                        num_beams=5, 
                        min_length=10 if self.args.giga_test_set != 'duc' else self.args.max_tgt_length,
                        max_length=self.args.max_tgt_length, 
                        early_stopping=not self.args.no_early_stopping, 
                        length_penalty=self.args.length_penalty,
                        repetition_penalty=self.args.repetition_penalty
                    )

                output_ids = []
                for idx in range(batch_size):
                    if idx in indice1:
                        output_ids.append(output_ids_1[indice1.index(idx)].tolist())
                    else:
                        output_ids.append(output_ids_2[indice2.index(idx)].tolist())

                output = [self.tokenizer.decode(g, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False) for g in output_ids]

                for j in range(cur_pos, cur_pos + batch_size):
                    sent = output[j - cur_pos]
                    print(cur_dep_constraints[j - cur_pos], sent)
                    f.write(sent + '\n')
                if self.args.debug:
                    break
                cur_pos += batch_size
        print('save to:', output_path)


class ReRankTester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.dataset = test_set

        self.parser = LeftToRightPointerParser()

    def test_one_epoch(self, device, output_path):
        # get constraints 
        all_lexical_constraints = []
        all_dep_constraints = []
        all_refs = []
        for obj in self.dataset.metadata:
            constraints = []
            dep_constraints = []
            for triplet in obj["dependencies"]:
                if triplet[0] == triplet[-1]:
                    constraints.append(triplet[0])
                    constraints.append(triplet[-1])
                else:
                    if triplet[0] not in constraints:
                        constraints.append(triplet[0])
                    if triplet[-1] not in constraints:
                        constraints.append(triplet[-1])
                dep_constraints.append(triplet)
            if "keywords" in obj:
                for w in obj['keywords']:
                    if w not in ['<', '>', 'unk', ',', '$'] and w not in constraints:
                        constraints.append(w)
            all_lexical_constraints.append(constraints)
            all_dep_constraints.append(dep_constraints)
            all_refs.append(obj['ref'] if 'ref' in obj else obj['text'])


        # loss_meter, nll_loss_meter = AverageMeter(), AverageMeter()
        self.model.eval()
        with open(output_path, 'w') as f:
            cur_pos = 0
            for i, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = {k: inputs[k].to(device) for k in inputs}
                batch_size = inputs['input_ids'].shape[0]

                cur_constraints = []
                for j in range(cur_pos, cur_pos + batch_size):
                    # For Bart/Roberta tokenizers which use byte-leval BPE, add an additional space 
                    cur_constraints.append(
                        self.tokenizer(all_lexical_constraints[j], add_special_tokens=False, 
                            add_prefix_space=True)["input_ids"]
                    )
                
                output_ids = self.model.generate(
                    input_ids=inputs['input_ids'], 
                    num_beams=self.args.beam_size, 
                    max_length=self.args.max_tgt_length, 
                    early_stopping=True, 
                    length_penalty=self.args.length_penalty,
                    repetition_penalty=self.args.repetition_penalty,
                    constraints=cur_constraints,
                    num_return_sequences=self.args.beam_size
                )

                top1_outputs = []
                for j in range(batch_size):
                    output = [self.tokenizer.decode(g, skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False) 
                        for g in output_ids[j*self.args.beam_size: (j+1)*self.args.beam_size]]
                    # in_docs = [stanza.Document([], text=s) for s in output]

                    gold_deps = all_dep_constraints[j+cur_pos]
                    word_pairs = [(w1, w2) for w1, r, w2 in gold_deps]

                    max_tp = -1
                    top1_output = None
                    # for idx, doc in enumerate(self.nlp(in_docs)):
                    all_pred_deps = self.parser.get_all_deps(output)
                    # predict_deps, all_deps = get_predict_and_all_deps(doc, word_pairs)
                    for idx, all_deps in enumerate(all_pred_deps):
                        predict_deps = []
                        for w1, r, w2 in all_deps:
                            if (w1, w2) in word_pairs:
                                predict_deps.append([w1, r, w2])
                        tp = 0
                        for gold_dep in gold_deps:
                            if gold_dep in predict_deps:
                                tp += 1
                        if tp > max_tp:
                            max_tp = tp
                            top1_output = output[idx]
                    top1_outputs.append(top1_output)
                
                for j in range(cur_pos, cur_pos + batch_size):
                    sent, cons = top1_outputs[j - cur_pos], all_lexical_constraints[j]
                    print(cons, sent, output_ids[j - cur_pos].tolist())
                    f.write(sent + '\n')

                if i % self.args.logging_steps == 0:
                    print(f'epoch: [1] | batch: [{i}]', file=sys.stderr)
                if self.args.debug:
                    break

                cur_pos += batch_size


class DepPredTester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer 
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.dataset = test_set

    def cal_metrics(self, probs, labels, mask):
        word_sum = mask.sum(-1).tolist()
        gold_labels = labels.tolist()
        tp, total_pred, total = 0, 0, 0
        for s, prob, gold_label in zip(word_sum, probs, gold_labels):
            for p, gold in zip(prob[:s], gold_label[:s]):
                pred = int(p > self.args.deppred_threshold)
                if pred == gold == 1:
                    tp += 1
                if pred == 1:
                    total_pred += 1
                if gold == 1:
                    total += 1
        return tp, total_pred, total

    def get_f1(self, tp, total_pred, total):
        precision = tp / (total_pred + 1e-10)
        recall = tp / (total + 1e-10)
        F1 = 2 * precision * recall / (precision + recall + 1e-10)  
        return precision, recall, F1

    def test_one_epoch(self, device, output_path):
        # loss_meter = AverageMeter()
        self.model.eval()
        tp, total_pred, total = 0, 0, 0
        tp1, total_pred1, total1 = 0, 0, 0
        cur_pos = 0
        with open(output_path, 'w') as f:
            for i, inputs in enumerate(self.test_loader):
                # print(inputs)
                # train batch
                inputs = {k: inputs[k].to(device) for k in inputs}
                batch_size = inputs['input_ids'].shape[0]

                logits, keyword_logits = self.model(**inputs)
                probs = torch.softmax(logits, dim=-1)[:, :, 1].tolist()
                keyword_probs = torch.softmax(keyword_logits, dim=-1)[:, :, 1].tolist()
                x, y, z = self.cal_metrics(probs, inputs['labels'], inputs['word_cnt_mask'])
                tp += x
                total_pred += y
                total += z
                precision, recall, F1 = self.get_f1(tp, total_pred, total)
                x, y, z = self.cal_metrics(keyword_probs, inputs['keyword_labels'], inputs['word_cnt_mask'])
                tp1 += x
                total_pred1 += y
                total1 += z
                precision1, recall1, F11 = self.get_f1(tp1, total_pred1, total1)
                # if i % self.args.logging_steps == 0:
                #     print(f'batch: [{i}] '\
                #         f'| p: {precision:.4f} ({tp}/{total_pred})' \
                #         f'| r: {recall:.4f} ({tp}/{total})' \
                #         f'| f: {F1:.4f}' \
                #         f'| k_p: {precision1:.4f} ({tp1}/{total_pred1})' \
                #         f'| k_r: {recall1:.4f} ({tp1}/{total1})' \
                #         f'| k_f: {F11:.4f}', file=sys.stderr)
                if self.args.debug:
                    break
                for j in range(cur_pos, cur_pos + batch_size):
                    obj = self.dataset.metadata[j]
                    entry = {
                        'text': obj['src'],
                        'ref': obj['ref'],
                        'src_deps': obj['src_deps'],
                        'probs': probs[j - cur_pos][:len(obj['src_deps'])],
                        'keyword_probs': keyword_probs[j - cur_pos][:len(obj['src_deps'])]
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                cur_pos += batch_size
            precision, recall, F1 = self.get_f1(tp, total_pred, total)
            precision1, recall1, F11 = self.get_f1(tp1, total_pred1, total1)
            print(f'Overall: '\
                f'| p: {precision:.4f} ({tp}/{total_pred})' \
                f'| r: {recall:.4f} ({tp}/{total})' \
                f'| f: {F1:.4f}' \
                f'| k_p: {precision1:.4f} ({tp1}/{total_pred1})' \
                f'| k_r: {recall1:.4f} ({tp1}/{total1})' \
                f'| k_f: {F11:.4f}', file=sys.stderr)


class BaseLSTMTester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer 

    def test_one_epoch(self, device, output_path):
        self.model.eval()
        with open(output_path, 'w') as f:
            for i, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = {k: inputs[k].to(device) for k in inputs}
                # pdb.set_trace()
                output_ids = my_beam_search(self.model, device, 
                    inputs=inputs, 
                    num_beams=self.args.beam_size, 
                    max_length=self.args.max_tgt_length, 
                    early_stopping=True, 
                    length_penalty=self.args.length_penalty,
                    repetition_penalty=self.args.repetition_penalty,
                    bos_token_id=self.tokenizer(['<bos>'])[0], 
                    pad_token_id=self.tokenizer(['<pad>'])[0], 
                    eos_token_id=self.tokenizer(['<eos>'])[0]
                )
                output = [self.tokenizer.decode(g) for g in output_ids]
                for sent in output:
                    # print(sent)
                    if self.args.dataset == 'webnlg':
                        f.write(convert_text(sent) + '\n')
                    else:
                        f.write(sent + '\n')
                if self.args.debug:
                    break


class RelPredTester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer 
    
    def get_f1(self, tp, total_pred, total):
        precision = tp / (total_pred + 1e-10)
        recall = tp / (total + 1e-10)
        F1 = 2 * precision * recall / (precision + recall + 1e-10)  
        return precision, recall, F1

    def test_one_epoch(self, device, output_path):
        # loss_meter, nll_loss_meter = AverageMeter(), AverageMeter()
        self.model.eval()
        tp, total, total_pred = 0, 0, 0
        with open(output_path, 'a') as f:
            for i, inputs in tqdm(enumerate(self.test_loader)):
                inputs = {k: inputs[k].to(device) for k in inputs}
                logits, hidden, _, _ = self.model(inputs['input_ids'].transpose(0, 1), inputs['real_length'].cpu())
                preds = logits.argmax(-1)
                batch_size = inputs['input_ids'].shape[0]
                for batch_idx in range(batch_size):
                    pred_triples = []
                    gold_triples = []
                    entity_ids = inputs['entity_masks'][batch_idx].nonzero().squeeze(-1).tolist()
                    for entity_id1 in entity_ids:
                        for entity_id2 in entity_ids:
                            pred_label = preds[batch_idx][entity_id1][entity_id2].item()
                            if pred_label > 0:
                                pred_triples.append((entity_id1, pred_label, entity_id2))
                    gold_triples = [(h, inputs['rel_labels'][batch_idx][h, t].item(), t) 
                        for h, t in (inputs['rel_labels'][batch_idx] > 0).nonzero().tolist()]
                    # pred_triples = [(s, e) for s, e in pred_triples if s < e]
                    # gold_triples = [(s, e) for s, e in gold_triples if s < e]
                    for item in gold_triples:
                        if item in pred_triples:
                            tp += 1
                    total_pred += len(pred_triples)
                    total += len(gold_triples)
        print(tp, total_pred, total)
        print(self.get_f1(tp, total_pred, total))


class DependencyConstrainedLSTMTester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.dataset = test_set

        assert self.args.parser_type == 'rel_checker'
        self.parser = RelChecker()

        with open('rels.txt', 'r') as f:
            self.rel_dict = json.loads(f.read().strip())
        print(self.rel_dict)

    def test_one_epoch(self, device, output_path):
        print('parser to device ...')
        self.parser._to_device(device)
        # get constraints 
        all_lexical_constraints = []
        all_dep_constraints = []
        all_refs = []
        for obj in self.dataset.metadata:
            constraints = []
            dep_constraints = []
            for h, r, t in obj['triple']:
                if '@@ROOT@@' in r:
                    continue 
                if h not in constraints:
                    constraints.append(h)
                if t not in constraints:
                    constraints.append(t)
                # if len(r) >= 3:
                #     print(r)
                #     constraints.append(r)
                rel = ' '.join(r)
                if rel in self.rel_dict:
                    dep_constraints.append((h, self.rel_dict[rel], t))
                # dep_constraints.append((h, 1, t))
            
            # cons_cnt = Counter(constraints)
            # constraints = []
            # for k in cons_cnt:
            #     if cons_cnt[k] == 1:
            #         constraints.append(k)
            #     else:
            #         constraints += [k] * int(cons_cnt[k] / 2)

            constraints = [self.tokenizer([c]) for c in constraints]
            all_lexical_constraints.append(constraints)
            all_dep_constraints.append(dep_constraints)
            all_refs.append(obj['revised'])

        self.model.eval()
        with open(output_path, 'w') as f:
            cur_pos = 0
            for i, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                # if i < 3:
                #     continue
                inputs = {k: inputs[k].to(device) for k in inputs}
                batch_size = inputs['input_ids'].shape[0]

                model_kwargs = {
                    'prune_size': self.args.prune_size,
                    'parser': self.parser,
                    'lamb': self.args.lamb,
                    'scorer': self.args.scorer,
                    'rho': self.args.rho,
                    'alpha_func': self.args.alpha_func,
                    'normalize': self.args.normalize,
                    'bank_count': self.args.bank_count,
                    'force_complete': self.args.force_complete,
                    'reset_avail_states': self.args.reset_avail_states,
                    'denoise': self.args.denoise,
                    'completion': self.args.completion,
                    'lower': False
                }
            
                output_ids = my_beam_search(self.model, device,
                    inputs=inputs,
                    num_beams=self.args.beam_size, 
                    max_length=self.args.max_tgt_length, 
                    early_stopping=self.args.early_stopping, 
                    length_penalty=self.args.length_penalty,
                    repetition_penalty=self.args.repetition_penalty,
                    bos_token_id=self.tokenizer(['<bos>'])[0], 
                    pad_token_id=self.tokenizer(['<pad>'])[0], 
                    eos_token_id=self.tokenizer(['<eos>'])[0],
                    lexical_constraints=all_lexical_constraints[cur_pos: cur_pos + batch_size],
                    dep_constraints=all_dep_constraints[cur_pos: cur_pos + batch_size],
                    **model_kwargs
                )

                output = [self.tokenizer.decode(g) for g in output_ids]

                # all_deps = self.parser.get_all_deps(output)
                
                for j in range(cur_pos, cur_pos + batch_size):
                    sent, cons = output[j - cur_pos], all_lexical_constraints[j]
                    f.write(sent + '\n')
                # exit()
                if i % self.args.logging_steps == 0:
                    print(f'epoch: [1] | batch: [{i}]', file=sys.stderr)
                if self.args.debug:
                    break

                cur_pos += batch_size


class DependencyConstrainedWebnlgTester(object):
    def __init__(self, args, model, test_loader, test_set, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.dataset = test_set

        assert self.args.parser_type == 'rel_checker'
        self.parser = RelCheckerForWebnlg()

    def test_one_epoch(self, device, output_path):
        print('parser to device ...')
        self.parser._to_device(device)

        h_token_id = self.tokenizer.encode('<H>')[0]
        r_token_id = self.tokenizer.encode('<R>')[0]
        t_token_id = self.tokenizer.encode('<T>')[0]

        self.model.eval()
        cons_path = output_path.replace('.out', '.cons')
        ff = open(cons_path, 'w')
        with open(output_path, 'w') as f:
            cur_pos = 0
            for i, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                # if i < 3:
                #     continue
                inputs = {k: inputs[k].to(device) for k in inputs}
                batch_size = inputs['input_ids'].shape[0]

                model_kwargs = {
                    'prune_size': self.args.prune_size,
                    'parser': self.parser,
                    'tokenizer': self.tokenizer,
                    'lamb': self.args.lamb,
                    'scorer': self.args.scorer,
                    'rho': self.args.rho,
                    'alpha_func': self.args.alpha_func,
                    'normalize': self.args.normalize,
                    'bank_count': self.args.bank_count,
                    'force_complete': self.args.force_complete,
                    'reset_avail_states': self.args.reset_avail_states,
                    'denoise': self.args.denoise,
                    'completion': self.args.completion,
                    'lower': True,
                    'partial': True,
                    'partial_top_k': 5,
                }

                # [bsz, seq_len]
                cons_prob = self.model.get_constraints_words(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=inputs['decoder_input_ids']
                )
                cons_label = (cons_prob > 0.9).long()

                # List[List[Constraint[]]]
                batch_raw_words, batch_lexical_constraints, batch_dep_constraints = \
                    get_constraints(inputs, cons_label, self.tokenizer, h_token_id, r_token_id, t_token_id)
            
                output_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    num_beams=self.args.beam_size, 
                    max_length=self.args.max_tgt_length, 
                    early_stopping=not self.args.no_early_stopping, 
                    length_penalty=self.args.length_penalty,
                    repetition_penalty=self.args.repetition_penalty,
                    bos_token_id=self.tokenizer.bos_token_id, 
                    pad_token_id=self.tokenizer.pad_token_id, 
                    eos_token_id=self.tokenizer.eos_token_id,
                    constraints=batch_lexical_constraints,
                    dep_constraints=batch_dep_constraints,
                    **model_kwargs
                )

                output = [self.tokenizer.decode(g, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False) for g in output_ids]

                for sent, cons, dep_cons in zip(output, batch_raw_words, batch_dep_constraints):
                    d = {'lexi_cons': cons, 'dep_cons': dep_cons}
                    print(d)
                    print(sent)
                    f.write(convert_text(sent) + '\n')
                    ff.write(json.dumps(d, ensure_ascii=False) + '\n')
                if i % self.args.logging_steps == 0:
                    print(f'epoch: [1] | batch: [{i}]', file=sys.stderr)
                if self.args.debug:
                    break

                cur_pos += batch_size
        ff.close()


tester_factory = {
    'base': BaseTester,
    'DBA': DBATester,
    'dep': DependencyConstrainedTester,
    'rerank': ReRankTester,
    'deppred': DepPredTester,
    'DDBA': DDBATester,
    'base-lstm': BaseLSTMTester,
    'dep-lstm': DependencyConstrainedLSTMTester,
    'relpred': RelPredTester,
    'dep-webnlg': DependencyConstrainedWebnlgTester
}

