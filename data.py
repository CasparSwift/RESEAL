import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import sys
import re
from utils import match_lst


def make_dataloader(args, tokenizer, mode='train'):
    if mode == 'train':
        train_set = dataset_factory[args.dataset](args, tokenizer, 'train') 
    test_set = dataset_factory[args.dataset](args, tokenizer, 'test')
    if mode == 'train':
        print(f'train: {len(train_set)} test: {len(test_set)}')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.n_workers, drop_last=False, collate_fn=lambda x: collate_fn_factory[args.dataset](x, tokenizer, args))
    else:
        print(f'test: {len(test_set)}')
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, 
        num_workers=args.n_workers, drop_last=False, collate_fn=lambda x: collate_fn_factory[args.dataset](x, tokenizer, args))
    if mode == 'train':
        return train_loader, test_loader, train_set, test_set
    else:
        return test_loader, test_set


def padding(tensor, max_length, padding_id=0):
    if tensor.shape[1] > max_length:
        return tensor[:, :max_length]
    else:
        values = tensor.tolist()
        pad_len = max_length - tensor.shape[1]
        values = [v + [padding_id] * pad_len for v in values]
        return torch.tensor(values)


def base_collate_fn(batch, tokenizer, args):
    batched_X = [item[0] for item in batch]
    batched_Y = [item[1] for item in batch]
    src_inputs = tokenizer(batched_X, return_tensors="pt", padding=True, truncation=True)
    tgt_inputs = tokenizer(batched_Y, return_tensors="pt", padding=True, truncation=True)
    constraint_label = []
    for batch_idx in range(src_inputs['input_ids'].shape[0]):
        labels = []
        tgt_tokens_set = set(tgt_inputs['input_ids'][batch_idx].tolist())
        for token in src_inputs['input_ids'][batch_idx].tolist():
            if token == tokenizer.pad_token_id:
                labels.append(-1)
            else:
                labels.append(int(token in tgt_tokens_set))
        constraint_label.append(labels)

    final_inputs = {
        'input_ids': padding(src_inputs['input_ids'], args.max_src_length, padding_id=tokenizer.pad_token_id),
        'attention_mask': padding(src_inputs['attention_mask'], args.max_src_length, padding_id=0),
        'decoder_input_ids': padding(tgt_inputs['input_ids'], args.max_tgt_length, padding_id=tokenizer.pad_token_id),
        'decoder_attention_mask': padding(tgt_inputs['attention_mask'], args.max_tgt_length, padding_id=0),
        'constraint_label': padding(torch.tensor(constraint_label), args.max_src_length, padding_id=-1)
    }
    return final_inputs


def base_collate_fn_with_constraint_label(batch, tokenizer, args):
    # batched_X = [item[0] for item in batch]
    batched_Y = [item[1] for item in batch]
    # all_labels = [item[2] for item in batch]
    src_inputs = []
    attention_masks = []
    tgt_inputs = tokenizer(batched_Y, return_tensors="pt", padding=True)

    constraint_label = []
    for item in batch:
        input_ids = []
        labels = []
        for word, label in zip(item[0].split(), item[2]):
            ids = tokenizer.encode(word, add_special_tokens=False)
            input_ids += ids
            labels += [label] * len(ids)
        input_ids.append(tokenizer.eos_token_id)
        labels.append(-1)
        pad_len = (args.max_src_length - len(input_ids))
        attention_masks.append([1] * len(input_ids) + [0] * pad_len)
        labels += [-1] * pad_len
        input_ids += [tokenizer.pad_token_id] * pad_len
        src_inputs.append(input_ids)
        constraint_label.append(labels)

    final_inputs = {
        'input_ids': torch.tensor(src_inputs),
        'attention_mask': torch.tensor(attention_masks),
        'decoder_input_ids': padding(tgt_inputs['input_ids'], args.max_tgt_length, padding_id=tokenizer.pad_token_id),
        'decoder_attention_mask': padding(tgt_inputs['attention_mask'], args.max_tgt_length, padding_id=0),
        'constraint_label': torch.tensor(constraint_label)
    }
    return final_inputs


def lstm_padding(values, max_length, padding_id):
    tensor = []
    for v in values:
        v = v[:max_length]
        tensor.append(v + [padding_id] * (max_length - len(v)))
    return torch.tensor(tensor)


def lstm_collate_fn(batch, tokenizer, args):
    batched_X = [tokenizer(item[0]) for item in batch]
    batched_Y = [tokenizer(item[1]) for item in batch]
    real_length = [len(x) for x in batched_X]
    tgt_real_length = [len(y) for y in batched_Y]

    batched_draft = [tokenizer(item[2]) for item in batch]
    batched_triple = [tokenizer(item[3]) for item in batch]
    draft_real_length = [len(d) for d in batched_draft]
    triple_real_length = [len(t) for t in batched_triple]

    triple_mask = [item[4] for item in batch]

    final_inputs = {
        'input_ids': lstm_padding(batched_X, max(real_length), padding_id=0),
        'decoder_input_ids': lstm_padding(batched_Y, max(tgt_real_length), padding_id=0),
        'real_length': torch.tensor(real_length),
        'draft_input_ids': lstm_padding(batched_draft, max(draft_real_length), padding_id=0),
        'triple_input_ids': lstm_padding(batched_triple, max(triple_real_length), padding_id=0),
        'draft_real_length': torch.tensor(draft_real_length),
        'triple_real_length': torch.tensor(triple_real_length),
        'triple_mask': lstm_padding(triple_mask, max(triple_real_length), padding_id=0)
    }
    return final_inputs


def list_padding(values, max_length, padding_id=0):
    pad_values = []
    for v in values:
        l = len(v)
        if l >= max_length:
            pad_values.append(v[:max_length])
        else:
            pad_values.append(v + [padding_id] * (max_length - l))
    return torch.tensor(pad_values)


def depparse_collate_fn(batch, tokenizer, args):
    batched_words = [item[0] for item in batch]
    batched_pos = [item[1] for item in batch]
    batched_head = [item[2] for item in batch]
    batched_dep = [item[3] for item in batch]
    input_mask = [[1]*len(w) for w in batched_words]
    final_inputs = {
        'input_ids': list_padding(batched_words, args.max_src_length, padding_id=0),
        'input_mask': list_padding(input_mask, args.max_src_length, padding_id=0),
        'pos': list_padding(batched_pos, args.max_src_length, padding_id=-1),
        'head': list_padding(batched_head, args.max_src_length, padding_id=-1),
        'dep': list_padding(batched_dep, args.max_src_length, padding_id=-1),
    }
    return final_inputs


def deppred_collate_fn(batch, tokenizer, args):
    batched_input_ids = [item[0] for item in batch]
    batched_word_id_to_token_span = [item[1] for item in batch]
    batched_heads = [item[2] for item in batch]
    batched_dep_rels = [item[3] for item in batch]
    batched_labels = [item[4] for item in batch]
    batched_keyword_labels = [item[5] for item in batch]
    input_mask = [[1]*len(w) for w in batched_input_ids]
    word_cnt_mask = [[1]*len(heads) for heads in batched_heads]

    final_inputs = {
        'input_ids': list_padding(batched_input_ids, args.max_src_length, padding_id=0),
        'attention_mask': list_padding(input_mask, args.max_src_length, padding_id=0),
        'word_cnt_mask': list_padding(word_cnt_mask, args.max_word_length, padding_id=0),
        'word_id_to_token_span': list_padding(batched_word_id_to_token_span, args.max_word_length, padding_id=[0, 0]),
        'heads': list_padding(batched_heads, args.max_word_length, padding_id=0),
        'dep_rels': list_padding(batched_dep_rels, args.max_word_length, padding_id=0),
        'labels': list_padding(batched_labels, args.max_word_length, padding_id=0),
        'keyword_labels': list_padding(batched_keyword_labels, args.max_word_length, padding_id=0),
    }
    return final_inputs


def rel_collate_fn(batch, tokenizer, args):
    batched_X = [tokenizer(item[0]) for item in batch]
    entity_masks = [item[1] for item in batch]
    rel_label_dicts = [item[2] for item in batch]

    real_length = [len(x) for x in batched_X]
    max_length = max(real_length)
    rel_labels_tensor = torch.zeros(len(batch), max_length, max_length)
    for i, rel_label_dict in enumerate(rel_label_dicts):
        for h_idx, t_idx in rel_label_dict:
            rel_labels_tensor[i, h_idx, t_idx] = rel_label_dict[(h_idx, t_idx)]
    final_inputs = {
        'input_ids': lstm_padding(batched_X, max_length, padding_id=0),
        'entity_masks': lstm_padding(entity_masks, max_length, padding_id=0),
        'rel_labels': rel_labels_tensor.long(),
        'real_length': torch.tensor(real_length)
    }
    return final_inputs


def webnlg_rel_collate_fn(batch, tokenizer, args):
    batched_X = [item[0] for item in batch]
    entity_masks =  [item[1] for item in batch]
    rel_labels = [item[2] for item in batch]

    real_length = [len(x) for x in batched_X]

    max_length = max(real_length)

    rel_labels_tensor = []
    for rel_label in rel_labels:
        rel_label = [r + [-1] * (max_length - len(r)) for r in rel_label.tolist()]
        rel_label += [[-1] * max_length] * (max_length - len(rel_label))
        rel_labels_tensor.append(rel_label)

    final_inputs = {
        'input_ids': lstm_padding(batched_X, max_length, padding_id=tokenizer.pad_token_id),
        'rel_labels': torch.tensor(rel_labels_tensor),
        'real_length': torch.tensor(real_length),
        'entity_masks': lstm_padding(entity_masks, max_length, padding_id=0)
    }
    return final_inputs


class GigaWordDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split_map = {
            'train': 'train',
            'dev': 'valid',
            'test': 'test'
        }
        self.split = self.split_map[split]
        self.data = self.read_data()

    def read_data(self):
        if self.args.giga_test_set == 'origin':
            postfix = 'txt.str'
            src_path = f'Dataset/GigaWord/{self.split}.article.{postfix}'
            tgt_path = f'Dataset/GigaWord/{self.split}.title.{postfix}'
        elif self.args.giga_test_set == 'internal':
            postfix = 'filter.txt.2k'
            src_path = f'Dataset/GigaWord/{self.split}.article.{postfix}'
            tgt_path = f'Dataset/GigaWord/{self.split}.title.{postfix}'
        elif self.args.giga_test_set == 'duc':
            src_path = 'Dataset/duc04/duc04.src'
            tgt_path = 'Dataset/duc04/task1_ref0.txt.tok'
        elif self.args.giga_test_set == 'MSR':
            src_path = 'Dataset/MSR/msr.src'
            tgt_path = 'Dataset/MSR/ref1.txt'
        else:
            exit('giga test set not exists!')

        src_data = []
        with open(src_path, 'r') as f:
            for line in f:
                line = re.sub(r'\d', '#', line)
                src_data.append(line.strip())
        tgt_data = []
        with open(tgt_path, 'r') as f:
            for line in f:
                tgt_data.append(line.strip())
        assert len(src_data) == len(tgt_data)
        return list(zip(src_data, tgt_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EnglishEWTDependencyDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split_map = {
            'train': 'train',
            'dev': 'dev',
            'test': 'test'
        }
        self.split = self.split_map[split]
        self.data, self.metadata = self.read_data()

    def read_data(self):
        data, metadata = [], []
        with open(os.path.join('Dataset', 'UD_English-EWT-Dep', f'{self.split}.json'), 'r') as f:
            for line in f:
                obj = json.loads(line.strip())
                X = ", ".join(" # ".join(triplet) for triplet in obj["dependencies"])
                Y = obj["text"]
                data.append((X, Y))
                metadata.append(obj)
        return data, metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GigaDependencyDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split_map = {
            'train': 'test',
            'dev': 'test',
            'test': 'test'
        }
        self.split = self.split_map[split]
        self.data, self.metadata = self.read_data()

    def read_data(self):
        data, metadata = [], []
        if self.args.dep_giga_path == 'stdin':
            for line in sys.stdin:
                obj = json.loads(line.strip())
                X = obj["text"]
                Y = obj["ref"]
                data.append((X, Y))
                keywords = set(obj['keywords']) if 'keywords' in obj else set([dep[-1] for dep in obj['src_deps']]) & set([dep[-1] for dep in obj['ref_deps']])
            
                obj['keywords'] = keywords

                # metadata.append(obj)
                metadata.append(obj)
            return data, metadata

        if self.args.dep_giga_path is not None:
            path = self.args.dep_giga_path
        else:
            path = os.path.join('Dataset', 'Giga-Dep', f'{self.split}.json') 
        with open(path, 'r') as f:
            for line in f:
                obj = json.loads(line.strip())
                X = obj["text"]
                Y = obj["ref"]
                keywords = set(obj['keywords']) if 'keywords' in obj else set([dep[-1] for dep in obj['src_deps']]) & set([dep[-1] for dep in obj['ref_deps']])
            
                obj['keywords'] = keywords

                metadata.append(obj)
                data.append((X, Y))
                # metadata.append(obj)
        return data, metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GigaDependencyPredDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split_map = {
            'train': 'train',
            'dev': 'train',
            'test': 'test'
        }
        self.split = self.split_map[split]
        self.DEP_DIC = {'acl': 0, 'acl:relcl': 1, 'advcl': 2, 'advmod': 3, 
            'amod': 4, 'appos': 5, 'aux': 6, 'aux:pass': 7, 'case': 8, 
            'cc': 9, 'cc:preconj': 10, 'ccomp': 11, 'compound': 12, 
            'compound:prt': 13, 'conj': 14, 'cop': 15, 'csubj': 16, 
            'csubj:pass': 17, 'dep': 18, 'det': 19, 'det:predet': 20, 
            'discourse': 21, 'dislocated': 22, 'expl': 23, 'fixed': 24, 
            'flat': 25, 'flat:foreign': 26, 'goeswith': 27, 'iobj': 28, 
            'list': 29, 'mark': 30, 'nmod': 31, 'nmod:npmod': 32, 'nmod:poss': 33, 
            'nmod:tmod': 34, 'nsubj': 35, 'nsubj:pass': 36, 'nummod': 37, 
            'obj': 38, 'obl': 39, 'obl:npmod': 40, 'obl:tmod': 41, 
            'orphan': 42, 'parataxis': 43, 'punct': 44, 'reparandum': 45, 
            'root': 46, 'vocative': 47, 'xcomp': 48}
        import nltk
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
        self.data, self.metadata = self.read_data()

    def read_data(self):
        data, metadata = [], []
        if self.args.giga_test_set == 'origin':
            if self.split == 'train':
                file = f'/dataA/giga/{self.split}_l2r.json'
            else:
                file = f'/dataA/giga/{self.split}.json'
        elif self.args.giga_test_set == 'internal':
            if self.split == 'train':
                # file = f'/dataA/giga/{self.split}_l2r.json'
                file = f'/dataA/giga/{self.split}.json'
            else:
                file = f'/dataA/giga/{self.split}_internal.json'
        elif self.args.giga_test_set == 'duc':
            if self.split == 'train':
                # file = f'/dataA/giga/{self.split}_l2r.json'
                file = f'/dataA/giga/{self.split}.json'
            else:
                file = f'/dataA/giga/{self.split}_duc.json' 
        elif self.args.giga_test_set == 'MSR':
            if self.split == 'train':
                # file = f'/dataA/giga/{self.split}_l2r.json'
                file = f'/dataA/giga/{self.split}.json'
            else:
                file = f'/dataA/giga/{self.split}_MSR.json'
        with open(file, 'r') as f:
            # max_length = 0
            # max_sent_length = 0
            # max_sent = None
            for idx, line in tqdm(enumerate(f)):
                try:
                    obj = json.loads(line.strip())
                    # print(obj)
                except:
                    continue
                # if obj['intersection'] == []:
                #     continue
                new_obj = {
                    'src': obj['src'] if 'src' in obj else obj['text'],
                    'ref': obj['ref'],
                    'src_deps': obj['src_deps'],
                    'gold': obj['intersection'] if 'intersection' in obj else obj['dependencies'],
                    'keywords': set(obj['keywords']) if 'keywords' in obj else set([dep[-1] for dep in obj['src_deps']]) & set([dep[-1] for dep in obj['ref_deps']])
                }
                gold = []
                for dep in new_obj['gold']:
                    # if dep[0] in self.stop_words or dep[-1] in self.stop_words:
                    if '_ROOT' in dep:
                        continue
                    gold.append(dep)
                new_obj['gold'] = gold

                gold_keywords = []
                for w in new_obj['keywords']:
                    if w not in self.stop_words and '#' not in w:
                        gold_keywords.append(w)
                new_obj['keywords'] = set(gold_keywords)

                metadata.append(new_obj)
                # input_ids = [101]
                # for word_id, w in enumerate(obj['src_deps']):
                    # sub_word_ids = self.tokenizer.encode(w[-1], add_special_tokens=False)
                    # input_ids += sub_word_ids
                # max_length = max(max_length, len(input_ids))
                # max_sent_length = max(max_sent_length, len(obj['src_deps']))
                # l = len(obj['src_deps'])
                # if l > max_sent_length:
                #     max_sent_length = l
                #     max_sent = [w[-1] for w in obj['src_deps']] 
                # if idx % 100000 == 0:
                #     print(max_sent_length)
                # if idx == 100000:
                    # break
            # print(max_sent_length)
            # print(max_sent)

        return data, metadata

    def get_data_from_metadata(self, obj):
        token_id_num = 1
        word_id_num = 0
        input_ids = [101]
        word_id_to_token_span = []
        token_id_to_word_id = [-1]
        word_to_word_id = {'_ROOT': [0]}
        labels = []
        keyword_labels = []
        heads = []
        dep_rels = []
        for word_id, w in enumerate(obj['src_deps']):
            sub_word_ids = self.tokenizer.encode(w[-1], add_special_tokens=False)
            sub_word_len = len(sub_word_ids)
            if token_id_num + sub_word_len > self.args.max_src_length - 1:
                break
            input_ids += sub_word_ids       
            word_id_to_token_span.append([token_id_num, token_id_num + sub_word_len - 1])
            token_id_to_word_id += [word_id] * sub_word_len
            token_id_num += sub_word_len
            word_to_word_id[w[-1]] = word_to_word_id.get(w[-1], []) + [word_id]
            labels.append(int(w in obj['gold']))
            keyword_labels.append(int(w[-1] in obj['keywords']))
            word_id_num += 1
        for word_id, w in enumerate(obj['src_deps'][:word_id_num]):
            head_word_idx = sorted(word_to_word_id[w[0]], 
                key=lambda x: abs(x - word_id))[0]
            heads.append(head_word_idx)
            # one_hot = [0] * 50
            # one_hot[self.DEP_DIC[w[1]] + 1] = 1
            # dep_rels.append(one_hot)
            dep_rels.append(self.DEP_DIC[w[1]] + 1)
        input_ids.append(102)
        heads_span = [word_id_to_token_span[head] if head != -1 else [0, 0] for head in heads]

        # print(obj["src"])
        # print(input_ids)
        # print(word_to_word_id)
        # print(word_id_to_token_span)
        # print(token_id_to_word_id)

        # obj['word_id_to_token_span'] = word_id_to_token_span
        # obj['token_id_to_word_id'] = token_id_to_word_id
        # obj['word_to_word_id'] = word_to_word_id
        # obj['heads'] = heads
        # obj['heads_span'] = heads_span
        # obj['dep_rels'] = dep_rels
        # metadata.append(obj)
        # pdb.set_trace()
        # print(keyword_labels)
        assert len(word_id_to_token_span) == len(heads_span) == len(dep_rels) == len(labels) == len(keyword_labels)
        return (input_ids, word_id_to_token_span, heads, dep_rels, labels, keyword_labels)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return self.get_data_from_metadata(self.metadata[idx])


class WebNLGDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split_map = {
            'train': 'train',
            'dev': 'val',
            'test': 'test_both'
        }
        self.split = self.split_map[split]
        self.data, self.metadata = self.read_data()

    def read_data(self):
        data, metadata = [], []
        source, target, labels = [], [], []
        with open(os.path.join('Dataset/webnlg/', f'{self.split}.target'), 'r') as f:
            for line in f:
                # line = line.lower()
                target.append(line.strip('\n'))
        with open(os.path.join('Dataset/webnlg', f'{self.split}.source'), 'r') as f:
            for idx, line in enumerate(f):
                # line = line.lower()
                text = 'translate Graph to English: ' + line.strip('\n')
                source.append(text)
                items = re.split('<[HRT]>', line.strip('\n'))
                assert items[0] == ''
                items = [item.strip() for item in items[1:]]
                triples = []
                for i in range(0, len(items), 3):
                    triples.append((items[i], items[i+1], items[i+2]))
                metadata.append({
                    'dependencies': triples,
                    'raw_text': text
                })

                triple_cnt = 0
                entity_type = 0
                constraints_label = []
                for word in text.split():
                    if word == '<H>':
                        triple_cnt += 1
                        entity_type = 1
                        constraints_label.append(int(triples[triple_cnt-1][0].lower() in target[idx].lower()))
                    elif word == '<R>':
                        entity_type = 2
                        constraints_label.append(-1)
                    elif word == '<T>':
                        entity_type = 3
                        constraints_label.append(int(triples[triple_cnt-1][2].lower() in target[idx].lower()))
                    else:
                        if entity_type == 2:
                            constraints_label.append(int(word in target[idx]))
                        else:
                            constraints_label.append(-1)
                assert len(constraints_label) == len(text.split())

                labels.append(constraints_label)

                # print(text)
                # print(target[idx])
        
        data = list(zip(source, target, labels))
        return data, metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WebNLGRelationDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split_map = {
            'train': 'train',
            'dev': 'val',
            'test': 'test_both'
        }
        self.split = self.split_map[split]
        self.data, self.metadata = self.read_data()

    def read_data(self):
        data, metadata = [], []
        all_triples, target = [], []
        with open(os.path.join('Dataset/webnlg', f'{self.split}.source'), 'r') as f:
            for line in f:
                items = re.split('<[HRT]>', line.strip('\n'))
                assert items[0] == ''
                items = [item.strip() for item in items[1:]]
                triples = []
                for i in range(0, len(items), 3):
                    triples.append((items[i], items[i+1], items[i+2]))
                all_triples.append(triples)
        with open(os.path.join('Dataset/webnlg/', f'{self.split}.target_eval'), 'r') as f:
            for line in f:
                target.append(line.strip('\n'))
        for triples, text in zip(all_triples, target):
            text_ids = self.tokenizer.encode(text.lower(), add_special_tokens=False)
            relation_labels = np.array([[0 for y in range(len(text_ids))] for x in range(len(text_ids))])
            entity_mask = [0 for y in range(len(text_ids))]
            for h, r, t in triples:
                head_ids = self.tokenizer.encode(h.lower(), add_special_tokens=False)
                # rel_ids = self.tokenizer.encode(r.lower(), add_special_tokens=False)
                tail_ids = self.tokenizer.encode(t.lower(), add_special_tokens=False)
                head_indice = match_lst(text_ids, head_ids)
                tail_indice = match_lst(text_ids, tail_ids)
                if len(head_indice) == 1 and len(tail_indice) == 1:
                    h_idx, t_idx = head_indice[0], tail_indice[0]
                    h_end, t_end = h_idx + len(head_ids), t_idx + len(tail_ids)
                    relation_labels[h_idx: h_end, h_idx: h_end] = 1
                    relation_labels[t_idx: t_end, t_idx: t_end] = 1
                    relation_labels[h_idx: h_end, t_idx: t_end] = 2
                    for idx in range(h_idx, h_end):
                        entity_mask[idx] = 1
                    for idx in range(t_idx, t_end):
                        entity_mask[idx] = 1
            if sum(entity_mask) > 0:
                data.append((text_ids, entity_mask, relation_labels))
        return data, metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EditingDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split_map = {
            'train': 'train',
            'dev': 'dev',
            'test': 'test'
        }
        # self.vocab = {'<H>', '<R>', '<T>'}
        self.split = self.split_map[split]
        self.data, self.metadata = self.read_data()
        

    def read_data(self):
        data, metadata = [], []
        max_length = 0
        with open(os.path.join('Dataset/webedit', f'{self.split}.jsonl'), 'r') as f:
            for line in f:
                obj = json.loads(line.strip('\n'))
                draft = ' '.join(' '.join(words) for words in obj['draft'])
                revise_text = ' '.join(' '.join(words) for words in obj['revised'])
                triples = [f'<H> {h} <R> {" ".join(r)} <T> {t}' for h, r, t in obj['triple'] if '@@ROOT@@' not in r]
                triple_text = ' '.join(triples)
                if 't5' in self.args.pretrain_model.lower():
                    text = 'translate Graph to English: ' + draft + triple_text
                else:
                    text = draft + triple_text
                data.append((text, revise_text))
                metadata.append(obj)
                # for sent in obj['draft']:
                #     for word in sent:
                #         self.vocab.add(word)
                # for sent in obj['revised']:
                #     for word in sent:
                #         self.vocab.add(word)
                # for h, r, t in obj['triple']:
                #     self.vocab.add(h)
                #     self.vocab.add(t)
                #     for relation in r:
                #         self.vocab.add(relation)     
                # ls.append(len(self.tokenizer(revise_text)['input_ids']))
        # length = len(data)
        # data = [data[i] for i in range(length) if i % 100 == 0]
        # metadata = [metadata[i] for i in range(length) if i % 100 == 0]
        # print(f'vocab size: {len(self.vocab)}')
        return data, metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WebEditingLSTMDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split_map = {
            'train': 'train',
            'dev': 'dev',
            'test': 'test'
        }
        self.vocab = []
        self.entitys = []
        self.t5_vocab = []
        self.relations = []
        self.split = self.split_map[split]
        self.data, self.metadata = self.read_data()   

    def read_data(self):
        data, metadata = [], []
        # max_length = 0
        max_oovs = 0
        # from transformers import AutoTokenizer
        # tok = AutoTokenizer('/dataB/pretrain/T5-base/')
        # with open('webedit_vocab2.txt', 'r') as f:
        #     all_vocabs = set(f.read().split('\n'))
        with open(os.path.join('Dataset/webedit', f'{self.split}.jsonl'), 'r') as f:
            for line in f:
                obj = json.loads(line.strip('\n'))
                draft = [w for words in obj['draft'] for w in words]
                revise_text = [w for words in obj['revised'] for w in words]
                triple_word_list, triple_mask = [], []
                for h, r, t in obj['triple']:
                    triple = ['<H>', h, '<R>'] + r + ['<T>', t]
                    triple_word_list += triple
                    triple_mask += [0] * (len(triple) - 1) + [1]
                src_word_list = ['<bos>'] + draft + triple_word_list + ['<eos>']
                tgt_word_list = ['<bos>'] + revise_text + ['<eos>']
                draft_word_list = ['<bos>'] + draft + ['<eos>']
                # max_length = max(max_length, len(src_word_list), len(tgt_word_list))
                data.append((src_word_list, tgt_word_list, draft_word_list, triple_word_list, triple_mask))
                meta = {
                    'revised': revise_text,
                    'draft': draft,
                    'triple': obj['triple']
                }
                metadata.append(meta)
                for word in draft:
                    self.vocab.append(word)
                for word in revise_text:
                    self.vocab.append(word)
                for h, r, t in obj['triple']:
                    self.entitys.append(h)
                    self.entitys.append(t)
                    for relation in r:
                        self.vocab.append(relation)
                    self.relations.append(' '.join(r))
                
                # oovs = []
                # for word in src_word_list:
                #     if word not in all_vocabs:
                #         oovs.append(word)
                # max_oovs = max(max_oovs, len(oovs))

                # t5_ids = tok(' '.join(src_word_list))['input_ids'].tolist()

                
        print(f'vocab size: {len(set(self.vocab))}')
        print('max', max_oovs)
        
        # exit()
        return data, metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EditingRelationDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.split_map = {
            'train': 'train',
            'dev': 'dev',
            'test': 'test'
        }
        with open('rels.txt', 'r') as f:
            self.rel_dict = json.loads(f.read().strip())
        self.split = self.split_map[split]
        self.data, self.metadata = self.read_data()


    def read_data(self):
        data, metadata = [], []
        relation_type = set()
        with open(os.path.join('Dataset/webedit_rel', f'{self.split}_filt_aug'), 'r') as f:
            for i, line in tqdm(enumerate(f)):
                # if i > 1000:
                #     break
                obj = json.loads(line.strip('\n'))
                tgt_word_list = obj['sentence']
                # all_entity = obj['all_entity']
                entity_mask = obj['entity_mask']
                entity2idx = obj['entity2idx']
                # rel_label = [[0 for __ in range(self.args.max_src_length)] 
                #     for _ in range(self.args.max_src_length)]
                # entity_ids = [entity_idx for entity_idx, mask in enumerate(entity_mask) if mask]
                # for h_idx in entity_ids:
                #     for t_idx in entity_ids:
                #         rel_label[h_idx][t_idx] = 0
                rel_label_dict = {}
                for h, r, t in obj['relations']:
                    if h == '@@ROOT@@':
                        continue
                    if h not in entity2idx or t not in entity2idx:
                        continue
                    # index_pairs = []
                    # for h_idx in entity2idx[h]:
                    #     for t_idx in entity2idx[t]:
                    #         index_pairs.append((h_idx, t_idx))
                    # index_pairs.sort(key=lambda x: (abs(sentence_position_ids[x[0]] - sentence_position_ids[x[1]]), abs(x[0] - x[1])))
                    # if sentence_position_ids[h_idx] != sentence_position_ids[t_idx]:
                    #     continue
                    # if len(index_pairs) == 0:
                    #     print(index_pairs)
                    #     pdb.set_trace()
                    # h_idx, t_idx = index_pairs[0]
                    # rel_label[h_idx][t_idx] = 1
                    for h_idx in entity2idx[h]:
                        for t_idx in entity2idx[t]:
                            rel_label_dict[(h_idx, t_idx)] = self.rel_dict.get(' '.join(r), 1)
                    # gold_rels.append((h_idx, t_idx))
                    # if sentence_position_ids[h_idx] != sentence_position_ids[t_idx]:
                    #     print(h_idx, t_idx)
                    #     print(h, t)
                        # print(tgt_word_list)
                        # print('-' * 20)
                        # pdb.set_trace()
                # print(rel_label)
                # pdb.set_trace()
                data.append((tgt_word_list, entity_mask, rel_label_dict))
                metadata.append(obj)
        # print(len(relation_type))
        return data, metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



dataset_factory = {
    'gigaword': GigaWordDataset,
    'english-ewt': EnglishEWTDependencyDataset,
    'Giga_Dep': GigaDependencyDataset,
    'Giga_Dep_Pred': GigaDependencyPredDataset,
    'webnlg': WebNLGDataset,
    'webedit': EditingDataset,
    'webedit_lstm': WebEditingLSTMDataset,
    'webedit_rel': EditingRelationDataset,
    'webnlg_rel': WebNLGRelationDataset
}

collate_fn_factory = {
    'gigaword': base_collate_fn,
    'english-ewt': base_collate_fn,
    'Giga_Dep': base_collate_fn,
    'Giga_Dep_Pred': deppred_collate_fn,
    'webnlg': base_collate_fn_with_constraint_label,
    'webedit': base_collate_fn,
    'webedit_lstm': lstm_collate_fn,
    'webedit_rel': rel_collate_fn,
    'webnlg_rel': webnlg_rel_collate_fn
}

