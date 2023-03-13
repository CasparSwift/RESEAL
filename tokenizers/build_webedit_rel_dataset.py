from tqdm import tqdm
import json
import os
import pdb
import copy
import random

random.seed(42)


def gen_data(split):
    fw = open(f'Dataset/webedit_rel/{split}_filt_aug', 'w')
    with open(os.path.join('Dataset/webedit', f'{split}.jsonl'), 'r') as f:
        for line in tqdm(f):
            obj = json.loads(line.strip('\n'))
            all_entity = set()
            for h, r, t in obj['triple']:
                all_entity.add(h)
                all_entity.add(t)
            all_entity -= {'@@ROOT@@'}
            for sent in obj['revised']:
                tgt_word_list = sent
                all_entity_this_sent = set(word for word in tgt_word_list if word in all_entity)

                entity_mask = []
                entity_ids = []
                entity2idx = {}
                for idx, word in enumerate(tgt_word_list):
                    if word in all_entity:
                        entity_mask.append(1)
                        entity2idx[word] = entity2idx.get(word, []) + [idx]
                        entity_ids.append(idx)
                    else:
                        entity_mask.append(0)
            
                valid_sample = True
                for k, v in entity2idx.items():
                    if len(v) >= 2:
                        valid_sample = False
                if not valid_sample:
                    continue
                
                relations = []
                for h, r, t in obj['triple']:
                    if h in all_entity_this_sent and t in all_entity_this_sent:
                        relations.append([h, r, t])

                data_entry = {
                    'sentence': tgt_word_list,
                    'relations': relations,
                    'entity2idx': entity2idx,
                    'entity_mask': entity_mask
                }

                fw.write(json.dumps(data_entry, ensure_ascii=False) + '\n')

                # negative samples
                for i in range(len(relations)):
                    h, r, t = relations[i]
                    # [start, end]
                    start_idx = entity2idx[h][0]
                    end_idx = entity2idx[t][0]
                    start_idx, end_idx = min(start_idx, end_idx), max(start_idx, end_idx)
                    if start_idx + 3 <= end_idx:
                        for drop_idx in range(start_idx + 2, end_idx):
                            if tgt_word_list[drop_idx] in all_entity:
                                continue
                            copy_data = copy.deepcopy(data_entry)
                            neg_tgt_word_list = tgt_word_list[:drop_idx] + tgt_word_list[end_idx:]
                            # print(tgt_word_list[drop_idx])
                            # print(tgt_word_list)    
                            # print(neg_tgt_word_list)
                            # pdb.set_trace()
                            copy_data['sentence'] = neg_tgt_word_list
                            copy_data['relations'][i][1] = 'other rel'
                            neg_entity_mask = []
                            neg_entity2idx = {}
                            for idx, word in enumerate(neg_tgt_word_list):
                                if word in all_entity:
                                    neg_entity_mask.append(1)
                                    neg_entity2idx[word] = neg_entity2idx.get(word, []) + [idx]
                                else:
                                    neg_entity_mask.append(0)
                            copy_data['entity_mask'] = neg_entity_mask
                            copy_data['entity2idx'] = neg_entity2idx
                            fw.write(json.dumps(copy_data, ensure_ascii=False) + '\n')
                        # print(copy_data)
                        # pdb.set_trace()                

    fw.close()

gen_data('train')
gen_data('test')
                