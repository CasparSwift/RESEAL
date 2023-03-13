import json
import numpy as np
import math
import sys
import argparse
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

parser = argparse.ArgumentParser("Constrained Decoding")
parser.add_argument("-t1", type=float, default=0.6)
parser.add_argument("-t2", type=float, default=0.75)
parser.add_argument("-n1", type=int, default=4)
parser.add_argument("-n2", type=int, default=3)

args = parser.parse_args()

def read_data(file):
    datas = []
    with open(file, 'r') as f:
        for line in f:
            datas.append(json.loads(line.strip()))
    return datas


def get_data_pred1():
    pred_datas = []

    for e in range(6, 8):
  # datas = read_data(f'output/0104/bert_Giga_Dep_Pred_e{e}_infer_deppred_internal.out')
  # pred_datas.append(datas)
        for step in range(2000, 20000, 2000):
            datas = read_data(f'output/0104/bert_Giga_Dep_Pred_e{e}_infer_deppred_internal_s{step}.out')
            pred_datas.append(datas)

    pred_datas_for_kw = []
    for e in range(3, 4):
        datas = read_data(f'/data1/chenxiang/kw_0108/bert_Giga_Dep_Pred_e{e}_infer_deppred.out')
        pred_datas_for_kw.append(datas)
    return pred_datas, pred_datas_for_kw
  # for step in range(0, 18000, 2000):
  #     datas = read_data(f'/data1/chenxiang/kw_0108/bert_Giga_Dep_Pred_e{e}_infer_deppred_internal_s{step}.out')
  #     pred_datas_for_kw.append(datas)
# datas = read_data('/data1/chenxiang/kw_0108/bert_Giga_Dep_Pred_e4_infer_deppred.out')
# pred_datas_for_kw.append(datas)

# print(len(datas))


def get_data_pred2():
    pred_datas = []
    for e in range(6, 7):
  # datas = read_data(f'output/0104/bert_Giga_Dep_Pred_e{e}_infer_deppred_internal.out')
  # pred_datas.append(datas)
        for step in range(2000, 20000, 2000):
            datas = read_data(f'output/0104/bert_Giga_Dep_Pred_e{e}_infer_deppred_duc_s{step}.out')
            pred_datas.append(datas)

    pred_datas_for_kw = []
    for e in range(1, 2):
        for step in range(10000, 18000, 2000):
            datas = read_data(f'/data1/chenxiang/kw_0108/bert_Giga_Dep_Pred_e{e}_infer_deppred_duc_s{step}.out')
            pred_datas_for_kw.append(datas)
    return pred_datas, pred_datas_for_kw


def get_data_pred3():
    pred_datas = []
    for e in range(6, 7):
  # datas = read_data(f'output/0104/bert_Giga_Dep_Pred_e{e}_infer_deppred_internal.out')
  # pred_datas.append(datas)
        for step in range(2000, 20000, 2000):
            datas = read_data(f'output/0104/bert_Giga_Dep_Pred_e{e}_infer_deppred_MSR_s{step}.out')
            pred_datas.append(datas)


    pred_datas_for_kw = []
    for e in range(3, 4):
        for step in range(10000, 18000, 2000):
            datas = read_data(f'/data1/chenxiang/kw_0108/bert_Giga_Dep_Pred_e{e}_infer_deppred_MSR_s{step}.out')
            pred_datas_for_kw.append(datas)
    return pred_datas, pred_datas_for_kw


pred_datas, pred_datas_for_kw = get_data_pred2()


# hyp_path = 'output/gigaword_1019/bart_gigaword_e5_infer_base_internal.out'
# gold_path = '/data1/giga/test_internal.json'

hyp_path = 'output/bart_gigaword_e1_infer_base_duc_s20000.out'
gold_path = '/data1/chenxiang/giga/test_duc_full.json'

# hyp_path = '/data1/chenxiang/gigaword_0112/bart_gigaword_e1_infer_base_MSR_l2.0s20000.out'
# gold_path = '/data1/chenxiang/giga/test_MSR.json'

with open(hyp_path, 'r') as f:
    hyps = f.read().strip('\n').split('\n')

gold_datas = read_data(gold_path)

tp, total_pred, total = 0, 0, 0
tp1, total_pred1, total1 = 0, 0, 0
for i, gold_data in enumerate(gold_datas):
    words = hyps[i].split()
    # words = set()

    src_deps = pred_datas[0][i]['src_deps']

    probs_list = [pred_data[i]['probs'] for pred_data in pred_datas]
    probs = np.mean(np.array(probs_list), axis=0).tolist()

    pred_deps = []
    sum_probs = sum(probs)
    sum_exp_probs = sum(math.exp(p) for p in probs)
    items = sorted(zip(src_deps, probs), key=lambda x: x[-1], reverse=True)
    for dep, prob in items[:args.n1]:
        if '_ROOT' in dep:
            continue
        if '#' in dep or '$' in dep or '&amp' in dep:
            continue
        if '<' in dep or '>' in dep or '0.9' in dep:
            continue
        if prob > args.t1:# and prob / sum_probs > 0.05:
            if '_ROOT' not in dep and 'unk' not in dep: 
                if dep[0] not in words or dep[2] not in words:
                    if dep not in pred_deps:
                        pred_deps.append(dep)

    keyword_probs_list = [pred_data[i]['keyword_probs'] for pred_data in pred_datas_for_kw]
    keyword_probs = np.max(np.array(keyword_probs_list), axis=0).tolist()
    pred_keywords = []
    items = sorted(zip(src_deps, keyword_probs), key=lambda x: x[-1], reverse=True)
    for dep, prob in items[:args.n2]:
        if prob > args.t2:
            if dep[-1] not in words and dep[-1] not in pred_keywords:
                pred_keywords.append(dep[-1])
    
    # pred_deps = []
    # for dep in src_deps:
    #     if dep[0] in pred_keywords and dep[2] in pred_keywords:
    #         pred_deps.append(dep)
    # if pred_deps:
    #     print(pred_keywords)
    #     print(pred_deps)
    #     print(gold_data['text'])

    # get metric
    total_pred += len(pred_deps)
    total += len(gold_data['dependencies'])
    for dep in gold_data['dependencies']:
        if dep in pred_deps:
            tp += 1

    src_words = set([dep[-1] for dep in gold_data['src_deps']])
    ref_words = set(gold_data['keywords'])
    gold_keywords = []
    for w in (src_words & ref_words):
        if w not in stop_words and '#' not in w:
            gold_keywords.append(w)
    total_pred1 += len(pred_keywords)
    total1 += len(gold_keywords)
    for word in gold_keywords:
        if word in pred_keywords:
            tp1 += 1

    result_dict = {
        'text': gold_data['text'],
        'ref': gold_data['ref'],
        'src_deps': gold_data['src_deps'],
        'dependencies': pred_deps,
        'keywords': pred_keywords
    }
    # print(pred_deps, pred_keywords)
    print(json.dumps(result_dict, ensure_ascii=False))


def print_f1(tp, total_pred, total):
    precision = tp / (total_pred + 1e-10)
    recall = tp / (total + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    print(f'{precision*100:.2f}, {recall*100:.2f}, {F1*100:.2f}', file=sys.stderr)
    print(tp, total_pred, total, file=sys.stderr)


print_f1(tp, total_pred, total)
print_f1(tp1, total_pred1, total1)



