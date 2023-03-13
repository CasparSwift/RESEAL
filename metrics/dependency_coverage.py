from tqdm import tqdm
import sys
import json
from .metric_utils import batched_list, lower_dep, get_word_pairs
from sacremoses import MosesDetokenizer, MosesTokenizer

sys.path.append('../')
from parser_api import StanzaParser, SpaCyParser

class Dependency_Coverage_Evaluator(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        if kwargs['parser'] == 'stanza':
            self.parser = StanzaParser()
        else:
            self.parser = SpaCyParser()

        self.tokenizer = MosesTokenizer(lang='en')
        self.detokenizer = MosesDetokenizer(lang='en')


    def get_score(self, hyps, refs, all_gold_deps, return_list=False):
        tp_list, e3_cnt_list, total_list, word_occur_list, word_total_list = [], [], [], [], []
        for batched_hyp, batched_ref, batched_gold_deps in tqdm(batched_list(
            hyps, refs=refs, deps=all_gold_deps, batch_size=self.kwargs['batch_size'])
        ):
            batched_gold_deps = list(map(lower_dep, batched_gold_deps))
            batched_word_pairs = list(map(get_word_pairs, batched_gold_deps))

            batched_all_deps = self.parser.get_all_deps(batched_hyp)
            # print("len of batched_all_deps is : ", len(batched_all_deps))
            for idx, all_deps in enumerate(batched_all_deps):
                tp, total = 0, 0
                tp1, total1 = 0, 0
                e1_cnt, e2_cnt, e3_cnt = 0, 0, 0
                word_occur, word_total = 0, 0

                hyp, ref = batched_hyp[idx], batched_ref[idx]
                gold_deps, word_pairs = batched_gold_deps[idx], batched_word_pairs[idx]

                predict_deps = []
                for w1, r, w2 in all_deps:
                    if (w1, w2) in word_pairs:
                        predict_deps.append([w1, r, w2])
           
                total += len(gold_deps)
                if len(gold_deps) == 1:
                    total1 += 1
                for gold_dep in gold_deps:
                    if gold_dep in predict_deps:
                        tp += 1
                        if len(gold_deps) == 1:
                            tp1 += 1
                    else:
                        # 三种情况，某个词没在句子中；两个词都在但是没有依存关系；两个词的依存关系不对
                        w1, r, w2 = gold_dep
                        if w1 not in hyp.lower() or w2 not in hyp.lower():
                            e1_cnt += 1
                        elif (w1, w2) not in [(t[0], t[2]) for t in all_deps]:
                            e2_cnt += 1
                        else:
                            e3_cnt += 1

                        if self.kwargs['verbose']:
                            print('gold_dep:', gold_dep, file=sys.stderr)
                            print('predict_dep:', predict_deps, file=sys.stderr)
                            print('all_dep:', all_deps, file=sys.stderr)
                            print('hyp: ', hyp, file=sys.stderr)
                            print('ref:', ref, file=sys.stderr)
                            print(e1_cnt, e2_cnt, e3_cnt, file=sys.stderr)
                            print('-'*50, file=sys.stderr)
                
                gold_words_occur_cnts = {}
                # word coverage
                for gold_dep in gold_deps:
                    w1, r, w2 = gold_dep
                    if w1 == w2:
                        gold_words_occur_cnts[w1] = 2
                    else:
                        for w in [w1, w2]:
                            if w not in gold_words_occur_cnts:
                                gold_words_occur_cnts[w] = 1
                
                all_hyp_words = hyp.lower().split()

                for w, cnt in gold_words_occur_cnts.items():
                    pred_word_occur_cnt = all_hyp_words.count(w)
                    if pred_word_occur_cnt >= cnt:
                        word_occur += cnt
                    else:
                        word_occur += pred_word_occur_cnt
                    word_total += cnt

                if self.kwargs['output']:
                    print(json.dumps({
                        'hyp': hyp,
                        'ref': ref,
                        'gold_deps': gold_deps,
                        'word_pairs': word_pairs,
                        'predict_deps': predict_deps,
                        'all_deps': all_deps
                    }, ensure_ascii=False))
                
                tp_list.append(tp)
                e3_cnt_list.append(e3_cnt)
                total_list.append(total)
                word_occur_list.append(word_occur)
                word_total_list.append(word_total)
        
        UC = (sum(tp_list) + sum(e3_cnt_list)) / sum(total_list)
        LC = sum(tp_list) / sum(total_list)
        word_coverage = sum(word_occur_list) / sum(word_total_list)

        print('======== Dependency Metrics =========')
        print(f'Unlabeled Coverage: {UC * 100:.2f}')
        print(f'Labeled Coverage: {LC * 100:.2f}')
        # print(f'1 dep coverage: {tp1 / total1 * 100:.2f} ({tp1}/{total1})')
        # print(f'>=2 dep coverage: {(tp-tp1) / (total-total1) * 100:.2f} ({tp-tp1}/{total-total1})')
        # print(f'error type counts: {e1_cnt} {e2_cnt} {e3_cnt}')
        print(f'Word Coverage: {word_coverage * 100:.2f}')

        if return_list:
            WC_list = [x/y for x, y in zip(word_occur_list, word_total_list)]
            UC_list = [(x+y)/z for x, y, z in zip(tp_list, e3_cnt_list, total_list)]
            LC_list = [x/y for x, y in zip(tp_list, total_list)]
            return WC_list, UC_list, LC_list

    def get_list_of_scores(self, hyps, refs, all_gold_deps):
        return self.get_score(hyps, refs, all_gold_deps, return_list=True)