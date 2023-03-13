import os

class BERT_Score_Evaluator(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_score(self, hyps, refs):
        with open('tmp.hyp', 'w') as f:
            f.write('\n'.join(hyps))
        with open('tmp.ref', 'w') as f:
            f.write('\n'.join(refs))
        os.system("CUDA_VISIBLE_DEVICES=3 bert-score -r tmp.ref -c tmp.hyp --model /data2/pretrain/roberta-large/ --num_layers=17 --lang en")
