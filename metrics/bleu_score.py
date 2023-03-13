import os
import subprocess
import math
import sacrebleu

class BLEU_Evaluator(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_score(self, hyps, refs):
        line = str(sacrebleu.corpus_bleu(hyps, [refs], lowercase=True))
        print(line)
        p = line.split(' (')[0].split(' ')[-1].split('/')
        BP = float(line.split('BP = ')[-1].split(' ')[0])
        p = list(map(float, p))
        for n in range(1, 5):
            bleu = 100 * BP * math.exp(sum(math.log(p[i]/100) for i in range(n))/n)
            print(f'BLEU-{n}: {bleu:.2f}')