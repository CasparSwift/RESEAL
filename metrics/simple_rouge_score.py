from rouge import Rouge


class Simple_Rouge_Evaluator(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.evaluator = Rouge()

    def get_score(self, hyps, refs, avg=True):
        # hyps = [' '.join(map(str, i)) for i in hyps]
        # refs = [' '.join(map(str, i)) for i in refs]
        # for hyp, ref in zip(hyps, refs):
            # scores = self.evaluator.get_scores([hyp], [ref], avg=avg)
            # f = scores['rouge-l']['f']
            # if f < 0.4:
            #     print(hyp)
            #     print(ref)
            #     print(f)
            #     print('-'*30)
        scores = self.evaluator.get_scores(hyps, refs, avg=avg)
        print('rouge-1', scores['rouge-1']['f'])
        print('rouge-2', scores['rouge-2']['f'])
        print('rouge-l', scores['rouge-l']['f'])
