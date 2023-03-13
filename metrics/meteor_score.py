import os

class METEOR_Score_Evaluator(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_score(self, hyps, refs):
        with open('tmp.hyp', 'w') as f:
            f.write('\n'.join(hyps).lower())
        with open('tmp.ref', 'w') as f:
            f.write('\n'.join(refs).lower())
        os.system("cd metrics/meteor-1.5/ && java -jar meteor-1.5.jar ../../tmp.hyp ../../tmp.ref -l en -norm -a data/paraphrase-en.gz")
