from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
from .metric_utils import batched_list


class GPT2_PPL_Evaluator(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = self.kwargs['device']
        self.model = GPT2LMHeadModel.from_pretrained(self.kwargs['model_dir']).to(self.device)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.kwargs['model_dir'])
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def gpt2score(self, sentence):
        with torch.no_grad():
            batch_size = len(sentence)
            tensor_input = self.tokenizer(sentence, return_tensors='pt', padding=True)
            tensor_input = {k: v.to(self.device) for k, v in tensor_input.items()}
            
            logits = self.model(**tensor_input, labels=tensor_input['input_ids'].clone()).logits
            lprobs = F.log_softmax(logits, dim=-1)

            target = tensor_input['input_ids'][:, 1:].unsqueeze(-1)
            target_mask = tensor_input['attention_mask'][:, 1:]
            nll_loss = -lprobs.gather(dim=-1, index=target)

            real_lengths = torch.sum(target_mask, dim=-1)
            nll_loss = torch.sum((nll_loss.view(batch_size, -1) * target_mask), dim=-1)

            nll_loss = torch.div(nll_loss, real_lengths)
            all_ppl = torch.exp(nll_loss)
            ppl = all_ppl.mean().item()
        return ppl, all_ppl.tolist()

    def get_score(self, hyps, refs=None, all_gold_deps=None): 
        total_ppls = 0.
        total_sentences = 0
        for batched_hyp in tqdm(batched_list(hyps, batch_size=self.kwargs['batch_size'])):
            new_batched_hyp = []
            for hyp in batched_hyp:
                if hyp.startswith(' '):
                    hyp = hyp[1:]
                if not hyp:
                    continue
                new_batched_hyp.append(hyp.lower())
            ppl, all_ppl = self.gpt2score(new_batched_hyp)
            total_ppls += ppl * len(new_batched_hyp)
            total_sentences += len(new_batched_hyp)
            if self.kwargs['verbose']:
                for p, hyp in zip(all_ppl, batched_hyp):
                    print(str(p) + '\t' + hyp, file=sys.stderr)

        print('======== GPT-2 PPL Metrics =========')
        print(f'GPT-2 PPL: {total_ppls / total_sentences:.2f} ({total_sentences})')

    def get_list_of_scores(self, hyps, refs=None, all_gold_deps=None): 
        total_ppls = 0.
        total_sentences = 0
        scores_list = []
        for batched_hyp in tqdm(batched_list(hyps, batch_size=self.kwargs['batch_size'])):
            new_batched_hyp = []
            for hyp in batched_hyp:
                if hyp.startswith(' '):
                    hyp = hyp[1:]
                if not hyp:
                    continue
                new_batched_hyp.append(hyp.lower())
            ppl, all_ppl = self.gpt2score(new_batched_hyp)
            # print("all_ppl is : ", all_ppl)
            scores_list.extend(all_ppl)
        return scores_list
