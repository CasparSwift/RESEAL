from data import EnglishEWTDependencyDataset
from tqdm import tqdm
import numpy as np
import argparse
import sys
from metrics.dependency_coverage import Dependency_Coverage_Evaluator
from metrics.gpt2_ppl import GPT2_PPL_Evaluator
from metrics.rouge_score import Rouge155_Evaluator
from metrics.simple_rouge_score import Simple_Rouge_Evaluator
from metrics.bleu_score import BLEU_Evaluator
from metrics.bart_score import BARTScore_Evaluator
from metrics.meteor_score import METEOR_Score_Evaluator
from metrics.bert_score import BERT_Score_Evaluator
import torch


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def simple_rouge(filename, Dataset):
    import rouge
    test_set = Dataset(None, None, 'test')
    all_refs = [Y for _, Y in test_set.data]
    with open(filename, 'r') as f:
        all_hyps = f.read().strip('\n').split('\n')

    for aggregator in ['Avg', 'Best']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                               max_n=2,
                               limit_length=False,
                               length_limit=100,
                               length_limit_type='words',
                               apply_avg=apply_avg,
                               apply_best=apply_best,
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               stemming=False)

        scores = evaluator.get_scores(all_hyps, all_refs)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
                print()
            else:
                # print(results)
                print(prepare_results(metric, results['p'], results['r'], results['f']))
        print()


def bert_ppl(filename, model_dir):
    from transformers import BertTokenizer, BertForMaskedLM
    import torch

    with torch.no_grad():
        model = BertForMaskedLM.from_pretrained(model_dir)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(model_dir)

        def score(sentence, mask_token_id=103):
            tensor_input = tokenizer(sentence, return_tensors='pt')['input_ids']
            repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
            mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
            masked_input = repeat_input.masked_fill(mask == 1, mask_token_id)
            loss = model(masked_input, labels=repeat_input).loss
            result = np.exp(loss.item())
            return result

        with open(filename, 'r') as f:
            all_hyps = f.read().strip('\n').split('\n')

        total_ppls = 0.
        for hyp in tqdm(all_hyps):
            ppl = score(hyp)
            total_ppls += ppl
            print(ppl)

    print('======== BERT PPL Metrics =========', file=sys.stderr)
    print(f'BERT PPL: {total_ppls / len(all_hyps):.2f}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--model_dir', type=str, default='/home/yangzhixian/pretrained_model/gpt2-base')
    parser.add_argument('-n', type=int, default=None, required=False)
    parser.add_argument('-p', action='store_true')
    parser.add_argument('-b', action='store_true')
    parser.add_argument('-c', action='store_true')
    parser.add_argument('-r', action='store_true')
    parser.add_argument('-sr', action='store_true')
    parser.add_argument('-meteor', action='store_true')
    parser.add_argument('-bartscore', action='store_true')
    parser.add_argument('-bertscore', action='store_true')
    parser.add_argument('--output', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--giga_test_set', type=str, default='origin')
    parser.add_argument('--parser', type=str, default='stanza', choices=['stanza', 'spacy'])

    args = parser.parse_args()

    if args.r or args.sr or args.bartscore:
        if args.giga_test_set == 'origin':
            postfix = 'txt.str'
            tgt_path = f'Dataset/GigaWord/test.title.{postfix}'
            with open(tgt_path, 'r') as f:
                all_refs = f.read().strip('\n').split('\n')
        elif args.giga_test_set == 'internal':
            postfix = 'filter.txt.2k'
            tgt_path = f'Dataset/GigaWord/test.title.{postfix}'
            with open(tgt_path, 'r') as f:
                all_refs = f.read().strip('\n').split('\n')
        elif args.giga_test_set == 'duc':
            all_refs = [[] for _ in range(500)]
            for ref_num in range(4):
                tgt_path = f'Dataset/duc04/task1_ref{ref_num}.txt.tok'
                with open(tgt_path, 'r') as f:
                    for i, line in enumerate(f):
                        all_refs[i].append(line.strip())
        elif args.giga_test_set == 'MSR':
            all_refs = [[] for _ in range(785)]
            for ref_num in range(1, 6):
                tgt_path = f'Dataset/MSR/ref{ref_num}.txt'
                with open(tgt_path, 'r') as f:
                    for i, line in enumerate(f):
                        if line.strip() == 'EMPTY':
                            continue
                        all_refs[i].append(line.strip())
        else:
            exit('giga test set not exists!')
    else:
        test_set = EnglishEWTDependencyDataset(None, None, 'test')
        all_refs = [Y for _, Y in test_set.data]
        objs = test_set.metadata
        all_deps = [obj["dependencies"] for obj in objs]

    with open(args.path, 'r') as f:
        all_hyps = f.read().strip('\n').split('\n')

    if args.n:
        all_hyps, all_refs, all_deps = all_hyps[:args.n], all_refs[:args.n], all_deps[:args.n]
        assert len(all_hyps) == len(all_refs) == len(all_deps), f"{len(all_hyps)} {len(all_refs)} {len(all_deps)}"
    
    print(f'samples: {len(all_hyps)} {len(all_refs)}')
    assert len(all_hyps) == len(all_refs), f"{len(all_hyps)} {len(all_refs)}"
    

    kwargs = {
        'batch_size': 16, 
        'device': torch.device('cuda:0'),
        'verbose': args.verbose,
        'output': args.output,
        'model_dir': args.model_dir,
        'use_gpu': True,
        'parser': args.parser,
        'giga_test_set': args.giga_test_set
    }

    if args.p:
        e = GPT2_PPL_Evaluator(**kwargs)
        e.get_score(hyps=all_hyps)

    if args.c:
        e = Dependency_Coverage_Evaluator(**kwargs)
        e.get_score(hyps=all_hyps, refs=all_refs, all_gold_deps=all_deps)

    if args.r:
        e = Rouge155_Evaluator(**kwargs)
        e.get_score(hyps=all_hyps, refs=all_refs)

    if args.sr:
        e = Simple_Rouge_Evaluator(**kwargs)
        e.get_score(hyps=all_hyps, refs=all_refs)

    if args.b:
        e = BLEU_Evaluator(**kwargs)
        e.get_score(hyps=all_hyps, refs=all_refs)

    if args.bartscore:
        e = BARTScore_Evaluator(**kwargs)
        e.get_score(hyps=all_hyps, refs=all_refs)
    
    if args.meteor:
        e = METEOR_Score_Evaluator(**kwargs)
        e.get_score(hyps=all_hyps, refs=all_refs)
    
    if args.bertscore:
        e = BERT_Score_Evaluator(**kwargs)
        e.get_score(hyps=all_hyps, refs=all_refs)
 


if __name__ == '__main__':
    main()

