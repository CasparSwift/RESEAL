from pyrouge import Rouge155
import os
import re


class Rouge155_Evaluator(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.r = Rouge155()

    def get_score(self, hyps, refs):
        os.system('rm -r eval_output')
        os.system('mkdir -p eval_output/hyps')
        os.system('mkdir -p eval_output/refs')

        for i, (ref, hyp) in enumerate(zip(refs, hyps)):
            with open(f'./eval_output/hyps/hyps.{i}.txt', 'w') as f:
                if hyp.startswith(' '):
                    hyp = hyp[1:]
                assert len(hyp) > 0
                hyp = re.sub(r'\s+', ' ', hyp)
                # if 'unk' in hyp:
                    # print(hyp)
                    # continue
                f.write(hyp)
            if isinstance(ref, list):
                for j, r in enumerate(ref):
                    c = chr(ord('A') + j)
                    with open(f'./eval_output/refs/refs.{c}.{i}.txt', 'w') as f:
                        r = re.sub(r'\d', '#', r)
                        # if len(r) < 10:
                        #     print(r)
                        #     continue
                        f.write(r)
            else:
                with open(f'./eval_output/refs/refs.A.{i}.txt', 'w') as f:
                    f.write(ref)

        self.r.system_dir = './eval_output/hyps'
        self.r.model_dir = './eval_output/refs'
        self.r.system_filename_pattern = 'hyps.(\d+).txt'
        self.r.model_filename_pattern = 'refs.[A-Z].#ID#.txt'

        # try:
            # output = self.r.convert_and_evaluate(rouge_args='-e /Users/casparswift/rouge/RELEASE-1.5.5/data -a -m -n 2 -w 1.2')
            # output_dict = self.r.output_to_dict(output)
            # keys = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_l_f_score', 'rouge_1_recall', 'rouge_2_recall', 'rouge_l_recall']
            # for key in keys:
            #     print(key, output_dict[key])
            # exit()
        # except:
        system_input_dir = './eval_output/hyps'
        model_input_dir = './eval_output/refs'
        system_output_dir = './eval_output/hyps_output'
        model_output_dir = './eval_output/refs_output'
        system_filename_pattern = 'hyps.(\d+).txt'
        model_filename_pattern = 'refs.[A-Z].#ID#.txt'
        config_file_path = './eval_output/config.xml'

        Rouge155.convert_summaries_to_rouge_format(system_input_dir, system_output_dir)
        Rouge155.convert_summaries_to_rouge_format(model_input_dir, model_output_dir)
        Rouge155.write_config_static(
            system_output_dir, system_filename_pattern,
            model_output_dir, model_filename_pattern,
            config_file_path
        )
        options = '-b 75 ' if self.kwargs['giga_test_set'] == 'duc' else ''
        cmd = f'/Users/casparswift/rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e /Users/casparswift/rouge/RELEASE-1.5.5/data -a -m {options}-n 2 -w 1.2 -m eval_output/config.xml'
        os.system(cmd)
        print(cmd)