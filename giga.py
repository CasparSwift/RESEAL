import pickle
from my_utils import read_conllu
from parser_api import StanzaParser
from metrics.metric_utils import get_predict_and_all_deps
from tqdm import tqdm
import json

sources = []
with open('Dataset/GigaWord/test.article.txt.str', 'r') as f:
	for line in f:
		sources.append(line.strip())

refs = []
with open('Dataset/GigaWord/test.title.txt.str', 'r') as f:
	for line in f:
		refs.append(line.strip())

hyps = []
with open('output/gigaword_1019/bart_gigaword_e5_infer.out', 'r') as f:
	for line in f:
		hyps.append(line.strip())		

parser = StanzaParser()

for src, ref, hyp in zip(sources, refs, hyps):
	docs = parser.parse([src, ref, hyp])
	_, src_deps = get_predict_and_all_deps(docs[0], [])
	_, ref_deps = get_predict_and_all_deps(docs[1], [])
	_, hyp_deps = get_predict_and_all_deps(docs[2], [])
	src_deps = [tuple(dep) for dep in src_deps]
	ref_deps = [tuple(dep) for dep in ref_deps]
	hyp_deps = [tuple(dep) for dep in hyp_deps]
	intersection1 = set(src_deps) & set(ref_deps)
	intersection2 = set(src_deps) & set(hyp_deps)
	# print(src)
	# print(ref)
	# print(hyp)
	# print(intersection1)
	# print(intersection2)
	# print('-'*20)
	data = {
		'dependencies': list(intersection1),
		'text': src,
		'ref': ref
	}
	print(json.dumps(data, ensure_ascii=False))

	# result_dict = {
	# 	'src': src,
	# 	'ref': ref,
	# 	'src_deps': src_deps,
	# 	'ref_deps': ref_deps,
	# 	'intersection': list(intersection)
	# }
	# print(json.dumps(result_dict, ensure_ascii=False))


	# exit()
