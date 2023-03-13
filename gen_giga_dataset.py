from my_utils import read_conllu
from parser_api import LeftToRightPointerParser
from metrics.metric_utils import get_predict_and_all_deps
from tqdm import tqdm
import json
import re


giga_test_set = 'duc'

path_ = {
	'duc': {
		'source_path': 'Dataset/duc04/duc04.src',
		'target_path': [f'Dataset/duc04/task1_ref{i}.txt.tok' for i in range(4)]
	}
}

def read_data(path):
	datas = []
	with open(path, 'r') as f:
		for line in f:
			line = re.sub(r'\d', '#', line)
			datas.append(line.strip())
	return datas

source_data = read_data(path_[giga_test_set]['source_path'])

target_data = (read_data(p) for p in path_[giga_test_set]['target_path'])
target_data = list(zip(*target_data))

# sources = []
# with open('Dataset/MSR/msr.src', 'r') as f:
# 	for line in f:
# 		line = re.sub(r'\d', '#', line)
# 		sources.append(line.strip())

# refs = []
# with open('Dataset/MSR/ref1.txt', 'r') as f:
# 	for line in f:
# 		line = re.sub(r'\d', '#', line)
# 		refs.append(line.strip())


def handle_batch(parser, src_batch, refs_batch, refs_id, f):
	all_src_deps = parser.get_all_deps(src_batch)
	all_ref_deps = parser.get_all_deps(refs_batch)
	src_id2ref_id = {}
	for ref_id, src_id in enumerate(refs_id):
		src_id2ref_id[src_id] = src_id2ref_id.get(src_id, []) + [ref_id]
	for i, (src_deps, src) in enumerate(zip(all_src_deps, src_batch)):
		src_words = [word for _, _, word in src_deps]
		refs = [refs_batch[ref_id] for ref_id in src_id2ref_id[i]]
		refs_deps = [all_ref_deps[ref_id] for ref_id in src_id2ref_id[i]]
		# Vote for the golden dependencies and keywords
		intersection_src_ref = []
		intersection_keywords = []
		keyword_labels = []
		for head, rel, word in src_deps:
			pos_vote, neg_vote = 0, 0
			pos_vote_kw, neg_vote_kw = 0, 0
			for ref_dep in refs_deps:
				all_words = set(dep[-1] for dep in ref_dep)
				if (head, rel, word) in ref_dep:
					pos_vote += 1
				else:
					neg_vote += 1
				if word in all_words:
					pos_vote_kw += 1
				else:
					neg_vote_kw += 1
			if pos_vote >= neg_vote:
				if (head, rel, word) not in intersection_src_ref:
					intersection_src_ref.append((head, rel, word))
			if pos_vote_kw >= neg_vote_kw:
				keyword_labels.append(1)
				if word not in intersection_keywords:
					intersection_keywords.append(word)
			else:
				keyword_labels.append(0)
		assert len(keyword_labels) == len(src_deps)

		# build phrases
		intersection_keyphrases = []
		phrase = []
		for word, label in zip(src_words, keyword_labels):
			if label:
				phrase.append(word)
			else:
				if phrase:
					intersection_keyphrases.append(' '.join(phrase))
					phrase = []
		if phrase:
			intersection_keyphrases.append(' '.join(phrase))

		result_dict = {
			'text': src,
			'ref': refs[0],
			'src_deps': src_deps,
			'ref_deps': refs_deps,
			'dependencies': intersection_src_ref,
			'keywords': intersection_keywords,
			'keyphrases': list(set(intersection_keyphrases))
		}
		f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

parser = LeftToRightPointerParser()

with open('/dataA/chenxiang/giga/test_duc_full.json', 'w') as f:
	batch, refs_batch, refs_id = [], [], []
	for src, refs in tqdm(zip(source_data, target_data)):
		batch.append(src)
		for ref in refs:
			if not ref:
				continue
			refs_batch.append(ref)
			refs_id.append(len(batch)-1)
		if len(batch) < 50:
			continue
		handle_batch(parser, batch, refs_batch, refs_id, f)
		batch, refs_batch, refs_id = [], [], []

	if batch:
		handle_batch(parser, batch, refs_batch, refs_id, f)

