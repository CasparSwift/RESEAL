import json


def get_all_deps(doc):
	all_deps = []
	for sentence in doc:
		for word in sentence:
			w2 = word['text']
			w2_pos = word['upos']
			r = word['deprel']
			head_id = word['head'] - 1
			head_word = sentence[head_id]
			w1 = head_word['text']
			w1_pos = head_word['upos']
			all_deps.append([[w1, w1_pos], r, [w2, w2_pos]])
	return all_deps

tp = 0
total_pred = 0
total = 0


counts = {}
counts2 = {}
with open('Dataset/Giga-Dep/test.json', 'r') as f:
	for line in f:
		obj = json.loads(line.strip())
		src_deps = obj['src_deps']
		ref_deps = obj['ref_deps']
		gold_deps = obj['dependencies']
		# for dep in deps:
		# 	counts[dep[1]] = counts.get(dep[1], 0) + 1
		# for dep in src_deps:
		# 	counts2[dep[1]] = counts2.get(dep[1], 0) + 1
		src_deps = get_all_deps(obj['src_doc'])
		ref_deps = get_all_deps(obj['ref_doc'])
		# print(src_deps)
		# print(ref_deps)
		inter_deps = []
		for dep in src_deps:
			(w1, w1_pos), r, (w2, w2_pos) = dep
			if [w1, r, w2] in gold_deps:
				inter_deps.append(dep)
		# print(inter_deps)
		# exit()
		for dep in inter_deps:
			(w1, w1_pos), r, (w2, w2_pos) = dep
			counts[(w1, r, w2)] = counts.get((w1, r, w2), 0) + 1
		for dep in src_deps:
			(w1, w1_pos), r, (w2, w2_pos) = dep
			counts2[(w1, r, w2)] = counts2.get((w1, r, w2), 0) + 1
		pred_deps = []
		for dep in src_deps:
			(w1, w1_pos), r, (w2, w2_pos) = dep
			if w1_pos == 'PROPN' and w2_pos == 'PROPN' and r == 'amod':
				pred_deps.append(dep)

		for dep in inter_deps:
			if dep in pred_deps:
				tp += 1
		total_pred += len(pred_deps)
		total += len(inter_deps)


precision = tp / (total_pred + 1e-10)
recall = tp / (total + 1e-10)
F1 = 2 * precision * recall / (precision + recall + 1e-10)
print(f'{precision*100:.2f}, {recall*100:.2f}, {F1*100:.2f}')

result = {}
for key, c in counts.items():
	# print(key, c/counts2[key])
	result[key] = (c / counts2[key], c, counts2[key])


for key, item in sorted(result.items(), key=lambda x: x[1][0], reverse=True):
	print(key, item)




