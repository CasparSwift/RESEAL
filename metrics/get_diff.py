import json


def get_json(filename):
	objs = []
	with open(filename, 'r') as f:
		for line in f:
			try:
				objs.append(json.loads(line.strip()))
			except:
				pass
	return objs

objs1 = get_json('out_base.json')
objs2 = get_json('out_dba.json')
objs3 = get_json('out_rerank.json')
objs4 = get_json('out_prob.json')

d = {
	'total_overlap': 0,
	'total_non_overlap': 0,
	'error_overlap': 0,
	'error_non_overlap': 0
}
for obj1, obj2, obj3, obj4 in zip(objs1, objs2, objs3, objs4):
	if len(obj1['gold_deps']) > 1:
		word_set = set()
		for w1, r, w2 in obj2['gold_deps']:
			word_set.add(w1)
			word_set.add(w2)
		has_overlap_words = (len(list(word_set)) != 2 * len(obj2['gold_deps']))
		if has_overlap_words:
			d['total_overlap'] += 1
		else:
			d['total_non_overlap'] += 1
		for dep2 in obj1['gold_deps']:
			if dep2 not in obj4['predict_deps'] and dep2 in obj2['predict_deps']:
				if has_overlap_words:
					d['error_overlap'] += 1
				else:
					d['error_non_overlap'] += 1		
				# if len(obj1['predict_deps']) > 0:
				# 	continue
				print('hyp1: ', obj1['hyp'])
				print('hyp2: ', obj2['hyp'])
				print('hyp3: ', obj3['hyp'])
				print('hyp4: ', obj4['hyp'])
				print('ref: ', obj1['ref'])
				print(dep2)
				print('pred1: ', obj1['predict_deps'])
				print('pred2: ', obj2['predict_deps'])
				print('pred3: ', obj3['predict_deps'])
				print('pred4: ', obj4['predict_deps'])
				print('gold_dep', obj2['gold_deps'])
				print('-'*50)

print(d)
print(d['error_overlap'] / d['total_overlap'])
print(d['error_non_overlap'] / d['total_non_overlap'])


