import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

def get_json(filename):
	objs = []
	with open(filename, 'r') as f:
		for line in f:
			try:
				objs.append(json.loads(line.strip()))
			except:
				pass
	return objs

objs = get_json('out1.json')

points = []
avg_spearman = 0
cnt = 0

spearman_and_acc = []
for obj in objs:
	ref_words = obj['ref'].lower().split()
	hyp_words = obj['hyp'].lower().split()

	word_list = []
	for w1, r, w2 in obj['gold_deps']:
		word_list.append(w1)
		word_list.append(w2)

	# print(ref_words)
	# print(hyp_words)
	# print(word_list)
	X = []
	Y = []
	for w in set(word_list):
		try:
			x = ref_words.index(w) / len(ref_words)
			y = hyp_words.index(w) / len(hyp_words)
		except:
			pass
		# print(x, y)
		points.append([x, y])
		X.append(x)
		Y.append(y)

	if len(X) > 1 and len(Y) > 1:
		s_r = spearmanr(X, Y)[0]

		if str(s_r) == 'nan':
			print(X, Y, obj)
			exit()
		
		avg_spearman += s_r
		cnt += 1

		tp = 0
		for dep in obj['gold_deps']:
			if dep in obj['predict_deps']:
				tp += 1
		acc = tp / len(obj['gold_deps'])

		spearman_and_acc.append([s_r < 0.99, acc])

print(avg_spearman / cnt)

def plot(x, y):
	print(pearsonr(x, y))
	plt.scatter(x, y, marker='.')
	plt.show()

points = np.array(points)
print(points.shape)
# plot(points[:, 0], points[:, 1])

spearman_and_acc = np.array(spearman_and_acc)
print(spearman_and_acc.shape)
plot(spearman_and_acc[:, 0], spearman_and_acc[:, 1])


