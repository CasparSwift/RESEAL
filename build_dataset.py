import random
import sys
import json
import os

random.seed(42)

def read_from_conll(split):
	datas = []
	with open(f'Dataset/UD_English-EWT/en_ewt-ud-{split}.conllu', 'r') as f:
		cur_sentence = []
		for line in f:
			if line.startswith('#'):
				continue
			if line == '\n':
				datas.append(cur_sentence)
				cur_sentence = []
				continue
			items = line.strip().split('\t')
			if not items[0].isdigit():
				continue
			word_info = {
				"word": items[1],
				"ori_word": items[2],
				"upos": items[3],
				"xpos": items[4],
				"head": int(items[6]),
				"dependency": items[7]
			}
			cur_sentence.append(word_info)
	return datas


def print_dataset_stats(new_datas):
	total_len, word_total_len, dep_total = len(new_datas), 0, 0
	dep_cnt = {}
	for data in new_datas:
		word_total_len += len(data["text"].split(" "))
		dep_total += len(data["dependencies"])
		for _, dep, _ in data["dependencies"]:
			dep_cnt[dep] = dep_cnt.get(dep, 0) + 1
	dep_cnt = dict(sorted(dep_cnt.items(), key=lambda x: x[-1], reverse=True))

	max_dep_num = max(len(data["dependencies"]) for data in new_datas)
	min_dep_num = min(len(data["dependencies"]) for data in new_datas)

	print("======= Dataset Stats =======", file=sys.stderr)
	print(f"sample: {total_len}", file=sys.stderr)
	print(f"avg sentence length: {word_total_len/total_len:.2f}", file=sys.stderr)
	print(f"avg number of dependencies: {dep_total/total_len:.2f}", file=sys.stderr)
	print(f"max number of dependencies: {max_dep_num}", file=sys.stderr)
	print(f"min number of dependencies: {min_dep_num}", file=sys.stderr)
	print(f"dependencies count: {dep_cnt}", file=sys.stderr)


def main():
	selected_pos = ["ADJ", "NOUN", "VERB"]
	os.mkdir(sys.argv[1])
	
	for split in ['train', 'dev', 'test']:
		new_datas = []
		datas = read_from_conll(split)
		for data in datas:
			word_list = [info["word"] for info in data]
			text = " ".join(word_list)
			if "01-Feb-02" in text:
				continue
			valid_dependencies = []
			for word_info in data:
				if word_info["upos"] in selected_pos:
					head_id = word_info["head"]
					# skip <root> tokens
					if head_id == 0:
						continue
					head_word_info = data[head_id - 1]
					# skip when both words are proper nouns
					if word_info["xpos"] in ["NNP", "NNPS"] and head_word_info["xpos"] in ["NNP", "NNPS"]:
						continue
					# skip number (may be phone numbers or else personal information)
					if word_info["word"].replace('/', '').replace('-', '').isdigit():
						continue
					if head_word_info["word"].replace('/', '').replace('-', '').isdigit():
						continue
					head_word = head_word_info["word"]
					dep = word_info["dependency"]
					# skip root dependency
					if dep == 'root':
						continue
					valid_dependencies.append([head_word, dep, word_info["word"]])
					# same word, must appear >= 2 times
					if head_word == word_info["word"]:
						if word_list.count(head_word) < 2:
							continue
						# print(valid_dependencies[-1])
						# print(text)
						# print('*'*50)

			if valid_dependencies:
				if len(valid_dependencies) > 1:
					number = round(0.4 * len(valid_dependencies))
					valid_dependencies = random.sample(valid_dependencies, number)
				# skip short sentences
				if len(word_list) <= 3 and len(valid_dependencies) == 1:
					continue
				new_datas.append({
					"dependencies": valid_dependencies,
					"text": text
				})
		print_dataset_stats(new_datas)
		with open(os.path.join(sys.argv[1], split + '.json'), 'w') as f:
			for data in new_datas:
				f.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
	main()