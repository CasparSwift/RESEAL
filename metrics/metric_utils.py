
def lower_dep(deps):
	return [[w1.lower(), r, w2.lower()] for w1, r, w2 in deps]


def get_word_pairs(deps):
	return [(w1, w2) for w1, r, w2 in deps]


def batched_list(hyps, batch_size=1, refs=None, deps=None):
    length = len(hyps)
    items = tuple(item for item in [hyps, refs, deps] if item is not None)
    if len(items) == 1:
        return [items[0][idx: idx+batch_size] 
            for idx in range(0, length, batch_size)]
    else:
        return [tuple(item[idx: idx+batch_size] for item in items) 
            for idx in range(0, length, batch_size)]


def get_predict_and_all_deps(doc, word_pairs):
    all_deps = []
    predict_deps = []
    for sent in doc.sentences:
        for word in sent.words:
            head_word_idx = word.head
            if head_word_idx == 0:
                w1 = '_ROOT'
            else:
                head_word = sent.words[head_word_idx - 1]
                w1 = head_word.text.lower()
            r = word.deprel
            w2 = word.text.lower()
            if (w1, w2) in word_pairs:
                predict_deps.append([w1, r, w2])
            all_deps.append([w1, r, w2])
    return predict_deps, all_deps


def get_predict_and_all_deps_from_json(doc, word_pairs):
    all_deps = []
    predict_deps = []
    for sent in doc:
        for word in sent:
            head_word_idx = word['head']
            if head_word_idx == 0:
                continue
            head_word = sent[head_word_idx - 1]
            w1 = head_word['text'].lower()
            r = word['deprel']
            w2 = word['text'].lower()
            if (w1, w2) in word_pairs:
                predict_deps.append([w1, r, w2])
            all_deps.append([w1, r, w2])
    return predict_deps, all_deps
