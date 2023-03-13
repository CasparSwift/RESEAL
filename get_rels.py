import pdb
import os, json
from collections import Counter

from data import EditingRelationDataset
class args:
    max_src_length = 128
train_set = EditingRelationDataset(args, None, 'train')

all_rels = []
for obj in train_set.metadata:
    for h, r, t in obj['triple']:
        if '@@ROOT@@' not in r:
            all_rels.append(' '.join(r))

print(Counter(all_rels))

pdb.set_trace()

# cnt = {'country': 17525, 'location': 11694, 'leader name': 6444, 'is part of': 4074, 'language': 3075, 'region': 3031, 'ethnic group': 2794, 'ingredient': 2745, 'birth place': 2331, 'creator': 2085, 'city served': 2027, 'capital': 1818, 'runway length': 1785, 'operating organisation': 1567, 'club': 1522, 'completion date': 1521, 'elevation above the sea level in metres': 1449, 'media type': 1418, 'author': 1406, 'main ingredients': 1341, 'floor count': 1315, 'architect': 1210, 'ground': 1126, 'nationality': 1106, 'city': 1094, 'established': 1034, 'publisher': 1007, 'runway name': 988, 'dish variation': 982, 'isbn number': 959, 'manager': 958, 'state': 873, 'oclc number': 739, 'number of pages': 713, 'course': 706, 'owner': 701, 'number of members': 693, 'affiliation': 680, 'was a crew member of': 633, 'alma mater': 632, 'abbreviation': 621, 'demonym': 616, 'alternative name': 603, 'leader title': 578, 'birth date': 558, 'leader': 549, 'preceded by': 543, 'currency': 469, 'death place': 446, 'occupation': 437, 'floor area': 423, 'order': 423, 'category': 409, 'tenant': 408, 'issn number': 380, 'lccn number': 368, 'league': 353, 'followed by': 348, 'status': 339, 'number of students': 339, 'date of retirement': 336, 'was selected by nasa': 324, 'season': 303, 'academic discipline': 303, 'starring': 284, 'academic staff size': 284, '1st runway surface type': 275, 'significant building': 275, 'operator': 273, 'fullname': 235, '3rd runway surface type': 216, 'building start date': 210, 'spoken in': 204, 'division': 201, 'battles': 196, 'editor': 192, 'municipality': 191, 'broadcasted by': 185, 'year of construction': 184, 'current tenants': 180, 'governing body': 168, 'parent company': 166, 'death date': 165, 'district': 158, 'largest city': 154, 'champions': 154, 'coden code': 145, 'family': 137, 'inauguration date': 131, 'architectural style': 129, 'first aired': 128, 'added to the national register of historic places': 126, 'full name': 121, 'class': 118, 'headquarter': 112, 'reference number in the national register of historic places': 111, 'time in space': 99, 'foundation place': 98, 'aircraft fighter': 96, 'influenced by': 95, 'material': 94, 'backup pilot': 93, 'hometown': 85, 'location city': 83, 'icao location identifier': 83, 'designer': 83, 'crew members': 81, 'cost': 80, 'series': 75, 'first publication year': 73, 'postal code': 72, 'dean': 70, 'address': 69, 'jurisdiction': 68, 'awards': 67, 'serving temperature': 66, 'official language': 63, 'doctoral advisor': 56, 'number of postgraduate students': 53, 'height': 52, 'nickname': 52, 'key person': 50, 'residence': 50, 'has to its southeast': 46, 'award': 44, 'affiliations': 44, 'owning organisation': 43, 'has to its west': 42, 'anthem': 42, '2nd runway surface type': 41, 'title': 40, 'voice': 40, 'carbohydrate': 39, 'campus': 38, 'transport aircraft': 36, 'has to its north': 35, 'government type': 35, 'bed count': 35, 'attack aircraft': 34, 'founder': 33, 'fat': 33, 'iata location identifier': 32, 'administrative county': 32, 'architecture': 32, 'director': 32, 'region served': 32, 'president': 31, 'legislature': 31, 'place of birth': 30, 'first appearance in film': 30, 'compete in': 30, 'part': 29, 'elevation above the sea level in feet': 29, 'nearest city': 28, 'headquarters': 28, 'hub airport': 28, 'protein': 27, 'commander': 27, 'served as chief of the astronaut office in': 27, 'has to its southwest': 26, 'religion': 26, 'representative': 26, 'genus': 25, 'significant project': 24, 'rector': 24, 'notable work': 24, 'aircraft helicopter': 23, 'higher': 23, 'distributor': 21, 'neighboring municipality': 21, 'place of death': 20, 'location identifier': 20, 'mayor': 19, 'parts type': 17, 'was given the technical campus status by': 16, 'has to its northeast': 15, 'child': 15, 'motto': 15, 'latin name': 15, 'leader party': 14, '1st runway length feet': 13, 'chief': 13, '3rd runway length feet': 13, 'impact factor': 13, 'chairman': 12, 'gemstone': 11, 'founded by': 10, 'senators': 10, 'has to its northwest': 9, 'bird': 9, 'building type': 9, 'year': 8, 'alternative names': 7, '1st runway number': 7, 'county seat': 7, 'frequency': 7, 'river': 6, 'similar dish': 6, 'eissn number': 6, 'number of rooms': 6, 'administrative arrondissement': 6, '5th runway number': 5, 'website': 5, 'former name': 5, 'developer': 5, 'youthclub': 5, 'patron saint': 4, 'related': 4, 'last aired': 4, 'number of undergraduate students': 4, 'libraryof congress classification': 4, 'outlook ranking': 4, 'official school colour': 4, '5th runway surface type': 4, 'was awarded': 4, 'birth name': 4, 'chancellor': 3, 'genre': 3, 'served': 2, '4th runway surface type': 2, '4th runway length feet': 2, 'product': 2}
# with open('rel_cnt.txt', 'w') as f:
#     f.write(json.dumps(cnt, ensure_ascii=False))

# exit()


def get_rels(split):
    rels = []
    with open(os.path.join('Dataset/webedit', f'{split}.jsonl'), 'r') as f:
        for i, line in enumerate(f):
            obj = json.loads(line.strip())
            for h, r, t in obj['triple']:
                if '@@ROOT@@' in r:
                    continue
                rels.append(' '.join(r))
    d = dict(Counter(rels))
    return d

# d_train = get_rels('train')
# d_test = get_rels('test')
# d_test_oov = get_rels('test_oov')

# train_rels = set(d_train.keys())
# test_rels = set(d_test.keys())
# test_oov_rels = set(d_test_oov.keys())

# print(len(test_rels))
# # print(test_rels & train_rels)
# for key in test_oov_rels & train_rels:
#     print(key, d_test_oov[key])

# train_rel_dict = {key: idx+2 for idx, key in enumerate(train_rels)}
# train_rel_dict['no rel'] = 0
# train_rel_dict['other rel'] = 1
# print(json.dumps(train_rel_dict, ensure_ascii=False))

from data import EditingLSTMDataset
# train_set = EditingLSTMDataset(None, None, 'train')
# train_vocab = set(train_set.vocab)
# split = 'test_oov'

# oov_obj = []
# with open(os.path.join('Dataset/webedit', f'{split}.jsonl'), 'r') as f:
#     for i, line in enumerate(f):
#         obj = json.loads(line.strip())
#         cnt = 0
#         for h, r, t in obj['triple']:
#             if ' '.join(r) in train_rels:
#                 cnt += 1
#         if cnt == len(obj['triple']) - 1:
#             oov_obj.append(obj)
# print(len(oov_obj))

# with open(os.path.join('Dataset/webedit', f'test_oov2.jsonl'), 'w') as f:
#     for obj in oov_obj:
#         f.write(json.dumps(obj, ensure_ascii=False) + '\n')
import pdb

test_set = EditingLSTMDataset(None, None, 'test')


tp, pred, total = 0, 0, 0
for obj in test_set.metadata:
    triples = obj['triple']
    constraints = []
    all_entitys = set()
    for h, r, t in triples:
        all_entitys.add(h)
        all_entitys.add(t)
        if '@@ROOT@@' in r:
            continue
        # if h not in constraints:
        constraints.append(h)
        # if t not in constraints:
        constraints.append(t)
    cons_cnt = Counter(constraints)
    constraints = []
    for k in cons_cnt:
        if cons_cnt[k] == 1:
            constraints.append(k)
        else:
            constraints += [k] * int(cons_cnt[k] / 1.5)

    all_entitys -= {'@@ROOT@@'}
    tgt_words = [w for sent in obj['revised'] for w in sent]
    tgt_entitys = [w for w in tgt_words if w in all_entitys]
            
    pred_cnt = Counter(constraints)
    gold_cnt = Counter(tgt_entitys)

    for entity in gold_cnt:
        total += gold_cnt[entity]
    
    for entity in pred_cnt:
        pred += pred_cnt[entity]
        tp += min(pred_cnt[entity], gold_cnt[entity])
        # if pred_cnt[entity] > gold_cnt[entity]:
        #     pdb.set_trace()

print(tp, pred, total)
precision = tp / pred
recall = tp / total
f1 = 2 * precision * recall / (precision + recall)
print(precision, recall , f1)