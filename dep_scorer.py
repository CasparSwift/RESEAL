import math
import pdb
import torch
import copy
from functools import cmp_to_key
from tokenizers.tokenizer import SimpleTokenizer
import time


def alpha_func(lamb, rho, transition_prob, type_prob, func_type='1'):
    # transition_prob = transition_prob if transition_prob < 0.5 else 1.0
    # type_prob = type_prob if type_prob < 0.5 else 1.0
    # return 2 / (1 + math.exp(lamb * (1 - transition_prob * type_prob)))
    if func_type == '1':
        transition_prob = transition_prob if transition_prob < rho else 1.0
        type_prob = type_prob if type_prob < rho else 1.0
        alpha1 = math.exp(-lamb * (1 - transition_prob))
        alpha2 = math.exp(-lamb * (1 - type_prob))
        return alpha1 * alpha2
    elif func_type == '2':
        transition_prob = transition_prob if transition_prob > rho else 0.0
        return math.exp(lamb * transition_prob * type_prob)
    elif func_type == '3':
        return 1
    else:
        exit('error!')


def prob_scorer(parser, sorted_candidates, input_ids, beam_size, prune_size, lamb):
    # strings = []
    for i, cand in enumerate(sorted_candidates):
        # candidate_input_id = input_ids[cand.row].tolist() + [cand.col]
        # candidate_string = tokenizer.decode(candidate_input_id, skip_special_tokens=True, 
        #             clean_up_tokenization_spaces=False)
        # strings.append(candidate_string)

        info = cand.hypothesis.col_to_info.get(cand.col, {})
        dep_correct_mets = info.get('dep_correct_mets', cand.hypothesis.dep_correct_met)
        cand.hypothesis.dep_correct_met = copy.deepcopy(dep_correct_mets)

        # dep_mets_dict = info.get('dep_mets_dict', cand.hypothesis.dep_constraints_dict)
        # cand.hypothesis.dep_constraints_dict = copy.deepcopy(dep_mets_dict)
        # print(i, cand.hypothesis.dep_num_correct_met())
        # pdb.set_trace()

    # for idx, (cand, string) in enumerate(zip(sorted_candidates, strings)):
    #     print(idx, string, cand.col, cand.score, cand.hypothesis.dep_num_met(), cand.hypothesis.dep_num_correct_met())

    sorted_candidates = sorted(sorted_candidates, 
        key=lambda c: c.score, 
        reverse=True
    )
    return sorted_candidates


def lstm_prob_scorer(parser, sorted_candidates, input_ids, beam_size, prune_size, lamb):
    strings = []
    for i, cand in enumerate(sorted_candidates):
        candidate_input_id = input_ids[cand.row].tolist() + [cand.col]
        candidate_string = tokenizer.decode(candidate_input_id, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False, return_list=True)
        strings.append(candidate_string)

        info = cand.hypothesis.col_to_info.get(cand.col, {})
        dep_correct_mets = info.get('dep_correct_mets', cand.hypothesis.dep_correct_met)
        cand.hypothesis.dep_correct_met = copy.deepcopy(dep_correct_mets)

        # if candidate_string[-1] == candidate_string[-2]:
            # weight = 0.0
        if len(candidate_string) >= 2:
            raw_cons = cand.hypothesis.raw_lexical_constraints
            if candidate_string[-2] in raw_cons and candidate_string[-1] in raw_cons:
                cand.score = -1e10
            # for prev_word, next_word in zip(candidate_string[:-1], candidate_string[1:]):
            #     if prev_word == next_word or (prev_word in raw_cons and next_word in raw_cons):
            #         cand.score = -1e10
            #         break
        

    sorted_candidates = sorted(sorted_candidates, 
        key=lambda c: c.score, 
        reverse=True
    )
    # for i, cand in enumerate(sorted_candidates):
    #     candidate_input_id = input_ids[cand.row].tolist() + [cand.col]
    #     candidate_string = tokenizer.decode(candidate_input_id, skip_special_tokens=True, 
    #                 clean_up_tokenization_spaces=False, return_list=True)
    #     print(i, candidate_string, cand.score, cand.row)
    # pdb.set_trace()
    return sorted_candidates



# adjust the probability
def prob_adjust(scores, beam_scores, hypotheses, input_ids, beam_size, **model_kwargs):
    parser = model_kwargs['parser']
    lamb = model_kwargs['lamb']
    rho = model_kwargs['rho']
    func_type = model_kwargs['alpha_func']
    normalize = model_kwargs['normalize']
    completion = model_kwargs['completion']
    tokenizer = model_kwargs['tokenizer']

    # (batch_size * num_beams, vocab_size)
    probs = torch.exp(scores)
    # beam_probs = torch.exp(beam_scores)

    checked_list = []

    for i in range(len(hypotheses)):
        if hypotheses[i] is None:
            continue
        allowed_tokens = hypotheses[i].allowed()
        for allowed_token in allowed_tokens:
            expand_token_list = [allowed_token]
            if completion:
                num_constraints = len(hypotheses[i].constraints)
                is_sequence = [False] + hypotheses[i].is_sequence[:-1]
                for j in range(num_constraints):
                    if hypotheses[i].constraints[j] == allowed_token and is_sequence[j] == False:
                        idx = j + 1
                        while idx < num_constraints and is_sequence[idx] == True:
                            expand_token_list.append(hypotheses[i].constraints[idx])
                            idx += 1
                        break
            # print(input_ids[i])
            # pdb.set_trace()
            candidate_input_id = input_ids[i].tolist() + expand_token_list
            if isinstance(tokenizer, SimpleTokenizer):
                candidate_string = tokenizer.decode(candidate_input_id, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False, return_list=True)
                candidate_tokens = [s for s in candidate_string]
            else:
                candidate_string = tokenizer.decode(candidate_input_id, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False)
                candidate_tokens = [s.lower() for s in candidate_string.split()]
            if len(candidate_tokens) == 0:
                continue
            candidate_token = candidate_tokens[-1]
            previous_tokens = set(candidate_tokens[:-1])

            rel_constraints = hypotheses[i].related_dep_constraints.get(candidate_token, [])
            checked_word_pairs = []
            checked_constraints = []
            for w1, r, w2 in rel_constraints:
                is_met = hypotheses[i].dep_constraints_dict[(w1, r, w2)]
                if is_met:
                    continue
                if (w1 in previous_tokens and w2 == candidate_token) or \
                    (w2 in previous_tokens and w1 == candidate_token):
                    checked_word_pairs.append((w1, w2))
                    checked_constraints.append((w1, r, w2))
            if checked_constraints:
                checked_list.append({
                    'index': i, 
                    'hyp': hypotheses[i], 
                    'string': candidate_string,
                    'checked_word_pairs': checked_word_pairs,
                    'checked_constraints': checked_constraints,
                    'allowed_token': allowed_token
                })

    # print(checked_list)

    if checked_list:
        # t1 = time.time()
        # print(checked_list)
        all_pred_deps, all_dep_probs = parser.get_dep_probs(checked_list)
        # print(time.time() - t1)
        # for c, dep_probs in zip(checked_list, all_dep_probs):
            # print(c['string'], dep_probs)
        # pdb.set_trace()

        prob_adjust = {}

        for idx, (pred_deps, dep_prob_dict) in enumerate(zip(all_pred_deps, all_dep_probs)):
            index = checked_list[idx]['index']
            allowed_token = checked_list[idx]['allowed_token']
            dep_correct_mets = copy.deepcopy(checked_list[idx]['hyp'].dep_correct_met)
            weight = 1.0
            for w1, r, w2 in checked_list[idx]['checked_constraints']:
                transition_prob, type_prob = dep_prob_dict[(w1, r, w2)]
                # checked_list[idx]['cand'].hypothesis.dep_constraints_dict[(w1, r, w2)] = True
                weight *= alpha_func(lamb, rho, transition_prob, type_prob, func_type)
                # weight = max(weight, alpha_func(lamb, transition_prob, type_prob))
                # print(weight)
                if transition_prob > rho and type_prob > rho:
                    dep_correct_mets.add((w1, r, w2))

            # dep_mets_dict = copy.deepcopy(checked_list[idx]['hyp'].dep_constraints_dict)

            # for w1, r, w2 in pred_deps:
            #     # not met but type is wrong --> set as met
            #     if (w1, w2) in checked_list[idx]['checked_word_pairs']:
            #         for tup in dep_mets_dict:
            #             if (w1, w2) == (tup[0], tup[2]):
            #                 print(dep_mets_dict, checked_list[idx]['string'])
            #                 dep_mets_dict[tup] = True

            if index not in prob_adjust:
                prob_adjust[index] = []
            prob_adjust[index].append({
                'allowed_token': allowed_token,
                # 'transition_prob': transition_prob,
                # 'type_prob': type_prob,
                'weight': weight,
                'dep_correct_mets': dep_correct_mets,
                # 'dep_mets_dict': dep_mets_dict
            })
            
        # pdb.set_trace()
        # do adjustment
        for index, lst in prob_adjust.items():
            if lst == []:
                continue
            lst = sorted(lst, key=lambda x: x['weight'], reverse=True)
            new_col_to_info = {}
            for i in range(len(lst)):
                allowed_token, weight = lst[i]['allowed_token'], lst[i]['weight']
                # print(probs[index][allowed_token], weight)
                probs[index][allowed_token] = probs[index][allowed_token] * weight
                # for w1, r, w2 in lst[i]['dep_correct_mets']:
                #     hypotheses[index].dep_correct_met.add((w1, r, w2))
                # print(hypotheses[index].dep_num_correct_met())
                new_col_to_info[allowed_token] = {
                    'dep_correct_mets': lst[i]['dep_correct_mets'],
                    # 'dep_mets_dict': lst[i]['dep_mets_dict']
                }
            hypotheses[index].col_to_info = copy.deepcopy(new_col_to_info)
        if normalize:
            probs = (probs.transpose(0, 1) / probs.sum(dim=-1)).transpose(0, 1)

    return torch.log(probs)



def lstm_prob_adjust(scores, beam_scores, hypotheses, input_ids, beam_size, **model_kwargs):
    parser = model_kwargs['parser']
    lamb = model_kwargs['lamb']
    rho = model_kwargs['rho']
    func_type = model_kwargs['alpha_func']
    normalize = model_kwargs['normalize']
    completion = model_kwargs['completion']

    # (batch_size * num_beams, vocab_size)
    probs = torch.exp(scores)
    # beam_probs = torch.exp(beam_scores)

    checked_list = []

    for i in range(len(hypotheses)):
        if hypotheses[i] is None:
            continue
        allowed_tokens = hypotheses[i].allowed()
        for allowed_token in allowed_tokens:
            expand_token_list = [allowed_token]
            if completion:
                num_constraints = len(hypotheses[i].constraints)
                is_sequence = [False] + hypotheses[i].is_sequence[:-1]
                for j in range(num_constraints):
                    if hypotheses[i].constraints[j] == allowed_token and is_sequence[j] == False:
                        idx = j + 1
                        while idx < num_constraints and is_sequence[idx] == True:
                            expand_token_list.append(hypotheses[i].constraints[idx])
                            idx += 1
                        break
            candidate_input_id = input_ids[i].tolist() + expand_token_list
            if isinstance(tokenizer, SimpleTokenizer):
                candidate_string = tokenizer.decode(candidate_input_id, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False, return_list=True)
                # stop_idx = -1
                # for idx, word in enumerate(candidate_string):
                #     if word == '.':
                #         stop_idx = idx
                # candidate_string = candidate_string[stop_idx + 1: ]
                candidate_tokens = [s for s in candidate_string]
            else:
                candidate_string = tokenizer.decode(candidate_input_id, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False)
                candidate_tokens = [s.lower() for s in candidate_string.split()]
            if len(candidate_tokens) == 0:
                continue
            candidate_token = candidate_tokens[-1]
            previous_tokens = set(candidate_tokens[:-1])

            rel_constraints = hypotheses[i].related_dep_constraints.get(candidate_token, [])
            checked_word_pairs = []
            checked_constraints = []
            for w1, r, w2 in rel_constraints:
                is_met = hypotheses[i].dep_constraints_dict[(w1, r, w2)]
                if is_met:
                    continue
                if (w1 in previous_tokens and w2 == candidate_token) or \
                    (w2 in previous_tokens and w1 == candidate_token):
                    checked_word_pairs.append((w1, w2))
                    checked_constraints.append((w1, r, w2))
            if checked_constraints:
                checked_list.append({
                    'index': i, 
                    'hyp': hypotheses[i], 
                    'string': candidate_string,
                    'checked_word_pairs': checked_word_pairs,
                    'checked_constraints': checked_constraints,
                    'allowed_token': allowed_token
                })


    if checked_list:
        # import time
        # t1 = time.time()
        # print(checked_list)
        all_dep_probs = parser.get_dep_probs(checked_list)
        # print(time.time() - t1)
        # for c, dep_probs in zip(checked_list, all_dep_probs):
        #     print(c['string'], dep_probs)
        # pdb.set_trace()

        prob_adjust = {}

        for idx, dep_prob_dict in enumerate(all_dep_probs):
            index = checked_list[idx]['index']
            allowed_token = checked_list[idx]['allowed_token']
            dep_correct_mets = copy.deepcopy(checked_list[idx]['hyp'].dep_correct_met)
            weight = 1.0
            for w1, r, w2 in checked_list[idx]['checked_constraints']:
                transition_prob, type_prob = dep_prob_dict[(w1, r, w2)]
                # checked_list[idx]['cand'].hypothesis.dep_constraints_dict[(w1, r, w2)] = True
                weight *= alpha_func(lamb, rho, transition_prob, type_prob, func_type)
                # weight = max(weight, alpha_func(lamb, transition_prob, type_prob))
                # print(weight)
                if transition_prob > rho and type_prob > rho:
                    dep_correct_mets.add((w1, r, w2))

            # dep_mets_dict = copy.deepcopy(checked_list[idx]['hyp'].dep_constraints_dict)

            # for w1, r, w2 in pred_deps:
            #     # not met but type is wrong --> set as met
            #     if (w1, w2) in checked_list[idx]['checked_word_pairs']:
            #         for tup in dep_mets_dict:
            #             if (w1, w2) == (tup[0], tup[2]):
            #                 print(dep_mets_dict, checked_list[idx]['string'])
            #                 dep_mets_dict[tup] = True

            # candidate_string = checked_list[idx]['string']
            # stop_idx = -1
            # for word_idx, word in enumerate(candidate_string):
            #     if word == '.':
            #         stop_idx = word_idx
            # candidate_string = candidate_string[stop_idx + 1: ]
            # if candidate_string and candidate_string[-1] in set(candidate_string[:-1]):
            #     pdb.set_trace()
            #     weight = 1e-30

            if index not in prob_adjust:
                prob_adjust[index] = []
            prob_adjust[index].append({
                'allowed_token': allowed_token,
                # 'transition_prob': transition_prob,
                # 'type_prob': type_prob,
                'weight': weight,
                'dep_correct_mets': dep_correct_mets,
                # 'dep_mets_dict': dep_mets_dict
            })
            
        # pdb.set_trace()
        
        # do adjustment
        for hyp_index, lst in prob_adjust.items():
            if lst == []:
                continue
            lst = sorted(lst, key=lambda x: x['weight'], reverse=True)
            new_col_to_info = {}
            for i in range(len(lst)):
                allowed_token, weight = lst[i]['allowed_token'], lst[i]['weight']
                # avoid repetition
                # candidate_string = checked_list[hyp_index]['string']
                # if len(candidate_string) >= 2:
                #     if candidate_string[-1] == candidate_string[-2]:
                #         weight = 1e-30
                #     raw_cons = hypotheses[hyp_index].raw_lexical_constraints
                #     if candidate_string[-1] in raw_cons and candidate_string[-2] in raw_cons:
                #         weight = 1e-30
                probs[hyp_index][allowed_token] = probs[hyp_index][allowed_token] * weight
                # print(checked_list[hyp_index], weight)
                # pdb.set_trace()
                # for w1, r, w2 in lst[i]['dep_correct_mets']:
                #     hypotheses[index].dep_correct_met.add((w1, r, w2))
                # print(hypotheses[index].dep_num_correct_met())
                new_col_to_info[allowed_token] = {
                    'dep_correct_mets': lst[i]['dep_correct_mets'],
                    # 'dep_mets_dict': lst[i]['dep_mets_dict']
                }
            hypotheses[hyp_index].col_to_info = copy.deepcopy(new_col_to_info)
        if normalize:
            probs = (probs.transpose(0, 1) / probs.sum(dim=-1)).transpose(0, 1)

    return torch.log(probs)


# adjust the probability
def webnlg_prob_adjust(scores, beam_scores, hypotheses, input_ids, beam_size, **model_kwargs):
    parser = model_kwargs['parser']
    lamb = model_kwargs['lamb']
    rho = model_kwargs['rho']
    func_type = model_kwargs['alpha_func']
    normalize = model_kwargs['normalize']
    completion = model_kwargs['completion']
    tokenizer = model_kwargs['tokenizer']

    # (batch_size * num_beams, vocab_size)
    probs = torch.exp(scores)
    # beam_probs = torch.exp(beam_scores)

    checked_list = []

    for i in range(len(hypotheses)):
        if hypotheses[i] is None:
            continue
        allowed_tokens = hypotheses[i].allowed()
        for allowed_token in allowed_tokens:
            expand_token_list = [allowed_token]
            if completion:
                num_constraints = len(hypotheses[i].constraints)
                is_sequence = [False] + hypotheses[i].is_sequence[:-1]
                for j in range(num_constraints):
                    if hypotheses[i].constraints[j] == allowed_token and is_sequence[j] == False:
                        idx = j + 1
                        while idx < num_constraints and is_sequence[idx] == True:
                            expand_token_list.append(hypotheses[i].constraints[idx])
                            idx += 1
                        break
            candidate_input_id = input_ids[i].tolist() + expand_token_list
            if isinstance(tokenizer, SimpleTokenizer):
                candidate_string = tokenizer.decode(candidate_input_id, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False, return_list=True)
                candidate_tokens = [s for s in candidate_string]
            else:
                candidate_string = tokenizer.decode(candidate_input_id, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False)
                candidate_tokens = [s.lower() for s in candidate_string.split()]
            if len(candidate_tokens) == 0:
                continue
            candidate_token = candidate_tokens[-1]
            previous_tokens = set(candidate_tokens[:-1])

            rel_constraints = hypotheses[i].related_dep_constraints.get(candidate_token, [])
            checked_word_pairs = []
            checked_constraints = []

            for w1, r, w2 in rel_constraints:
                is_met = hypotheses[i].dep_constraints_dict[(w1, r, w2)]
                if is_met:
                    continue
                if (w1 in previous_tokens and w2 == candidate_token) or \
                    (w2 in previous_tokens and w1 == candidate_token):
                    checked_word_pairs.append((w1, w2))
                    checked_constraints.append((w1, r, w2))
            if checked_constraints:
                checked_list.append({
                    'index': i, 
                    'hyp': hypotheses[i], 
                    'string': candidate_string,
                    'input_ids': candidate_input_id,
                    'checked_word_pairs': checked_word_pairs,
                    'checked_constraints': checked_constraints,
                    'allowed_token': allowed_token
                })

    if checked_list:
        # import time
        # t1 = time.time()
        # print(checked_list)
        all_dep_probs = parser.get_dep_probs(checked_list)
        # print(time.time() - t1)
        # for c, dep_probs in zip(checked_list, all_dep_probs):
            # print(c['string'], dep_probs)
        # pdb.set_trace()

        prob_adjust = {}

        for idx, dep_prob_dict in enumerate(all_dep_probs):
            index = checked_list[idx]['index']
            allowed_token = checked_list[idx]['allowed_token']
            dep_correct_mets = copy.deepcopy(checked_list[idx]['hyp'].dep_correct_met)
            weight = 1.0
            for w1, r, w2 in checked_list[idx]['checked_constraints']:
                transition_prob, type_prob = dep_prob_dict[(w1, r, w2)]
                # checked_list[idx]['cand'].hypothesis.dep_constraints_dict[(w1, r, w2)] = True
                weight *= alpha_func(lamb, rho, transition_prob, type_prob, func_type)
                # weight = max(weight, alpha_func(lamb, transition_prob, type_prob))
                # print(weight)
                if transition_prob > rho and type_prob > rho:
                    dep_correct_mets.add((w1, r, w2))

            if index not in prob_adjust:
                prob_adjust[index] = []
            prob_adjust[index].append({
                'allowed_token': allowed_token,
                # 'transition_prob': transition_prob,
                # 'type_prob': type_prob,
                'weight': weight,
                'dep_correct_mets': dep_correct_mets,
                # 'dep_mets_dict': dep_mets_dict
            })
            
        # pdb.set_trace()
        # do adjustment
        for index, lst in prob_adjust.items():
            if lst == []:
                continue
            lst = sorted(lst, key=lambda x: x['weight'], reverse=True)
            new_col_to_info = {}
            for i in range(len(lst)):
                allowed_token, weight = lst[i]['allowed_token'], lst[i]['weight']
                # print(probs[index][allowed_token], weight)
                probs[index][allowed_token] = probs[index][allowed_token] * weight
                # for w1, r, w2 in lst[i]['dep_correct_mets']:
                #     hypotheses[index].dep_correct_met.add((w1, r, w2))
                # print(hypotheses[index].dep_num_correct_met())
                new_col_to_info[allowed_token] = {
                    'dep_correct_mets': lst[i]['dep_correct_mets'],
                    # 'dep_mets_dict': lst[i]['dep_mets_dict']
                }
            hypotheses[index].col_to_info = copy.deepcopy(new_col_to_info)
        if normalize:
            probs = (probs.transpose(0, 1) / probs.sum(dim=-1)).transpose(0, 1)

    return torch.log(probs)




scorer_factory = {
    'prob': prob_scorer,
    'lstm_prob': lstm_prob_scorer,
    'webnlg_prob': prob_scorer
}