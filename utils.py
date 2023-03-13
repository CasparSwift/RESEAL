import pdb
import torch
import torch.nn.functional as F


def label_smoothed_nll_loss(logits, decoder_input_ids, ignore_index=1, reduce=True, epsilon=0.1):
    # logits: [bsz * seq_length-1 * vocab_size]
    # log probs: [bsz * seq_length-1 * vocab_size]
    lprobs = F.log_softmax(logits[:, :-1, :], dim=-1)

    # [bsz * seq_length-1]
    target = decoder_input_ids[:, 1:]

    # [bsz * seq_length-1 * 1]
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    # [bsz * seq_length-1 * 1]
    nll_loss = -lprobs.gather(dim=-1, index=target)

    # [bsz * seq_length-1 * 1]
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        det = torch.sum(target.ne(ignore_index))
        nll_loss = nll_loss.sum() / det
        smooth_loss = smooth_loss.sum() / det
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def get_constraints(inputs, cons_label, tokenizer, h_token_id, r_token_id, t_token_id):
    batch_raw_words = []
    batch_lexical_constraints = []
    batch_dep_constraints = []

    for batch_idx in range(cons_label.shape[0]):
        pred_constraint_label = cons_label[batch_idx].tolist()
        input_ids = inputs['input_ids'][batch_idx].tolist()
        new_pred_constraint_label = []
        entity_label = 0
        for token_idx, label in enumerate(pred_constraint_label):
            if input_ids[token_idx] in [h_token_id, t_token_id]:
                entity_label = label
                new_pred_constraint_label.append(0)
            elif input_ids[token_idx] == r_token_id:
                entity_label = -1 # in a relation 
                new_pred_constraint_label.append(0)
            else:
                if entity_label == -1:
                    new_pred_constraint_label.append(label)
                else:
                    new_pred_constraint_label.append(entity_label)
        # print(pred_constraint_label)
        # print(new_pred_constraint_label)
        # print(input_ids)

        lexical_constraints_and_idx = []
        lexical_cons = []
        for token_idx, label in enumerate(new_pred_constraint_label):
            if label and input_ids[token_idx] != tokenizer.eos_token_id:
                lexical_cons.append(input_ids[token_idx])
            else:
                if lexical_cons:
                    lexical_constraints_and_idx.append((token_idx-1, lexical_cons))
                    lexical_cons = []
        
        position_embed = []
        position_id, triple_cnt = (0, 0), 0
        for token_id in input_ids:
            if token_id == h_token_id:
                triple_cnt += 1
                position_id = (triple_cnt, 1)
            elif token_id == r_token_id:
                position_id = (triple_cnt, 2)
            elif token_id == t_token_id:
                position_id = (triple_cnt, 3)
            elif token_id == 0:
                position_id = (triple_cnt, 0)
            position_embed.append(position_id)
        
        # generate dep_constraints
        dep_constraints = []
        for triple_idx in range(triple_cnt):
            head_lex_cons = [c for token_idx, c in lexical_constraints_and_idx 
                if position_embed[token_idx] == (triple_idx+1, 1)]
            tail_lex_cons = [c for token_idx, c in lexical_constraints_and_idx 
                if position_embed[token_idx] == (triple_idx+1, 3)]
            if head_lex_cons and tail_lex_cons:
                for head in head_lex_cons:
                    for tail in tail_lex_cons:
                        head_words = tokenizer.decode(head, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False).split()
                        tail_words = tokenizer.decode(tail, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False).split()
                        if head_words and tail_words and head_words[0] != tail_words[0]:
                            dep_constraints.append((head_words[0], 2, tail_words[0]))
        raw_words = []
        for token_idx, c in lexical_constraints_and_idx:
            if position_embed[token_idx][1] == 2:
                raw_word = tokenizer.decode(c, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False)
                # for w in raw_word.split():
                #     if w not in raw_words:
                #         raw_words.append(w)
                if raw_word:
                    raw_words.append(raw_word)
        for head, _, tail in dep_constraints:
            raw_words += [head, tail]
        raw_words = list(set(raw_words))
        batch_lexical_constraints.append([tokenizer.encode(w, add_special_tokens=False)
            for w in raw_words])
        batch_raw_words.append(raw_words)
        batch_dep_constraints.append(dep_constraints)

        # print(raw_words)
        # print(dep_constraints)
        # pdb.set_trace()
    return batch_raw_words, batch_lexical_constraints, batch_dep_constraints


def get_constraints_seq_labeling(inputs, cons_label, tokenizer, cur_src_deps):
    batch_raw_words = []
    batch_lexical_constraints = []
    batch_dep_constraints = []

    for batch_idx in range(cons_label.shape[0]):
        pred_constraint_label = cons_label[batch_idx].tolist()
        input_ids = inputs['input_ids'][batch_idx].tolist()

        lexical_constraints = []
        lexical_cons = []
        for token_idx, label in enumerate(pred_constraint_label):
            if input_ids[token_idx] == tokenizer.bos_token_id:
                continue
            if input_ids[token_idx] == tokenizer.eos_token_id:
                break
            if label:
                lexical_cons.append(input_ids[token_idx])
            else:
                if lexical_cons:
                    lexical_constraints.append(lexical_cons)
                    lexical_cons = []
        if lexical_cons:
            lexical_constraints.append(lexical_cons)
        
        raw_words = []
        for c in lexical_constraints:
            raw_word = tokenizer.decode(c, skip_special_tokens=True, 
                clean_up_tokenization_spaces=False)
            if raw_word:
                raw_words.append(raw_word)
        raw_words = list(set(raw_words))

        # generate dep_constraints
        dep_constraints = []
        all_tokenized_words = []
        for word in raw_words:
            for w in word.split():
                if w:
                    all_tokenized_words.append(w)
        # print(lexical_constraints)
        # print(raw_words)
        # print(all_tokenized_words)
        for w1, r, w2 in cur_src_deps[batch_idx]:
            if w1 in all_tokenized_words and w2 in all_tokenized_words:
                dep_constraints.append([w1, r, w2])
        # print(dep_constraints)

        batch_lexical_constraints.append(lexical_constraints)
        batch_raw_words.append(raw_words)
        batch_dep_constraints.append(dep_constraints)

    return batch_raw_words, batch_lexical_constraints, batch_dep_constraints


def match_lst(lst1, lst2):
    length1 = len(lst1)
    length2 = len(lst2)
    match_idx = []
    for i in range(length1):
        if i + length2 > length1:
            break
        if lst1[i: i+length2] == lst2:
            match_idx.append(i)
    return match_idx