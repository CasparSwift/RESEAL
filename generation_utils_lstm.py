from typing import Iterable, List, Optional, Tuple
import torch 
import torch.nn.functional as F
import sys
from torch import Tensor

# sys.path.append('../')
from DDBA import init_constraints, dba_topk
from Dep_DBA import init_dep_constraints, dep_dba_topk
from dep_scorer import lstm_prob_adjust
import pdb
from dependency_prediction.fact_editor import FactBaseModel, FactBaseGRUModel, FactBaseDualModel


class DependencyConstrainedTester(object):
    def __init__(self, args, model, test_loader, tokenizer):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer 
        # self.nlp = stanza.Pipeline(processors="tokenize,pos,lemma,depparse", lang="en", use_gpu=False)

    def test_one_epoch(self, device, output_path):
        # loss_meter, nll_loss_meter = AverageMeter(), AverageMeter()
        # self.model.eval()
        with open(output_path, 'a') as f:
            for i, inputs in enumerate(self.test_loader):
                inputs = {k: inputs[k].to(device) for k in inputs}
                # TODO 实现此函数
                output_ids = self.my_beam_search(device, inputs, constraints, self.args.max_tgt_length)
                output = self.get_output_string(output_ids)
                for sent in output:
                    f.write(sent + '\n')
                # outputs, loss, nll_loss = self.model(inputs)
                # loss_meter.update(loss.item())
                # nll_loss_meter.update(nll_loss.item())
                if i % self.args.logging_steps == 0:
                    print(f'epoch: [1] | batch: [{i}]', file=sys.stderr)
                if self.args.debug:
                    break

    def do_decode(self, ids):
        return self.tokenizer.decode(ids, 
            skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def get_output_string(self, output_ids):
        output = [self.do_decode(g) for g in output_ids]
        return output

    # batched input_ids parsing
    def dependency_parse(self, input_ids):
        pass


    def make_input_dict(self, inputs, input_ids, num_beams):
        batch_size, seq_len = inputs['input_ids'].shape
        return {
            'input_ids': inputs['input_ids'].repeat(1, num_beams).view(batch_size*num_beams, seq_len),
            'attention_mask': inputs['attention_mask'].repeat(1, num_beams).view(batch_size*num_beams, seq_len),
            'decoder_input_ids': input_ids
        }



def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """
    Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
    """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def set_scores_to_inf_for_banned_tokens(scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]

    banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    scores.masked_fill_(banned_mask, -float("inf"))


def postprocess_next_token_scores(scores, input_ids, no_repeat_ngram_size, bad_words_ids, 
    cur_len, min_length, max_length, eos_token_id, repetition_penalty, batch_size, num_beams,):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            scores,
            batch_size,
            num_beams,
            input_ids,
            repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    # if eos_token_id is not None and cur_len < min_length:
    #     scores[:, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    # if bad_words_ids is not None:
    #     # Exclude EOS token (already processed)
    #     bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
    #     # calculate a list of banned tokens according to bad words
    #     banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
    #     # Modify the scores in place by setting the banned tokens logits to `-inf`
    #     set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

    # ban some invalid tokens
    

    return scores


def create_copy_input(input_ids, real_length, batch_size, num_beams, pad_token_id):
    # copy src input_ids
    # [batch_size * num_beams, src_len]
    copy_input_ids = input_ids.repeat(1, num_beams).view(batch_size * num_beams, -1)
    copy_mask = (copy_input_ids != pad_token_id)
    copy_real_length = real_length.cpu().unsqueeze(-1).repeat(1, num_beams).view(batch_size * num_beams)
    return copy_input_ids, copy_real_length, copy_mask


def my_beam_search(model, device, inputs, max_length, num_beams=4, 
        bos_token_id=0, pad_token_id=1, eos_token_id=2, length_penalty=1.0, 
        early_stopping=True, no_repeat_ngram_size=2, bad_words_ids=None,
        min_length=None, repetition_penalty=None, 
        lexical_constraints=None, dep_constraints=None, **model_kwargs):
    length_penalty = length_penalty if length_penalty else 1.0
    repetition_penalty = repetition_penalty if repetition_penalty else 1.0
    batch_size = inputs['input_ids'].shape[0]

    # [src_len, batch_size * num_beams, 1024]
    if isinstance(model, FactBaseGRUModel):
        copy_src_input_ids, copy_real_length, copy_mask = create_copy_input(inputs['input_ids'], 
            inputs['real_length'], batch_size, num_beams, pad_token_id)
        encoder_outputs, hidden_states = model.encoder(copy_src_input_ids.transpose(0, 1), copy_real_length)
    elif isinstance(model, FactBaseDualModel):
        copy_src_input_ids1, copy_real_length1, copy_mask1 = create_copy_input(inputs['triple_input_ids'], 
            inputs['triple_real_length'], batch_size, num_beams, pad_token_id)
        encoder_outputs1, hidden_states1 = model.table_encoder(copy_src_input_ids1.transpose(0, 1), copy_real_length1)
        copy_src_input_ids2, copy_real_length2, copy_mask2 = create_copy_input(inputs['draft_input_ids'], 
            inputs['draft_real_length'], batch_size, num_beams, pad_token_id)
        encoder_outputs2, hidden_states2 = model.text_encoder(copy_src_input_ids2.transpose(0, 1), copy_real_length2)
        hidden_states = torch.cat((hidden_states1, hidden_states2), dim=1)
        copy_triple_mask = inputs['triple_mask'].repeat(1, num_beams).view(batch_size * num_beams, -1)
    else:
        raise NotImplementedError

    # 建立beam容器，每个样本一个
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    inactive = torch.zeros(batch_size * num_beams).to(device)

    # DBA or dep_DBA
    have_lexical_constraints = (lexical_constraints is not None)
    have_dep_constraints = (dep_constraints is not None)

    # initialize the constraints
    if have_lexical_constraints and have_dep_constraints:
        constraints = init_dep_constraints(lexical_constraints, dep_constraints, num_beams, bos_token_id, eos_token_id, **model_kwargs)
    elif have_lexical_constraints and not have_dep_constraints:
        constraints = init_constraints(lexical_constraints, num_beams, bos_token_id, eos_token_id)
    else:
        constraints = None

    # 每个beam容器的得分，共batch_size*num_beams个
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
    beam_scores = beam_scores.view(-1)

    # 每个样本是否完成生成，共batch_size个
    done = [False for _ in range(batch_size)]

    # 为了并行计算，一次生成batch_size*num_beams个序列
    # 第一步自动填入bos_token
    input_ids = torch.full(
        (batch_size*num_beams, 1),  
        bos_token_id,
        dtype=torch.long,
        device=device,
    )

    # 当前长度设为1
    cur_len = 1

    while cur_len < max_length:
        # 将编码器得到的上下文向量和当前结果输入解码器，即图中1
        # [batch_size * num_beams, vocab_size]
        # import pdb
        # pdb.set_trace()
        if isinstance(model, FactBaseGRUModel):
            output, hidden_states = model.decoder(
                input_ids.transpose(0, 1)[-1], 
                hidden_states, 
                encoder_outputs, 
                copy_mask
            )
        elif isinstance(model, FactBaseDualModel):
            output, hidden_states = model.decoder(
                input_ids.transpose(0, 1)[-1], 
                hidden_states, 
                copy_src_input_ids1.transpose(0, 1), 
                encoder_outputs1, 
                copy_mask1,
                copy_triple_mask,
                encoder_outputs2, 
                copy_mask2
            )
        # output = output.permute(1, 0, 2)
        vocab_size = output.shape[-1]
        
        # (batch_size * num_beams, vocab_size)
        # scores = next_token_logits = F.log_softmax(output, dim=-1)[:, -1, :]
        # print(output)
        # pdb.set_trace()
        # scores = next_token_logits = F.log_softmax(output, dim=-1)
        scores = next_token_logits = output.log()

        scores = postprocess_next_token_scores(
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        # if dep_constraints is not None:
        # assert model_kwargs.get('scorer') == 'lstm_prob'
        if model_kwargs.get('scorer') == 'lstm_prob':
            # print('call lstm prob adjust')
            scores = lstm_prob_adjust(scores, beam_scores, constraints, input_ids, num_beams, **model_kwargs)
 
        # (batch_size * num_beams, vocab_size)
        
        # 计算序列条件概率的，因为取了log，所以直接相加即可。得到图中2矩阵
        # (batch_size * num_beams, vocab_size)
        next_scores = scores + beam_scores[:, None].expand_as(scores)  

        curr_scores = next_scores.clone()

        # 为了提速，将结果重排成 [batch_size, num_beams * vocab_size]
        next_scores = next_scores.view(batch_size, num_beams * vocab_size)  

        # 取出分数最高的token（图中黑点）和其对应得分
        # sorted=True，保证返回序列是有序的
        # next_token = [batch_size, 2 * num_beams]
        next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True, sorted=True)

        beam_offset = torch.arange(
            0,
            batch_size * num_beams,
            step=num_beams,
            dtype=torch.long,
            device=next(model.parameters()).device)

        topk_ids = next_tokens.fmod(vocab_size)
        topk_beam_index = next_tokens // vocab_size
        next_tokens = (topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))

        if have_lexical_constraints and have_dep_constraints:
            # pdb.set_trace()
            res = dep_dba_topk(input_ids=input_ids,
                                batch_size=batch_size,
                                beam_size=num_beams,
                                inactive=inactive,
                                scores=curr_scores,
                                hypotheses=constraints,
                                best_ids=next_tokens,
                                best_word_ids=topk_ids,
                                seq_scores=next_scores,
                                debug=False,
                                skip=False,
                                **model_kwargs)
            next_tokens, topk_ids, next_scores, constraints, inactive = res
        
        elif have_lexical_constraints and not have_dep_constraints:
            dba_res = dba_topk(batch_size=batch_size,
                                beam_size=num_beams,
                                inactive=inactive,
                                scores=curr_scores,
                                hypotheses=constraints,
                                best_ids=next_tokens,
                                best_word_ids=topk_ids,
                                seq_scores=next_scores,
                                debug=False,
                                skip=False,
                                partial=model_kwargs['partial'] if 'partial' in model_kwargs else False,
                                top_k=model_kwargs['partial_top_k'] if 'partial_top_k' in model_kwargs else None,
                                top_p=model_kwargs['partial_top_p'] if 'partial_top_p' in model_kwargs else None,
                                min_score=model_kwargs[
                                    'partial_min_score'] if 'partial_min_score' in model_kwargs else -1e10,
                                attn_scores=None,
                                score_mode=model_kwargs[
                                    'partial_score_mode'] if 'partial_score_mode' in model_kwargs else 0,
                                )
            next_tokens, topk_ids, next_scores, constraints, inactive = dba_res
        
        assert next_scores.size() == next_tokens.size() == (batch_size, num_beams)
        
        # next batch beam content
        next_batch_beam = []
        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence
            if done[batch_idx]:
                assert (
                        len(generated_hyps[batch_idx]) >= num_beams - 2
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_id, token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], topk_ids[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                effective_beam_id = beam_id
                # add to generated hypotheses if end of sentence or last iteration
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # print("beam_token_rank:",beam_token_rank)

                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                    next_sent_beam.append((-1e8, token_id, effective_beam_id))
                    #     print(sorted_hyps)
                else:
                    # add next predicted token if it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # Check if were done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len=cur_len
            )

            # update next beam content  #"Beam should always be full"
            assert len(next_sent_beam) == num_beams
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # 如果全部样本都已经生成结束便可以直接退出了
        if all(done):
            break
        
        # 把三元组列表再还原成三个独立列表
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # 准备下一时刻的解码器输入
        # 取出实际被扩展的beam
        input_ids = input_ids[beam_idx, :]
        # 在这些beam后面接上新生成的token
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)

        # model_kwargs['tokenizer']

        # 更新当前长度
        cur_len = cur_len + 1
        # end of length while

    # 将未结束的生成结果结束，并置入容器中
    for batch_idx in range(batch_size):
        # 已经结束的样本不需处理
        if done[batch_idx]:
            continue

        # 把结果加入到generated_hyps容器
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # select the best hypotheses，最终输出
    # 每个样本返回几个句子
    output_num_return_sequences_per_batch = 1
    output_batch_size = output_num_return_sequences_per_batch * batch_size

    # 记录每个返回句子的长度，用于后面pad
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # 对每个样本取出最好的output_num_return_sequences_per_batch个句子
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # 如果长短不一则pad句子，使得最后返回结果的长度一样
    if sent_lengths.min().item() != sent_lengths.max().item():
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        # 先把输出矩阵填满PAD token
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # 填入真正的内容
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            # 填上eos token
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # 所有生成序列都还没结束，直接堆叠即可
        decoded = torch.stack(best).type(torch.long).to(device)

    # 返回的结果包含BOS token
    # print(decoded)
    return decoded


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
        self.length_penalty = length_penalty
        self.early_stopping=early_stopping

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            # 可更新的情况：数量未饱和或超过最差得分
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # 数量饱和需要删掉一个最差的
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        相关样本是否已经完成生成。
        best_sum_logprobs是新的候选序列中的最高得分。
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # 是否最高分比当前保存的最低分还差
            ret = self.worst_score >= cur_score
            return ret