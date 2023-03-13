
"""
Implementation of all kinds of parser or relation checkers
"""

import os
import sys
import gc
import json

current_path = os.path.dirname(os.path.realpath(__file__))
# root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(current_path, 'L2RParser'))
sys.path.append(os.path.join(current_path, 'tokenizers'))

import time
import numpy as np
import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from L2RParser.neuronlp2.nn.utils import total_grad_norm
from L2RParser.neuronlp2.io import get_logger, conllx_data, conllx_stacked_data, iterate_data
from L2RParser.neuronlp2.models import DeepBiAffine, NeuroMST, L2RPtrNet
from L2RParser.neuronlp2.optim import ExponentialScheduler
from L2RParser.neuronlp2 import utils
from L2RParser.neuronlp2.io import CoNLLXWriter
from L2RParser.neuronlp2.tasks import parser
from L2RParser.neuronlp2.nn.utils import freeze_embedding

from L2RParser.neuronlp2.io.instance import DependencyInstance, NERInstance
from L2RParser.neuronlp2.io.instance import Sentence
from L2RParser.neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID
from L2RParser.neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from L2RParser.neuronlp2.io.common import ROOT, END, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE

import stanza
import spacy
import pdb

from fact_editor import FactRelModel
from tokenizer import SimpleTokenizer
from data import lstm_padding
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
from models import WebNlgRelModel
from transformers import AutoTokenizer
from utils import match_lst


def get_args():
    class args:
        word_path = 'L2RParser/emb/sskip.eng.100.gz'
        punctuation = ['.', '``', "''", ':', ',']
        model_path = 'L2RParser/experiments/models/l2rmymodel'
        remove_cycles = True
        beam = 1
    return args


logger = get_logger("Parsing")


class CoNLLXLineReader(object):
    def __init__(self, data_lines, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.data_lines = data_lines
        self.data_len = len(data_lines)
        self.line_idx = 0
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def close(self):
        pass

    def readline(self):
        if self.line_idx >= self.data_len:
            return None
        l = self.data_lines[self.line_idx]
        self.line_idx += 1
        return l

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        line = self.readline()
        # skip multiple blank lines.
        # while len(line) > 0 and len(line.strip()) == 0:
        #     line = self.readline()
        if line is None:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split('\t'))
            line = self.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            heads.append(0)

        for tokens in lines:
            chars = []
            char_ids = []
            if len(tokens) == 1:
                continue
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[4]

            head = int(tokens[6])
            type = tokens[7]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, heads, types, type_ids)


class LeftToRightPointerParser(object):
    def __init__(self):  
        # POS tagger
        self.pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
        
        self.args = get_args()
        self.device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
        print('parser device:', self.device)
        remove_cycles= self.args.remove_cycles

        model_path = self.args.model_path
        model_name = os.path.join(model_path, 'model.pt')
        punctuation = self.args.punctuation

        logger.info("Creating Alphabets")
        alphabet_path = os.path.join(model_path, 'alphabets')
        assert os.path.exists(alphabet_path)
        self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet \
            = conllx_data.create_alphabets(alphabet_path, None)

        num_words = self.word_alphabet.size()
        num_chars = self.char_alphabet.size()
        num_pos = self.pos_alphabet.size()
        num_types = self.type_alphabet.size()

        logger.info("Word Alphabet Size: %d" % num_words)
        logger.info("Character Alphabet Size: %d" % num_chars)
        logger.info("POS Alphabet Size: %d" % num_pos)
        logger.info("Type Alphabet Size: %d" % num_types)

        result_path = os.path.join(model_path, 'tmp')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        self.punct_set = None
        if punctuation is not None:
            self.punct_set = set(punctuation)
            logger.info("punctuations(%d): %s" % (len(self.punct_set), ' '.join(self.punct_set)))

        logger.info("loading network...")
        hyps = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
        model_type = hyps['model']
        assert model_type in ['DeepBiAffine', 'NeuroMST', 'L2RPtr']
        word_dim = hyps['word_dim']
        char_dim = hyps['char_dim']
        use_pos = hyps['pos']
        pos_dim = hyps['pos_dim']
        mode = hyps['rnn_mode']
        hidden_size = hyps['hidden_size']
        arc_space = hyps['arc_space']
        type_space = hyps['type_space']
        p_in = hyps['p_in']
        p_out = hyps['p_out']
        p_rnn = hyps['p_rnn']
        activation = hyps['activation']
        self.prior_order = None

        self.alg = 'transition' if model_type == 'L2RPtr' else 'graph'
        if model_type == 'DeepBiAffine':
            num_layers = hyps['num_layers']
            self.network = DeepBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                                        mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                        p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
        elif model_type == 'NeuroMST':
            num_layers = hyps['num_layers']
            self.network = NeuroMST(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                                    mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                    p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
        elif model_type == 'L2RPtr':
            encoder_layers = hyps['encoder_layers']
            decoder_layers = hyps['decoder_layers']
            num_layers = (encoder_layers, decoder_layers)
            self.prior_order = hyps['prior_order']
            grandPar = hyps['grandPar']
            sibling = hyps['sibling']
            self.network = L2RPtrNet(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                                     mode, hidden_size, encoder_layers, decoder_layers, num_types, arc_space, type_space,
                                     prior_order=self.prior_order, activation=activation, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                                     pos=use_pos, grandPar=grandPar, sibling=sibling, remove_cycles=remove_cycles)
        else:
            raise RuntimeError('Unknown model type: %s' % model_type)

        self.network = self.network.to(self.device)
        self.network.load_state_dict(torch.load(model_name, map_location=self.device))
        self.network.eval()
        model = "{}-{}".format(model_type, mode)
        logger.info("Network: %s, num_layer=%s, hidden=%d, act=%s" % (model, num_layers, hidden_size, activation))


    def _read_data(self, data_lines, max_size=None, normalize_digits=True):
        data = []
        max_length = 0
        max_char_length = 0
        # print('Reading data from %s' % source_path)
        counter = 0
        reader = CoNLLXLineReader(data_lines, self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet)
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
        while inst is not None and (not max_size or counter < max_size):
            counter += 1
            if counter % 10000 == 0:
                print("reading data: %d" % counter)

            sent = inst.sentence
            stacked_heads, children, siblings, stacked_types, skip_connect = conllx_stacked_data._generate_stack_inputs(inst.heads, inst.type_ids, self.prior_order)
            data.append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect])
            max_len = max([len(char_seq) for char_seq in sent.char_seqs])
            if max_char_length < max_len:
                max_char_length = max_len
            if max_length < inst.length():
                max_length = inst.length()
            inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
        reader.close()
        # print("Total number of data: %d" % counter)

        data_size = len(data)
        char_length = min(MAX_CHAR_LENGTH, max_char_length)
        wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        hid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        tid_inputs = np.empty([data_size, max_length], dtype=np.int64)

        masks_e = np.zeros([data_size, max_length], dtype=np.float32)
        single = np.zeros([data_size, max_length], dtype=np.int64)
        lengths = np.empty(data_size, dtype=np.int64)

        """
        stack_hid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
        chid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)

        masks_d = np.zeros([data_size, 2 * max_length - 1], dtype=np.float32)
        """

        stack_hid_inputs = np.empty([data_size, max_length - 1], dtype=np.int64)
        chid_inputs = np.empty([data_size, max_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([data_size, max_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([data_size, max_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([data_size, max_length - 1], dtype=np.int64)

        masks_d = np.zeros([data_size, max_length - 1], dtype=np.float32)
                            
        for i, inst in enumerate(data):
            wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks_e
            masks_e[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if self.word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            #inst_size_decoder = 2 * inst_size - 1
            inst_size_decoder = inst_size - 1 
            # stacked heads
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # siblings
            ssid_inputs[i, :inst_size_decoder] = ssids
            ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # skip connects
            skip_connect_inputs[i, :inst_size_decoder] = skip_ids
            skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # masks_d
            masks_d[i, :inst_size_decoder] = 1.0

        words = torch.from_numpy(wid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks_e = torch.from_numpy(masks_e)
        single = torch.from_numpy(single)
        lengths = torch.from_numpy(lengths)

        stacked_heads = torch.from_numpy(stack_hid_inputs)
        children = torch.from_numpy(chid_inputs)
        siblings = torch.from_numpy(ssid_inputs)
        stacked_types = torch.from_numpy(stack_tid_inputs)
        skip_connect = torch.from_numpy(skip_connect_inputs)
        masks_d = torch.from_numpy(masks_d)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                       'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                       'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d}
        return data_tensor, data_size


    @torch.no_grad()
    def parse(self, sentences, batch_size=2048):
        assert len(sentences) < batch_size
        # pos tagging
        in_docs = [stanza.Document([], text=sent) for sent in sentences]
        docs = self.pos_tagger(in_docs)

        # t0 = time.time()
        original_words = []
        data_lines = []

        # tmp_path = f'tmp{time.time()}'
        # os.mkdir(tmp_path)
        for idx, doc in enumerate(docs):
            doc_dicts = doc.to_dict() # List[List[Dict]]
            # avoid sentence cutting
            word_dicts = [d for sent_dicts in doc_dicts for d in sent_dicts]
            ws = ['_ROOT']
            for word_idx, d in enumerate(word_dicts):
                d['id'] = word_idx + 1
                ws.append(d['text'].lower())
                items = [str(d['id']), d['text'], '_', d['upos'], d['xpos'], 
                    d.get('feats', '_'), str(d['id']-1), '_', '_', 
                    f"start_char={d['start_char']}|end_char={d['end_char']}"]
                data_lines.append('\t'.join(items))
            data_lines.append('\n')
            original_words.append(ws)
        #     new_doc = stanza.Document([word_dicts])
        #     CoNLL.write_doc2conll(new_doc, os.path.join(tmp_path, f"output-{idx}.conllu"))
        #     os.system(f'cat {tmp_path}/output-{idx}.conllu >> {tmp_path}/output_all.conllu')
        #     # os.system(f'cat {tmp_path}/output_all.conllu')
        # conllu_to_conllx(f'{tmp_path}/output_all.conllu')

        # test_path = f'{tmp_path}/output_all.conllx'

        if self.alg == 'graph':
            data_test = conllx_data.read_data(test_path, self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet, symbolic_root=True)
        else:
            data_test = self._read_data(data_lines)

        # os.system(f'rm -r {tmp_path}')

        # print('data latency', time.time() - t0)
        
        beam = self.args.beam
        for data in iterate_data(data_test, batch_size):
            assert len(sentences) == len(original_words) == data['WORD'].shape[0]
            words = data['WORD'].to(self.device)
            chars = data['CHAR'].to(self.device)
            postags = data['POS'].to(self.device)
            heads = data['HEAD'].numpy()
            types = data['TYPE'].numpy()
            lengths = data['LENGTH'].numpy()

            # set get_prob=True to obtain two probs of last word
            if self.alg == 'graph':
                masks = data['MASK'].to(self.device)
                heads_pred, types_pred, all_arc_probs, all_type_probs = \
                    self.network.decode(words, chars, postags, mask=masks, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS, get_probs=True)
            else:
                # t1 = time.time()
                masks = data['MASK_ENC'].to(self.device)
                heads_pred, types_pred, all_arc_probs, all_type_probs = \
                    self.network.decode(words, chars, postags, mask=masks, beam=beam, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS, get_probs=True)
                # print('decoding latency', time.time() - t1)
            # print(heads_pred)
            # print(types_pred)
            # print(last_word_head_prob)
            # print(last_word_tail_prob)
            # print(all_arc_probs.shape)
            # print(all_type_probs.shape)
            # pdb.set_trace()
            # print(self.type_alphabet.instance2index)
            parse_result = []
            for idx in range(heads_pred.shape[0]):
                l = lengths[idx]
                # print(idx, len(original_words), len(sentences), words[idx], [self.word_alphabet.get_instance(x) for x in words[idx].tolist()])
                parse_result.append({
                    'words': original_words[idx],
                    'heads_pred': heads_pred[idx][: l],
                    'types_pred': types_pred[idx][: l],
                    'all_arc_probs': all_arc_probs[idx],
                    'all_type_probs': all_type_probs[idx],
                    'length': l

                })
            return parse_result


    def get_all_deps(self, sentences):
        parse_result = self.parse(sentences)

        all_pred_deps = []
        for pr in parse_result:
            pred_arcs = []
            pred_deps = []
            for child, (head, dep_label) in enumerate(zip(pr['heads_pred'], pr['types_pred'])):
                l = self.type_alphabet.get_instance(dep_label)
                if l == '_<PAD>':
                    continue
                pred_deps.append((pr['words'][head], 
                    l, 
                    pr['words'][child])
                )
            all_pred_deps.append(pred_deps)
        return all_pred_deps


    def get_dep_probs(self, checked_list):
        # t1 = time.time()
        parse_result = self.parse([c['string'] for c in checked_list])
        # print('parse latency', time.time() - t1)

        all_pred_deps = []
        all_dep_probs = []
        for idx, (pr, c) in enumerate(zip(parse_result, checked_list)):
            pred_arcs = []
            pred_deps = []
            for child, (head, dep_label) in enumerate(zip(pr['heads_pred'], pr['types_pred'])):
                try:
                    pred_arcs.append((pr['words'][head], pr['words'][child]))
                except:
                    pdb.set_trace()
                pred_deps.append((pr['words'][head], 
                    self.type_alphabet.get_instance(dep_label), 
                    pr['words'][child])
                )

            dep_probs_dict = {}
            for (w1, r, w2) in c['checked_constraints']:
                if w1 not in set(pr['words']) or w2 not in set(pr['words']):
                    dep_probs_dict[(w1, r, w2)] = (0.0, 0.0)
                    continue
                w1_indice, w2_indice = [], []
                for word_idx, word in enumerate(pr['words']):
                    if word == w1:
                        w1_indice.append(word_idx)
                    if word == w2:
                        w2_indice.append(word_idx)
                # if (w1, w2) not in set(pred_arcs):
                    # if w1 == pr['words'][-1]:
                    #     transition_prob = max(pr['last_word_head_prob'][w2_idx] for w2_idx in w2_indice)
                    #     # print()
                    # elif w2 == pr['words'][-1]:
                    #     transition_prob = max(pr['last_word_tail_prob'][w1_idx] for w1_idx in w1_indice)
                    #     print(all_pred_arcs, (w1, w2), c['string'], transition_prob)
                    # else:
                    #     # print(pr['words'])
                    #     # print('error!')
                    #     # print(all_pred_arcs, (w1, w2), c['string'])
                    #     transition_prob = 0.0
                    #     # exit()
                    # transition_prob = 0.0
                    # dep_probs[(w1, r, w2)] = transition_prob
                # else:
                transition_prob = 0.0
                for w1_idx in w1_indice:
                    for w2_idx in w2_indice:
                        if w1_idx == w2_idx:
                            continue
                        transition_prob = max(transition_prob, pr['all_arc_probs'][w2_idx-1][w1_idx])

                type_ = self.type_alphabet.get_index(r)
                type_prob = max(pr['all_type_probs'][w2_idx-1][type_] for w2_idx in w2_indice)
                dep_probs_dict[(w1, r, w2)] = (transition_prob, type_prob)
                # pdb.set_trace()
            all_dep_probs.append(dep_probs_dict)
            all_pred_deps.append(pred_deps)

        # print(all_dep_probs)
        return all_pred_deps, all_dep_probs


class StanzaParser(object):
    def __init__(self):
        self.parser = stanza.Pipeline(processors="tokenize,pos,lemma,depparse", 
            lang="en", use_gpu=True)

    def parse(self, sentences):
        in_docs = [stanza.Document([], text=s) for s in sentences]
        return self.parser(in_docs)

    def get_dep_probs(self, checked_list):
        all_pred_deps = self.get_all_deps([c['string'] for c in checked_list])
        all_dep_probs = []

        for idx, (pred_deps, c) in enumerate(zip(all_pred_deps, checked_list)):
            dep_probs_dict = {}
            for (w1, r, w2) in c['checked_constraints']:
                if [w1, r, w2] in pred_deps:
                    dep_probs_dict[(w1, r, w2)] = (1.0, 1.0)
                else:
                    dep_probs_dict[(w1, r, w2)] = (0.0, 0.0)
            all_dep_probs.append(dep_probs_dict)

        return all_pred_deps, all_dep_probs


    def get_all_deps(self, sentences):
        in_docs = [stanza.Document([], text=s) for s in sentences]
        results = []
        for doc in self.parser(in_docs):
            all_deps = []
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
                    all_deps.append([w1, r, w2])
            results.append(all_deps)
        return results


class SpaCyParser(object):
    def __init__(self):
        spacy.prefer_gpu()
        # spacy.require_gpu()
        self.parser = spacy.load("en_udv25_englishewt_trf")

    def get_all_deps(self, sentences):
        results = []
        docs = self.parser.pipe(sentences)
        for doc in docs:
            all_deps = []
            for sent in doc.sents:
                for token in sent:
                    w1 = token.head.text.lower()
                    r = token.dep_
                    w2 = token.text.lower()
                    all_deps.append([w1, r, w2])
            results.append(all_deps)
        return results


class RelChecker(object):
    def __init__(self):        
        self.model = FactRelModel()
        self.tokenizer = SimpleTokenizer('webedit_vocab.txt')
        self.model.load_state_dict(torch.load('/data1/chenxiang/webedit_rc_filt_aug/model.pt'))
        with open('rels.txt', 'r') as f:
            self.rel_dict = json.loads(f.read().strip())

    def _to_device(self, device):
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def parse(self, sentences, batch_size=10000):
        assert len(sentences) < batch_size
        new_X = [self.tokenizer(x) for x in sentences]
        # forward new X
        real_length = torch.tensor([len(x) for x in new_X])
        new_X = lstm_padding(new_X, max_length=max(real_length), padding_id=0).to(self.device)
        logits, (hidden_h, hidden_c), head_outputs, tail_outputs = self.model(new_X.transpose(0, 1), real_length)
        prob = F.softmax(logits, dim=-1)
        return prob

    def get_dep_probs(self, checked_list):
        # t1 = time.time()
        parse_result = self.parse([c['string'] for c in checked_list])
        # print('parse latency', time.time() - t1)

        all_pred_deps = []
        all_dep_probs = []
        for idx, (pr, c) in enumerate(zip(parse_result, checked_list)):
            dep_probs_dict = {}
            for (w1, r, w2) in c['checked_constraints']:
                # same entity can not have rels
                if w1 == w2:
                    continue
                if w1 not in set(c['string']) or w2 not in set(c['string']):
                    dep_probs_dict[(w1, r, w2)] = (0.0, 0.0)
                    continue
                w1_indice, w2_indice = [], []
                for word_idx, word in enumerate(c['string']):
                    if word == w1:
                        w1_indice.append(word_idx)
                    if word == w2:
                        w2_indice.append(word_idx)
                transition_prob = 0.0
                for w1_idx in w1_indice:
                    for w2_idx in w2_indice:
                        # adjecent entities have no relations
                        if abs(w1_idx - w2_idx) <= 1:
                            continue
                        transition_prob = max(transition_prob, pr[w1_idx][w2_idx][r].item())
                        # if transition_prob > 0.5:
                        #     print(c['string'])
                        #     print(w1, r, w2)
                        #     print(transition_prob)
                # type_ = self.type_alphabet.get_index(r)
                # type_prob = max(pr['all_type_probs'][w2_idx-1][type_] for w2_idx in w2_indice)
                dep_probs_dict[(w1, r, w2)] = (transition_prob, 1.0)
            all_dep_probs.append(dep_probs_dict)
            # all_pred_deps.append(pred_deps)
        # print(all_dep_probs)
        # pdb.set_trace()
        
        return all_dep_probs


class RelCheckerForWebnlg(object):
    def __init__(self):        
        self.model = WebNlgRelModel()
        self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
        self.model.load_state_dict(torch.load('/data1/chenxiang/webnlg_rc_filt/model.pt'))

    def _to_device(self, device):
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        print('relcheck device', self.device)
    
    @torch.no_grad()
    def parse(self, sentences, batch_size=10000):
        assert len(sentences) < batch_size
        new_X = batched_X = sentences
        if new_X:
            real_length = torch.tensor([len(x) for x in new_X])
            new_X = lstm_padding(new_X, max_length=max(real_length), padding_id=0).to(self.device)
            logits, _, _, _ = self.model(new_X.transpose(0, 1), real_length)
            prob = F.softmax(logits, dim=-1)
        return prob

    def get_dep_probs(self, checked_list):
        t1 = time.time()
        parse_result = self.parse([c['input_ids'] for c in checked_list])
        # print('parse latency', time.time() - t1)
        # pdb.set_trace()

        all_dep_probs = []
        for idx, (pr, c) in enumerate(zip(parse_result, checked_list)):
            dep_probs_dict = {}
            for (w1, r, w2) in c['checked_constraints']:
                # same entity can not have rels
                if w1 == w2:
                    continue
                # if w1 not in set(c['string']) or w2 not in set(c['string']):
                #     dep_probs_dict[(w1, r, w2)] = (0.0, 0.0)
                #     continue

                w1_ids = self.tokenizer.encode(w1, add_special_tokens=False)
                w2_ids = self.tokenizer.encode(w2, add_special_tokens=False)
                input_ids = c['input_ids']
                # pdb.set_trace()
                w1_indice, w2_indice = match_lst(input_ids, w1_ids), match_lst(input_ids, w2_ids)
                
                transition_prob = 0.0
                for w1_idx in w1_indice:
                    for w2_idx in w2_indice:
                        # adjecent entities have no relations
                        if abs(w1_idx - w2_idx) <= 1:
                            continue
                        transition_prob = max(transition_prob, pr[w1_idx][w2_idx][r].item())
                dep_probs_dict[(w1, r, w2)] = (transition_prob, 1.0)
            all_dep_probs.append(dep_probs_dict)
        # print('parse latency', time.time() - t1)
        
        return all_dep_probs



