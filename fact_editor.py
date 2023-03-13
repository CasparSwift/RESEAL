import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pdb
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, embedding):
        super(GRUEncoder, self).__init__()
        self.embedding = embedding
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, num_layers=1)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, real_length):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, real_length, batch_first=False, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=False)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden


class SimpleAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attention_hid_dim):
        super(SimpleAttention, self).__init__()
        hidden_size = 300
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, attention_hid_dim)
        self.v = nn.Linear(attention_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # pdb.set_trace()
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim = 1)


class GRUDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(GRUDecoder, self).__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=1)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
          
        weighted = torch.bmm(a, encoder_outputs)    
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  
        return prediction, hidden.squeeze(0)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.Q = nn.Linear(dec_hid_dim, enc_hid_dim * 2, bias=False)
        self.K = nn.Linear(enc_hid_dim * 2, enc_hid_dim * 2, bias=False)
        self.V = nn.Linear(enc_hid_dim * 2, enc_hid_dim * 2, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        num_layers = hidden.shape[0]

        # [num_layers, bsz, 300 * 2]
        query = self.Q(hidden)
        # [src_len, bsz, 300 * 2]
        key = self.K(encoder_outputs)
        # [src_len, bsz, 300 * 2]
        value = self.V(encoder_outputs)
        # [bsz, num_layers, src_len]
        attention = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0))
        # pdb.set_trace()
        attention = attention.masked_fill(mask.unsqueeze(1).repeat(1, num_layers, 1) == 0, -1e10)
        attention = F.softmax(attention, dim=-1)
        return torch.bmm(attention, value.permute(1, 0, 2))


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(LSTMEncoder, self).__init__()
        self.num_layers = 2
        self.num_directions = 2
        self.embedding = nn.Embedding(input_dim, emb_dim)   
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=(self.num_directions == 2), num_layers=self.num_layers)
        self.fc_h = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_c = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, real_length):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, real_length, batch_first=False, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=False)

        hidden1 = torch.tanh(self.fc_h(torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)))
        hidden2 = torch.tanh(self.fc_h(torch.cat((hidden[2,:,:], hidden[3,:,:]), dim=1)))

        cell1 = torch.tanh(self.fc_c(torch.cat((cell[0,:,:], cell[1,:,:]), dim=1)))
        cell2 = torch.tanh(self.fc_c(torch.cat((cell[2,:,:], cell[3,:,:]), dim=1)))
        
        # [2, batch_size, decoder_hid_dim]
        hidden = torch.stack([hidden1, hidden2], dim=0)
        cell = torch.stack([cell1, cell2], dim=0)
        return outputs, (hidden, cell)


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(LSTMDecoder, self).__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(enc_hid_dim * 2 * 2 + emb_dim, dec_hid_dim, num_layers=2)
        self.fc_out = nn.Linear(enc_hid_dim * 2 * 2 + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden_states, encoder_outputs, mask):
        hidden, cell = hidden_states
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        # [batch_size, num_layers, 300 * 2]
        weighted = self.attention(hidden, encoder_outputs, mask)

        # [1, batch_size, 300 * 2 * num_layers]
        weighted = torch.cat((weighted[:, 0, :], weighted[:, 1, :]), dim=1).unsqueeze(0)

        # [1, batch_size, 300 * 2 * num_layers + dec_hid_dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  
        return prediction, (hidden, cell)


class FactBaseGRUModel(nn.Module):
    def __init__(self) -> None:
        super(FactBaseGRUModel, self).__init__()
        INPUT_DIM = 4176
        OUTPUT_DIM = 4176
        ENC_EMB_DIM = 300
        DEC_EMB_DIM = 300
        ENC_HID_DIM = 300
        DEC_HID_DIM = 300
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5

        attn = SimpleAttention(ENC_HID_DIM, DEC_HID_DIM)
        self.encoder = GRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        self.decoder = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
        self.pad_id = 0
    
    def forward(self, src, trg, real_length, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device)

        encoder_outputs, hidden = self.encoder(src, real_length)
    
        input = trg[0,:]
        mask = (src != self.pad_id).permute(1, 0)
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1

        return outputs


class FactBaseModel(nn.Module):
    def __init__(self) -> None:
        super(FactBaseModel, self).__init__()
        INPUT_DIM = 4176
        OUTPUT_DIM = 4176
        ENC_EMB_DIM = 300
        DEC_EMB_DIM = 300
        ENC_HID_DIM = 300
        DEC_HID_DIM = 600
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5

        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        self.encoder = LSTMEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        self.decoder = LSTMDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

        self.pad_id = 0
    
    def forward(self, src, trg, real_length, teacher_forcing_ratio=0.75):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device)

        encoder_outputs, (hidden, cell) = self.encoder(src, real_length)
    
        input = trg[0,:]
        mask = (src != self.pad_id).permute(1, 0)
        
        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input, (hidden, cell), encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1

        return outputs


class DualGRUDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, copy_attention, embedding):
        super(DualGRUDecoder, self).__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = embedding
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=1)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, output_dim)

        self.copy_attention = copy_attention
        self.copy_gate = nn.Linear((enc_hid_dim * 2) + emb_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.pad_id = 0
        self.unk_id = 1
        
    def forward(self, input, hidden, src1, encoder_outputs1, mask1, triple_mask, encoder_outputs2, mask2):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        encoder_outputs = torch.cat((encoder_outputs1, encoder_outputs2), dim=0)
        mask = torch.cat((mask1, mask2), dim=1)
        # pdb.set_trace()
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
          
        weighted = torch.bmm(a, encoder_outputs)    
        weighted = weighted.permute(1, 0, 2)

        # pdb.set_trace()
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        state = torch.cat((output, weighted), dim=1)
        # B * vocab_size
        vocab_dist = F.softmax(self.fc_out(state), dim=-1)

        p_gen = torch.sigmoid(self.copy_gate(torch.cat((weighted, embedded), dim=-1)))
        
        # pdb.set_trace()
        # print(mask1 * triple_mask)
        attn_dist = self.copy_attention(hidden.squeeze(0), encoder_outputs1, mask1 * triple_mask)

        # pdb.set_trace()
        
        # memory-consuming !
        # [bsz, src_len, vocab]
        # one_hot = torch.zeros(src1.shape[1], src1.shape[0], score_gen.shape[-1]).to(score_gen.device)
        # self.one_hot.zero_()
        # one_hot.scatter_(2, src1.transpose(0, 1).unsqueeze(2), 1)

        # [bsz, 1, src_len] * [bsz, src_len, vocab]
        # score_copy = torch.bmm(a_copy, one_hot).squeeze(1)

        # score_copy = torch.zeros_like(score_gen).to(score_gen.device)
        # score_copy.scatter_add_(1, src1.transpose(0, 1), a_copy.squeeze(1))

        # if (p_gate > 0.5).sum() > 0:
        #     print(p_gate)
        #     pdb.set_trace()

        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist

        # pdb.set_trace()

        final_dist = vocab_dist_.scatter_add(1, src1.transpose(0, 1), attn_dist_)
        # pdb.set_trace()
        
        return final_dist, hidden.squeeze(0)


class FactBaseDualModel(nn.Module):
    def __init__(self, vocab_dim, embed_dim, enc_hid_dim, dec_hid_dim, attention_hid_dim) -> None:
        super(FactBaseDualModel, self).__init__()
        INPUT_DIM = OUTPUT_DIM = vocab_dim
        TEXT_ENC_EMB_DIM = TABLE_ENC_EMB_DIM = DEC_EMB_DIM = embed_dim
        TEXT_ENC_HID_DIM = TABLE_ENC_HID_DIM = enc_hid_dim
        DEC_HID_DIM = dec_hid_dim
        ENC_DROPOUT = DEC_DROPOUT = 0.5

        # self.outputs_holder = torch.zeros(args.max_tgt_length, 
        #     args.batch_size if not args.test else args.test_batch_size, OUTPUT_DIM).cuda()
        # one_hot = torch.zeros(args.batch_size if not args.test else args.test_batch_size, 
        #     args.max_src_length, OUTPUT_DIM).cuda()
        # one_hot.requires_grad = False
        embedding = nn.Embedding(vocab_dim, embed_dim)
        attn = SimpleAttention(TABLE_ENC_HID_DIM, DEC_HID_DIM, attention_hid_dim)
        copy_attn = SimpleAttention(TABLE_ENC_HID_DIM, DEC_HID_DIM, attention_hid_dim)
        self.table_encoder = GRUEncoder(INPUT_DIM, TABLE_ENC_EMB_DIM, TABLE_ENC_HID_DIM, DEC_HID_DIM // 2, ENC_DROPOUT, embedding)
        self.text_encoder = GRUEncoder(INPUT_DIM, TEXT_ENC_EMB_DIM, TEXT_ENC_HID_DIM, DEC_HID_DIM // 2, ENC_DROPOUT, embedding)
        self.decoder = DualGRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, (TABLE_ENC_HID_DIM + TEXT_ENC_HID_DIM) // 2, 
            DEC_HID_DIM, DEC_DROPOUT, attn, copy_attn, embedding)
        self.pad_id = 0
        self.unk_id = 1
        
    
    def forward(self, src1, src2, trg, real_length1, real_length2, triple_mask, teacher_forcing_ratio=0.5):
        batch_size = src1.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src1.device)
        # self.outputs_holder.zero_()

        encoder_outputs1, hidden1 = self.table_encoder(src1, real_length1)
        encoder_outputs2, hidden2 = self.text_encoder(src2, real_length2)
        hidden = torch.cat((hidden1, hidden2), dim=1)
    
        input = trg[0,:]
        mask1 = (src1 != self.pad_id).permute(1, 0)
        mask2 = (src2 != self.pad_id).permute(1, 0)
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, src1, encoder_outputs1, mask1, triple_mask, encoder_outputs2, mask2)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1

        return outputs


class FactRelModel(nn.Module):
    def __init__(self, input_dim=4176, emb_dim=300, enc_hid_dim=300, dropout=0.5):
        super(FactRelModel, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)   
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=False, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.head_mlp = nn.Sequential(
            nn.Linear(enc_hid_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 300)
        )
        self.tail_mlp = nn.Sequential(
            nn.Linear(enc_hid_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 300)
        )
        self.relation_num = 229
        # self.relation_num = 2
        self.biaffine = nn.Parameter(torch.randn(300 + 1, self.relation_num, 300 + 1), requires_grad=True)
        
    def forward(self, src, real_length):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, 
                real_length, batch_first=False, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # [bsz, src_len, hidden_size]
        # outputs = outputs.permute(1, 0, 2)

        head_outputs = self.head_mlp(outputs)
        tail_outputs = self.tail_mlp(outputs)

        x = torch.cat((head_outputs, torch.ones_like(head_outputs[..., :1])), dim=-1)
        y = torch.cat((tail_outputs, torch.ones_like(tail_outputs[..., :1])), dim=-1)
        logits = torch.einsum('bxi,ioj,byj->bxyo', x, self.biaffine, y).contiguous()
        
        return logits, hidden, head_outputs, tail_outputs
    
    def forward_one_step(self, src, prev_hidden, prev_head_outputs, prev_tail_outputs):
        # src = [1, batch_size]
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded, prev_hidden)
        # [bsz, 1, hidden_size]
        outputs = outputs.permute(1, 0, 2)

        one_head_outputs = self.head_mlp(outputs)
        one_tail_outputs = self.tail_mlp(outputs)

        head_outputs = torch.cat((prev_head_outputs, one_head_outputs), dim=1)
        tail_outputs = torch.cat((prev_tail_outputs, one_tail_outputs), dim=1)
        hidden = torch.cat((prev_hidden, hidden), dim=0)

        x = torch.cat((head_outputs, torch.ones_like(head_outputs[..., :1])), dim=-1)
        y = torch.cat((tail_outputs, torch.ones_like(tail_outputs[..., :1])), dim=-1)
        x1 = torch.cat((one_head_outputs, torch.ones_like(one_head_outputs[..., :1])), dim=-1)
        y1 = torch.cat((one_tail_outputs, torch.ones_like(one_tail_outputs[..., :1])), dim=-1)

        head_logits = torch.einsum('bxi,ioj,byj->bxyo', x, self.biaffine, y1).contiguous()
        tail_logits = torch.einsum('bxi,ioj,byj->bxyo', x1, self.biaffine, y).contiguous()

        pdb.set_trace()

        return head_logits, tail_logits, hidden, head_outputs, tail_outputs

