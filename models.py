import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DepPredBertModel(nn.Module):
    def __init__(self, args):
        super(DepPredBertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pretrain_model)

        self.pred_dep = True
        if self.pred_dep:
            self.classifier = nn.Sequential(
                nn.Linear(768 * 2 + 200, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(768 * 2 + 200, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 2)
            )
            self.keyword_classifier = nn.Sequential(
                nn.Linear(768, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 2)
            )
        self.dep_embedding = nn.Embedding(50, 200)

    def forward(self, **inputs):
        output = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        # [batch, src_len, 768]
        features = output.last_hidden_state
        batch_size = features.shape[0]
        word_cnt = inputs['heads'].shape[1]
        feature_dim = features.shape[2]

        # get word level representations
        start_indice = inputs['word_id_to_token_span'][:, :, 0] \
            .unsqueeze(-1).expand(batch_size, word_cnt, feature_dim)
        end_indice = inputs['word_id_to_token_span'][:, :, 1] \
            .unsqueeze(-1).expand(batch_size, word_cnt, feature_dim)
        # [batch, src_len, 768]
        start_feats = features.gather(1, start_indice)
        end_feats = features.gather(1, end_indice)
        mean_feats = (start_indice + end_feats) / 2

        # for head words
        heads_indice = inputs['heads'].unsqueeze(-1).expand(batch_size, word_cnt, feature_dim)
        head_feats = mean_feats.gather(1, heads_indice)

        # for dep relations
        dep_feats = self.dep_embedding(inputs['dep_rels'])
        # flat_dep_feats = inputs['dep_rels'].view(-1).unsqueeze(-1)
        # dep_feats = torch.zeros((batch_size * word_cnt, 50), requires_grad=False).to(features.device)
        # dep_feats.scatter_(1, flat_dep_feats, 1)
        # dep_feats = dep_feats.view(batch_size, word_cnt, 50)
        # pdb.set_trace()

        # fusion_vector = torch.cat([mean_feats, head_feats, inputs['dep_rels']], dim=-1)
        fusion_vector = torch.cat([mean_feats, head_feats, dep_feats], dim=-1)

        # [batch, src_len, 2]
        logits = self.classifier(fusion_vector)
        if not self.pred_dep:
            keyword_logits = self.keyword_classifier(mean_feats)
        else:
            keyword_logits = logits.clone()
        return logits, keyword_logits


class WebNlgRelModel(nn.Module):
    def __init__(self, input_dim=32128, emb_dim=300, enc_hid_dim=300, dropout=0.5):
        super(WebNlgRelModel, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)   
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.head_mlp = nn.Sequential(
            nn.Linear(enc_hid_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 300)
        )
        self.tail_mlp = nn.Sequential(
            nn.Linear(enc_hid_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 300)
        )
        # self.relation_num = 229
        self.relation_num = 3
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
