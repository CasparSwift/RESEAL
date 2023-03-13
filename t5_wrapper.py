import torch.nn as nn
import torch
import transformers
import torch.nn.functional as F
from utils import label_smoothed_nll_loss


class T5Wrapper(nn.Module):
    def __init__(self, model, with_predictive_head=True):
        super(T5Wrapper, self).__init__()
        self.model = model
        self.with_predictive_head = with_predictive_head
        if with_predictive_head:
            self.classifier = nn.Sequential(
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Linear(384, 2)
            )
    
    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        if isinstance(outputs, tuple):
            nll_loss = outputs[0]
        else:
            nll_loss = outputs.loss
        if self.with_predictive_head:
            if isinstance(outputs, tuple):
                logits = self.classifier(outputs[-1])
            else:
                logits = self.classifier(outputs.encoder_last_hidden_state)
            return nll_loss, logits
        else:
            return nll_loss
    
    def get_constraints_words(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        if isinstance(outputs, tuple):
            logits = self.classifier(outputs[-1])
        else:
            logits = self.classifier(outputs.encoder_last_hidden_state)
        return F.softmax(logits, dim=-1)[:, :, -1]

    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
    
    def resize_token_embeddings(self, length):
        self.model.resize_token_embeddings(length)


class BARTWrapper(nn.Module):
    def __init__(self, model, with_predictive_head=True):
        super(BARTWrapper, self).__init__()
        self.model = model
        self.with_predictive_head = with_predictive_head
        if with_predictive_head:
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )
    
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        if isinstance(outputs, tuple):
            nll_loss, _ = label_smoothed_nll_loss(outputs[0], decoder_input_ids)
        else:
            nll_loss, _ = label_smoothed_nll_loss(outputs.logits, decoder_input_ids)
        if self.with_predictive_head:
            if isinstance(outputs, tuple):
                logits = self.classifier(outputs[-1])
            else:
                logits = self.classifier(outputs.encoder_last_hidden_state)
            return nll_loss, logits
        else:
            return nll_loss
    
    def get_constraints_words(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        if isinstance(outputs, tuple):
            logits = self.classifier(outputs[-1])
        else:
            logits = self.classifier(outputs.encoder_last_hidden_state)
        return F.softmax(logits, dim=-1)[:, :, -1]

    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)