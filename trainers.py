from cmath import log
import torch
import sys
from my_utils import AverageMeter
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
import pdb
from pathlib import Path
from utils import label_smoothed_nll_loss


class BaseTrainer(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

        total_steps = len(train_loader) * args.epoch_num
        print(f'total_steps: {total_steps}', file=sys.stderr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=args.warmup * total_steps,
                                                         num_training_steps=total_steps)

    def train_one_epoch(self, device, epoch):
        loss_meter, nll_loss_meter = AverageMeter(), AverageMeter()
        self.model.train()
        for i, inputs in enumerate(self.train_loader):
            # train batch
            inputs = {k: inputs[k].to(device) for k in inputs}

            if 't5' in self.args.pretrain_model.lower():
                lm_labels = inputs["decoder_input_ids"]
                lm_labels[lm_labels[:, :] == 0] = -100
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=lm_labels,
                    decoder_attention_mask=inputs['decoder_attention_mask']
                )
                loss = outputs.loss
                nll_loss = outputs.loss
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits
                loss, nll_loss = label_smoothed_nll_loss(logits, inputs)
            loss_meter.update(loss.item())
            nll_loss_meter.update(nll_loss.item())
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) ' \
                    f'| nll_loss: {nll_loss_meter.val:.4f}({nll_loss_meter.avg:.4f})', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm, norm_type=2)
            self.optimizer.step()
            self.scheduler.step()
            # if i >= 0 and i % 2000 == 0:
            #     print('save!')
            #     model_dir = Path(self.args.output_dir) / 'bart'
            #     torch.save(self.model.state_dict(), f'{model_dir}_e{epoch}_s{i}.pt')
            if self.args.debug:
                break


class DepPredTrainer(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

        total_steps = len(train_loader) * args.epoch_num
        print(f'total_steps: {total_steps}', file=sys.stderr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=args.warmup * total_steps,
                                                         num_training_steps=total_steps)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def masked_ce_loss(self, logits, labels, mask):
        loss = self.ce_loss(logits.view(-1, 2), labels.view(-1))
        loss = (loss * mask.view(-1))
        det = mask.sum()
        if det.item() == 0:
            return torch.tensor(0.0, requires_grad=True).to(mask.device)
        return loss.sum() / det

    def cal_metrics(self, logits, labels, mask):
        word_sum = mask.sum(-1).tolist()
        pred_labels = torch.argmax(logits, dim=-1).tolist()
        gold_labels = labels.tolist()
        tp, total_pred, total = 0, 0, 0
        for s, pred_label, gold_label in zip(word_sum, pred_labels, gold_labels):
            for pred, gold in zip(pred_label[:s], gold_label[:s]):
                if pred == gold == 1:
                    tp += 1
                if pred == 1:
                    total_pred += 1
                if gold == 1:
                    total += 1
        return tp, total_pred, total

    def get_f1(self, tp, total_pred, total):
        precision = tp / (total_pred + 1e-10)
        recall = tp / (total + 1e-10)
        F1 = 2 * precision * recall / (precision + recall + 1e-10)  
        return precision, recall, F1

    def train_one_epoch(self, device, epoch):
        error_cnt, total_cnt = 0, 0
        loss_meter = AverageMeter()
        self.model.train()
        tp, total_pred, total = 0, 0, 0
        tp1, total_pred1, total1 = 0, 0, 0
        for i, inputs in enumerate(self.train_loader):
            # print(inputs)
            # train batch
            inputs = {k: inputs[k].to(device) for k in inputs}
            logits, keyword_logits = self.model(**inputs)
            dep_loss = self.masked_ce_loss(logits, inputs['labels'], inputs['word_cnt_mask'])
            keyword_loss = self.masked_ce_loss(keyword_logits, inputs['keyword_labels'], inputs['word_cnt_mask'])
            loss = 0 * dep_loss + keyword_loss
            loss_meter.update(loss.item())
            x, y, z = self.cal_metrics(logits, inputs['labels'], inputs['word_cnt_mask'])
            tp += x
            total_pred += y
            total += z
            precision, recall, F1 = self.get_f1(tp, total_pred, total)
            x, y, z = self.cal_metrics(keyword_logits, inputs['keyword_labels'], inputs['word_cnt_mask'])
            tp1 += x
            total_pred1 += y
            total1 += z
            precision1, recall1, F11 = self.get_f1(tp1, total_pred1, total1)
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f})' \
                    f'| p: {precision:.4f} ({tp}/{total_pred})' \
                    f'| r: {recall:.4f} ({tp}/{total})' \
                    f'| f: {F1:.4f}' \
                    f'| k_p: {precision1:.4f} ({tp1}/{total_pred1})' \
                    f'| k_r: {recall1:.4f} ({tp1}/{total1})' \
                    f'| k_f: {F11:.4f}', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm, norm_type=2)
            self.optimizer.step()
            self.scheduler.step()
            if i >= 0 and i % 2000 == 0:
                print('save!')
                model_dir = Path(self.args.output_dir) / 'bert'
                torch.save(self.model.state_dict(), f'{model_dir}_e{epoch}_s{i}.pt')
            if self.args.debug:
                break


class LSTMTrainer(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

        def init_weights(m):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    torch.nn.init.constant_(param.data, 0)
        self.model.apply(init_weights)

        # total_steps = len(train_loader) * args.epoch_num
        # print(f'total_steps: {total_steps}', file=sys.stderr)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
        #                                                  num_warmup_steps=args.warmup * total_steps,
        #                                                  num_training_steps=total_steps)
            
    def train_one_epoch(self, device, epoch):
        loss_meter, nll_loss_meter = AverageMeter(), AverageMeter()
        # self.model = self.model.to(device)
        # self.criterion = self.criterion.to(device)
        self.model.train()
        for i, inputs in enumerate(self.train_loader):
            # train batch
            inputs = {k: inputs[k].to(device) for k in inputs}
            # pdb.set_trace()
            # output = self.model(
            #     inputs['input_ids'].transpose(0, 1), 
            #     inputs['decoder_input_ids'].transpose(0, 1), 
            #     inputs['real_length'].cpu()
            # )
            output = self.model(
                inputs['triple_input_ids'].transpose(0, 1),
                inputs['draft_input_ids'].transpose(0, 1),
                inputs['decoder_input_ids'].transpose(0, 1),
                inputs['triple_real_length'].cpu(),
                inputs['draft_real_length'].cpu(),
                inputs['triple_mask']
            )

            # loss, nll_loss = label_smoothed_nll_loss(logits, inputs)
            # loss_meter.update(loss.item())
            # nll_loss_meter.update(nll_loss.item())
            output_dim = output.shape[-1]
            # output = output[1:].view(-1, output_dim)
            # trg = inputs['decoder_input_ids'].transpose(0, 1).contiguous()[1:].view(-1)
            # trg = inputs['decoder_input_ids'].transpose(0, 1).contiguous()[1:]
            # pdb.set_trace()
            output = output[1:].log().permute(1, 0, 2).contiguous()
            trg = inputs['decoder_input_ids'][:, 1:]
            lprobs = output.gather(dim=-1, index=trg.unsqueeze(-1))
            trg_mask = (trg != 0).unsqueeze(-1)
            # pdb.set_trace()
            det = trg_mask.sum()
            loss = -(lprobs * trg_mask).sum() / det
            # pdb.set_trace()
            # loss, nll_loss = label_smoothed_nll_loss(logits=output[1:].transpose(0, 1), inputs=inputs, ignore_index=0)
            # loss = self.criterion(output, trg)
            loss_meter.update(loss.item())
            nll_loss_meter.update(loss.item())

            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) ' \
                    f'| nll loss: {nll_loss_meter.val:.4f}({nll_loss_meter.avg:.4f})', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
            self.optimizer.step()
            # self.scheduler.step()
            if self.args.debug:
                break


class RelPredTrainer(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
        # def init_weights(m):
        #     for name, param in m.named_parameters():
        #         if 'weight' in name:
        #             torch.nn.init.normal_(param.data, mean=0, std=0.01)
        #         else:
        #             torch.nn.init.constant_(param.data, 0)
        # self.model.apply(init_weights)
            
    def train_one_epoch(self, device, epoch):
        loss_meter = AverageMeter() 
        self.model.train()
        for i, inputs in enumerate(self.train_loader):
            # train batch
            inputs = {k: inputs[k].to(device) for k in inputs}
            logits, hidden, _, _ = self.model(inputs['input_ids'].transpose(0, 1), inputs['real_length'].cpu())

            # entity_masks = inputs[entity_masks]
            # pdb.set_trace()
            loss = self.criterion(logits.view(-1, logits.shape[-1]), inputs['rel_labels'].contiguous().view(-1))
            loss_meter.update(loss.item())

            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.10f}({loss_meter.avg:.10f}) ', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
            self.optimizer.step()
            # self.scheduler.step()
            if self.args.debug:
                break


class BaseTrainerWithPredHead(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        total_steps = len(train_loader) * args.epoch_num
        print(f'total_steps: {total_steps}', file=sys.stderr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=args.warmup * total_steps,
                                                         num_training_steps=total_steps)

    def train_one_epoch(self, device, epoch):
        loss_meter, nll_loss_meter, cls_loss_meter = AverageMeter(), AverageMeter(), AverageMeter()
        self.model.train()
        for i, inputs in enumerate(self.train_loader):
            # train batch
            inputs = {k: inputs[k].to(device) for k in inputs}

            if 't5' in self.args.pretrain_model.lower():
                lm_labels = inputs["decoder_input_ids"]
                lm_labels[lm_labels[:, :] == 0] = -100
                nll_loss, logits = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=lm_labels,
                    decoder_attention_mask=inputs['decoder_attention_mask']
                )
            elif 'bart' in self.args.pretrain_model.lower():
                nll_loss, logits = self.model(**inputs)
            else:
                raise NotImplementedError
            nll_loss = nll_loss.sum()
            cls_loss = self.ce_loss(logits.view(-1, 2), inputs['constraint_label'].view(-1))
            loss = nll_loss + cls_loss

            if str(cls_loss.item()) == 'nan':
                pdb.set_trace()

            loss_meter.update(loss.item())
            nll_loss_meter.update(nll_loss.item())
            cls_loss_meter.update(cls_loss.item())
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) ' \
                    f'| nll_loss: {nll_loss_meter.val:.4f}({nll_loss_meter.avg:.4f})', \
                    f'| cls_loss: {cls_loss_meter.val:.4f}({cls_loss_meter.avg:.4f})', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm, norm_type=2)
            self.optimizer.step()
            self.scheduler.step()
            if self.args.debug:
                break


trainer_factory = {
    'gigaword': BaseTrainerWithPredHead,
    'english-ewt': BaseTrainer,
    'Giga_Dep': BaseTrainer,
    'Giga_Dep_Pred': DepPredTrainer,
    'webedit': BaseTrainer,
    'webedit_lstm': LSTMTrainer,
    'webedit_rel': RelPredTrainer,
    'webnlg': BaseTrainerWithPredHead,
    'webnlg_rel': RelPredTrainer
}
