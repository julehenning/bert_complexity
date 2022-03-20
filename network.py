import torch
import torch.nn as nn
from transformers import BertModel, get_constant_schedule_with_warmup
from tqdm import tqdm


class CompBert(nn.Module):
    def __init__(self, bert_word: BertModel, bert_sentence: BertModel, output_dim: int):
        super().__init__()

        self.bert_sentence = bert_sentence
        self.bert_word = bert_word

        embedding_dim_s = bert_sentence.config.to_dict()['hidden_size']
        embedding_dim_w = bert_word.config.to_dict()['hidden_size']

        self.out_sentence = nn.Linear(embedding_dim_s, output_dim)
        self.out_word = nn.Linear(embedding_dim_w, output_dim)

    def forward(self, sentence_input_ids, sentence_token_type_ids, sentence_attention_masks,
                word_input_ids, word_token_type_ids, word_attention_masks):
        embedded_sentence = self.bert_sentence(input_ids=sentence_input_ids, token_type_ids=sentence_token_type_ids,
                                               attention_mask=sentence_attention_masks).pooler_output
        embedded_word = self.bert_word(input_ids=word_input_ids, token_type_ids=word_token_type_ids,
                                       attention_mask=word_attention_masks).pooler_output

        output_sentence = torch.sigmoid(self.out_sentence(embedded_sentence))
        output_word = torch.sigmoid(self.out_word(embedded_word))

        output = output_word * 0.8 + output_sentence * 0.2

        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args)


def get_scheduler(optimizer, warmup_steps):
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    return scheduler


def train(model, xs, ys, optimizer, criterion, scheduler, device):
    """
    Trains the model
    :param model: Model to be trained
    :param xs: Cases
    :param ys: Labels
    :param optimizer: Optimizer
    :param criterion: Criterion
    :param scheduler: Scheduler
    :return: Returns epoch loss and accuracy
    """

    epoch_loss = 0

    model.train()

    for i, y in enumerate(tqdm(ys, desc='Training', position=0, leave=True)):
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        x = tuple(tensor[i].to(device) for tensor in xs)
        y = y.to(device)
        predictions = model(*x)
        loss = criterion(predictions, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(ys)


def evaluate(model, xs, ys, optimizer, criterion, scheduler, device):
    """
    Evaluates the model
    :param model: Model to be evaluated
    :param xs: Cases
    :param ys: Labels
    :param optimizer: Optimizer
    :param criterion: Criterion
    :param scheduler: Scheduler
    :return: Returns epoch loss
    """
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for i, y in enumerate(ys):
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            x = tuple(tensor[i].to(device) for tensor in xs)
            y = y.to(device)

            predictions = model(*x)
            loss = criterion(predictions, y)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

    return epoch_loss / len(ys)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time - (elapsed_minutes * 60))
    return elapsed_minutes, elapsed_seconds

