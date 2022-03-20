from math import ceil
from time import time

import torch
from transformers import BertModel, AdamW
from tqdm import trange

from network import CompBert, get_scheduler, train, evaluate, epoch_time
from data_preprocessing import prepare_data, get_data_length

MAX_TOKEN_LENGTH = 512
NUMBER_OF_EPOCHS = 16
BATCH_SIZE = 16

TRAIN_DATA = 'data/lcp_single_train.tsv'
TEST_DATA = 'data/lcp_single_test_labels.tsv'

train_data_len = get_data_length(16, TRAIN_DATA)

bert_s = BertModel.from_pretrained('prajjwal1/bert-small')
bert_w = BertModel.from_pretrained('prajjwal1/bert-small')

device = torch.device('cuda')

model = CompBert(bert_w, bert_s, 1).half().to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)

warmup_percent = 0.2
total_steps = ceil(NUMBER_OF_EPOCHS*train_data_len*1./BATCH_SIZE)
warmup_steps = int(total_steps*warmup_percent)
criterion = torch.nn.MSELoss().to(device)
scheduler = get_scheduler(optimizer, warmup_steps)
train_x, train_y = prepare_data(BATCH_SIZE, TRAIN_DATA, MAX_TOKEN_LENGTH)
valid_x, valid_y = prepare_data(BATCH_SIZE, TEST_DATA, MAX_TOKEN_LENGTH)

best_valid_loss = float('inf')

for epoch in trange(NUMBER_OF_EPOCHS, desc='Epochs'):
    start_time = time()

    train_loss = train(model, train_x, train_y, optimizer, criterion, scheduler, device)
    valid_loss = evaluate(model, valid_x, valid_y, optimizer, criterion, scheduler, device)

    end_time = time()

    epoch_minutes, epoch_seconds = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'complexity_predictor_eng.pt')

    print(f'Epoch: {epoch+1:02} | Epoch time: {epoch_minutes}m {epoch_seconds}s')
    print(f'\tTrain loss: {train_loss:.3f}')
    print(f'\t Validation loss {valid_loss:.3f}')

model.train()
