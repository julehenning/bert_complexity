from transformers import AutoTokenizer
import torch


def prepare_data(batch_size: int, data_path: str, max_token_length: int):
    """
    Prepare data for neural network

    :param batch_size: The batch size to be used by the network
    :param data_path: The path of the data to prepare
    :param max_token_length: Maximum token length of data
    :return: Tokenized and batched data and batched complexity labels
    """

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)

    data = open(data_path, 'r', encoding='utf8')

    lines = [line for line in data.read().splitlines()[1:]]
    sentences, words, complexities = [], [], []

    data.close()

    number_of_batches = len(lines)//batch_size
    data_size = number_of_batches*batch_size

    for line in lines[:data_size]:
        sentence, word, complexity = line.split('\t')[2:5]
        sentences.append(sentence)
        words.append(word)
        complexities.append(float(complexity))

    encoded_sentences = tokenizer(sentences,
                                  add_special_tokens=True, max_length=max_token_length, padding='max_length',
                                  truncation=True, return_tensors='pt')
    encoded_words = tokenizer(words,
                              add_special_tokens=True, max_length=max_token_length, padding='max_length',
                              truncation=True, return_tensors='pt')

    complexities = torch.Tensor(complexities)

    encoded_sentences['input_ids'] = encoded_sentences['input_ids'].reshape(number_of_batches,
                                                                            batch_size,
                                                                            max_token_length).int()
    encoded_sentences['token_type_ids'] = encoded_sentences['token_type_ids'].reshape(number_of_batches,
                                                                                      batch_size,
                                                                                      max_token_length).int()
    encoded_sentences['attention_mask'] = encoded_sentences['attention_mask'].reshape(number_of_batches,
                                                                                      batch_size,
                                                                                      max_token_length).int()
    encoded_words['input_ids'] = encoded_words['input_ids'].reshape(number_of_batches,
                                                                    batch_size,
                                                                    max_token_length).int()
    encoded_words['token_type_ids'] = encoded_words['token_type_ids'].reshape(number_of_batches,
                                                                              batch_size,
                                                                              max_token_length).int()
    encoded_words['attention_mask'] = encoded_words['attention_mask'].reshape(number_of_batches,
                                                                              batch_size,
                                                                              max_token_length).int()

    complexities = complexities.reshape(number_of_batches, batch_size, 1).half()

    print('Tokenization successful')

    return (encoded_sentences['input_ids'], encoded_sentences['token_type_ids'], encoded_sentences['attention_mask'],
            encoded_words['input_ids'], encoded_words['token_type_ids'], encoded_words['attention_mask']), complexities


def get_data_length(batch_size: int, data_path: str):
    """
    Finds the length of the data
    :param batch_size: The batch size to be used
    :param data_path: Training data
    :return: Data length as an integer
    """

    data = open(data_path, 'r', encoding='utf8')
    length = (len(data.read().splitlines()) - 1)
    number_of_batches = length // batch_size
    data_length = number_of_batches * batch_size

    print(f'Length of training data: {data_length}')
    return data_length
