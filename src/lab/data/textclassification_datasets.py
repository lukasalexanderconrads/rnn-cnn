from torchtext.datasets import *
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import itertools
from tqdm import tqdm

class TextClassificationDataset(Dataset):
    """
    basis for all text classification data sets
    """
    def __init__(self, data, vocab_size):
        super(TextClassificationDataset, self).__init__()

        input, target, length = data

        self.input = input
        self.target = target
        self.length = length

        self.data_dim = self.input.size(1)
        self.n_classes = int(max(self.target) + 1)
        self.vocab_size = vocab_size

    @staticmethod
    def _get_data():
        raise NotImplementedError('_get_data() is not implemented')

    def __getitem__(self, item):
        return {'input': self.input[item],
                'target': self.target[item],
                'length': self.length[item]}

    def __len__(self):
        return self.input.size(0)

class RTETextClassificationDataset(TextClassificationDataset):
    """
    RTE text classification data set
    """

    def __init__(self, data, vocab_size):
        super(TextClassificationDataset, self).__init__()

        input1, input2, target, length1, length2 = data

        self.input1 = input1
        self.input2 = input2
        self.target = target
        self.length1 = length1
        self.length2 = length2

        self.data_dim = self.input1.size(1)
        self.n_classes = int(max(self.target) + 1)
        self.vocab_size = vocab_size

    def __getitem__(self, item):
        return {'input1': self.input1[item],
                'input2': self.input2[item],
                'target': self.target[item],
                'length1': self.length1[item],
                'length2': self.length2[item]}

    def __len__(self):
        return self.input1.size(0)


class YelpDataset:
    def __init__(self, **kwargs):
        self.path = kwargs.get('path', './data')
        self.tokenizer = get_tokenizer('basic_english')

    def get_data(self):
        train_iter = YelpReviewFull(root=self.path, split='train')
        test_iter = YelpReviewFull(root=self.path, split='test')

        self.vocab = build_vocab_from_iterator(self._yield_tokens(train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])


        train_set = TextClassificationDataset(self._setup_dataset(train_iter), len(self.vocab))
        test_set = TextClassificationDataset(self._setup_dataset(test_iter), len(self.vocab))

        train_set, valid_set = random_split(train_set, [len(train_set) - len(test_set), len(test_set)])

        return train_set, valid_set, test_set

    def _yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def _setup_dataset(self, data_iter):
        input = []
        target = []
        length = []
        print('tokenizing data...')
        for label, text in tqdm(data_iter):

            target.append(label - 1)
            input_seq = torch.tensor(self.vocab(self.tokenizer(text)))
            input.append(input_seq)
            length.append(len(input_seq))

        input = pad_sequence(input, batch_first=True, padding_value=len(self.vocab))

        input = torch.tensor(input, dtype=torch.int64)
        target = torch.tensor(target, dtype=torch.int64)
        length = torch.tensor(length, dtype=torch.int64)
        return input, target, length

class YahooDataset:
    def __init__(self, **kwargs):
        self.path = kwargs.get('path', './data')
        self.tokenizer = get_tokenizer('basic_english')
        self.max_length = kwargs.get('max_length', 400)

    def get_data(self):
        train_iter = YahooAnswers(root=self.path, split='train')
        test_iter = YahooAnswers(root=self.path, split='test')

        self.vocab = build_vocab_from_iterator(self._yield_tokens(train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])


        train_set = TextClassificationDataset(self._setup_dataset(train_iter), len(self.vocab))
        test_set = TextClassificationDataset(self._setup_dataset(test_iter), len(self.vocab))

        train_set, valid_set = random_split(train_set, [len(train_set) - len(test_set), len(test_set)])

        return train_set, valid_set, test_set

    def _yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def _setup_dataset(self, data_iter):
        input = []
        target = []
        length = []
        print('tokenizing data...')
        for label, text in tqdm(data_iter):

            input_seq = torch.tensor(self.vocab(self.tokenizer(text)))
            if len(input_seq) > self.max_length:
                continue
            target.append(label - 1)
            input.append(input_seq)
            length.append(len(input_seq))

        input = pad_sequence(input, batch_first=True, padding_value=len(self.vocab))

        input = torch.tensor(input, dtype=torch.int64)
        target = torch.tensor(target, dtype=torch.int64)
        length = torch.tensor(length, dtype=torch.int64)
        return input, target, length

class DBDataset:
    def __init__(self, **kwargs):
        self.path = kwargs.get('path', './data')
        self.tokenizer = get_tokenizer('basic_english')
        self.max_length = kwargs.get('max_length', 400)

    def get_data(self):
        train_iter = DBpedia(root=self.path, split='train')
        test_iter = DBpedia(root=self.path, split='test')

        self.vocab = build_vocab_from_iterator(self._yield_tokens(train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])


        train_set = TextClassificationDataset(self._setup_dataset(train_iter), len(self.vocab))
        test_set = TextClassificationDataset(self._setup_dataset(test_iter), len(self.vocab))

        train_set, valid_set = random_split(train_set, [len(train_set) - len(test_set), len(test_set)])

        return train_set, valid_set, test_set

    def _yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def _setup_dataset(self, data_iter):
        input = []
        target = []
        length = []
        print('tokenizing data...')
        for label, text in tqdm(data_iter):

            input_seq = torch.tensor(self.vocab(self.tokenizer(text)))
            if len(input_seq) > self.max_length:
                continue
            target.append(label - 1)
            input.append(input_seq)
            length.append(len(input_seq))

        input = pad_sequence(input, batch_first=True, padding_value=len(self.vocab))

        input = torch.tensor(input, dtype=torch.int64)
        target = torch.tensor(target, dtype=torch.int64)
        length = torch.tensor(length, dtype=torch.int64)
        return input, target, length

class QNLIDataset:
    def __init__(self, **kwargs):
        self.path = kwargs.get('path', './data')
        self.tokenizer = get_tokenizer('basic_english')
        self.max_length = kwargs.get('max_length', 400)
        self.max_vocab_size = kwargs.get('max_vocab_size', 20000)

    def get_data(self):
        train_iter = QNLI(root=self.path, split='train')
        valid_iter = QNLI(root=self.path, split='dev')

        self.vocab = build_vocab_from_iterator(self._yield_tokens(train_iter), specials=["<unk>"],
                                               max_tokens=self.max_vocab_size)
        self.vocab.set_default_index(self.vocab["<unk>"])

        train_set = RTETextClassificationDataset(self._setup_dataset(train_iter), len(self.vocab))
        valid_set = RTETextClassificationDataset(self._setup_dataset(valid_iter), len(self.vocab))

        train_set, test_set = random_split(train_set, [len(train_set) - len(valid_set), len(valid_set)])

        return train_set, valid_set, test_set

    def _yield_tokens(self, data_iter):
        for _, text1, text2 in data_iter:
            yield self.tokenizer(text1 + ' ' + text2)

    def _setup_dataset(self, data_iter):
        input1 = []
        input2 = []
        target = []
        length1 = []
        length2 = []
        print('tokenizing data...')
        for label, text1, text2 in tqdm(data_iter):

            input_seq1 = torch.tensor(self.vocab(self.tokenizer(text1)))
            input_seq2 = torch.tensor(self.vocab(self.tokenizer(text2)))
            if max(len(input_seq1), len(input_seq2)) > self.max_length:
                continue
            target.append(label)
            input1.append(input_seq1)
            input2.append(input_seq2)
            length1.append(len(input_seq1))
            length2.append(len(input_seq2))

        input1 = pad_sequence(input1, batch_first=True, padding_value=len(self.vocab))
        input2 = pad_sequence(input2, batch_first=True, padding_value=len(self.vocab))

        input1 = torch.tensor(input1, dtype=torch.int64)
        input2 = torch.tensor(input2, dtype=torch.int64)
        target = torch.tensor(target, dtype=torch.int64)
        length1 = torch.tensor(length1, dtype=torch.int64)
        length2 = torch.tensor(length2, dtype=torch.int64)
        return input1, input2, target, length1, length2



if __name__ == '__main__':

    ds = YahooDataset(path='/home/iai/user/conrads/rnn-cnn/data/RTE')