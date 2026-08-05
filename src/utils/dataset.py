import random
import torch

class GraphData(object):

    def __init__(self, data, feat_dim):
        super(GraphData, self).__init__()
        self.data = data
        self.feat_dim = feat_dim
        self.idx = list(range(len(data)))
        self.pos = 0

    def __reset__(self):
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.idx)

    def __len__(self):
        return len(self.data) // self.batch + (1 if len(self.data) % self.batch != 0 else 0)

    def __getitem__(self, idx):
        window_graphs = self.data[idx]

        A_list, feas_list, label_list = [], [], []
        for g in window_graphs:
            A = g.A
            feas = g.feas
            A_list.append(A)
            feas_list.append(feas)
            label_list.append(int(g.label))

        As_seq = torch.stack(A_list, dim=0)
        hs_seq = torch.stack(feas_list, dim=0)
        ys_seq = torch.LongTensor(label_list)

        return As_seq, hs_seq, ys_seq

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.data):
            self.__reset__()
            raise StopIteration

        cur_idx = self.idx[self.pos: self.pos + self.batch]
        samples = [self.__getitem__(idx) for idx in cur_idx]
        self.pos += len(cur_idx)

        As_seqs, hs_seqs, ys_seqs = map(list, zip(*samples))
        L0 = ys_seqs[0].shape[0]
        assert all(y.shape[0] == L0 for y in ys_seqs), "All windows in a batch must share the same L"

        ys_seqs = torch.stack(ys_seqs, dim=0)
        As_seqs = torch.stack(As_seqs, dim=0)
        hs_seqs = torch.stack(hs_seqs, dim=0)

        return ys_seqs.shape[0], As_seqs, hs_seqs, torch.LongTensor(ys_seqs)

    def loader(self, batch, shuffle):
        self.batch = batch
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.idx)
        return self
