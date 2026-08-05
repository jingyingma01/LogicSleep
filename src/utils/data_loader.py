import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class G_data(object):
    def __init__(self, num_class, feat_dim, g_list, train_g_list,
                 test_g_list, target_g_list, args):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.seed = args.seed
        self.g_list = g_list
        self.fold_idx = args.fold
        self.train_gs = train_g_list
        self.test_gs = test_g_list
        self.target_gs = target_g_list

class FileLoader(object):
    def __init__(self, args):
        self.args = args
        self.delta_t = args.delta_t
        self.delta_p = args.delta_p
        self.num_class = args.num_class
        self.feat_dim = args.feat_dim
        self.num_patch = args.num_patch
        self.num_node = args.num_node

    def line_genor(self, lines):
        for line in lines:
            yield line

    def gen_graph(self, psg, label):
        g = nx.Graph()
        node_features = []
        for j in range(len(psg)):
            g.add_node(j, features = psg[j])
            node_features.append(psg[j])
        similarity_matrix = cosine_similarity(node_features)
        for i in range(similarity_matrix.shape[0]):
            for j in range(i + 1, similarity_matrix.shape[1]):
                if similarity_matrix[i, j] > 0.5:
                    g.add_edge(i, j, weight=similarity_matrix[i, j])
        g.label = label
        return g

    def add_context_data(self, data, win_len = 16, overlap = 2):
        n = len(data)
        step = win_len - overlap
        starts = list(range(0, n - win_len + 1, step))
        windows = []
        for s in starts:
            chunk = data[s: s + win_len]
            win = np.stack(chunk, axis=0)
            windows.append(win)
        X_win = np.stack(windows, axis = 0)
        return X_win

    def add_context_label(self, label, win_len = 16, overlap = 2):
        n = len(label)
        step = win_len - overlap
        starts = np.arange(0, n - win_len + 1, step, dtype=int)
        y_windows = [np.asarray(label[s: s + win_len], dtype=np.int64) for s in starts]
        y_win = np.stack(y_windows, axis=0)
        return y_win

    def process_g(self, g):
        node_features = []
        num_node = self.num_node
        for j in range(self.num_patch * num_node):
            node_features.append(g.nodes[j]['features'])
        node_features = np.array(node_features)
        g.feas = torch.tensor(node_features)
        A = torch.FloatTensor(nx.to_numpy_array(g))
        g.A = A + torch.eye(g.number_of_nodes())
        time_matrix = np.zeros((self.num_patch * num_node, self.num_patch * num_node))
        for i in range(self.num_patch):
            for j in range(self.num_patch):
                time_matrix[i * num_node:(i + 1) * num_node,
                j * num_node:(j + 1) * num_node] = self.delta_t** abs(i - j)
        position_matrix = np.zeros((self.num_patch * num_node, self.num_patch * num_node))
        for i in range(0, self.num_patch * num_node):
            for j in range(0, self.num_patch * num_node):
                if i % num_node == j % num_node:
                    position_matrix[i, j] = 1
                else:
                    position_matrix[i, j] = self.delta_p
        g.A = g.A * time_matrix * position_matrix
        return g

    def build_window_graphs(self, window_data, window_labels):
        L = window_data.shape[0]
        graphs = []
        for t in range(L):
            x_epoch = window_data[t]
            y_epoch = window_labels[t]
            g = self.gen_graph(x_epoch, y_epoch)
            g = self.process_g(g)
            graphs.append(g)
        return graphs

    def load_data(self, if_tta):
        args = self.args
        print('loading data ...')
        data = np.load('/data_path/%s.npz' % (args.data), allow_pickle=True)
        fold = args.fold
        test_datas = []
        test_labels = []
        target_datas = []
        train_datas = []
        train_labels = []
        target_labels = []
        new_g_list = []

        train_g_list = []
        test_g_list = []
        target_g_list = []


        if if_tta:
            fold_data = data[str(fold)].item()
            target_datas.extend(self.add_context_data(fold_data['datas'], self.args.window, self.args.overlap))
            target_labels.extend(self.add_context_label(fold_data['labels'], self.args.window, self.args.overlap))
            for i in tqdm(range(len(target_datas)), desc="Create target graph", unit='Graph'):
                g = self.build_window_graphs(target_datas[i], target_labels[i])
                target_g_list.append(g)
            return G_data(self.num_class, self.feat_dim, new_g_list,
                          train_g_list, target_g_list, test_g_list, self.args)
        else:
            if int(fold) == 9:
                test_idx = 0
            else:
                test_idx = int(fold) + 1

            for key in data.files:
                fold_data = data[key].item()
                if int(key) % 10 == int(fold):
                    target_datas.extend(self.add_context_data(fold_data['datas'], self.args.window, self.args.overlap))
                    target_labels.extend(self.add_context_label(fold_data['labels'], self.args.window, self.args.overlap))
                elif int(key) % 10 == test_idx:
                    test_datas.extend(self.add_context_data(fold_data['datas'], self.args.window, self.args.overlap))
                    test_labels.extend(self.add_context_label(fold_data['labels'], self.args.window, self.args.overlap))
                else:
                    train_datas.extend(self.add_context_data(fold_data['datas'], self.args.window, self.args.overlap))
                    train_labels.extend(self.add_context_label(fold_data['labels'], self.args.window, self.args.overlap))

            for i in tqdm(range(len(train_datas)), desc="Create train graph", unit='Graph'):
                g = self.build_window_graphs(train_datas[i], train_labels[i])
                train_g_list.append(g)

            for i in tqdm(range(len(test_datas)), desc="Create test graph", unit='Graph'):
                g = self.build_window_graphs(test_datas[i], test_labels[i])
                test_g_list.append(g)

            for i in tqdm(range(len(target_datas)), desc="Create target graph", unit='Graph'):
                g = self.build_window_graphs(target_datas[i], target_labels[i])
                target_g_list.append(g)

            return G_data(self.num_class, self.feat_dim, new_g_list, train_g_list, test_g_list, target_g_list, self.args)