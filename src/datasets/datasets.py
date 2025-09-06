import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import os
from os.path import join
import pickle as pkl
import requests

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import zipfile
import io


def feature_norm(self, features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


class Dataset(object):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset") -> None:
        self.adj_ = None
        self.features_ = None
        self.labels_ = None
        self.idx_train_ = None
        self.idx_val_ = None
        self.idx_test_ = None
        self.sens_ = None
        self.sens_idx_ = None
        self.is_normalize = is_normalize

        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.path_name = ""

    def download(self, url: str, filename: str):
        r = requests.get(url)
        assert r.status_code == 200
        open(os.path.join(self.root, self.path_name, filename), "wb").write(r.content)

    def download_zip(self, url: str):
        r = requests.get(url)
        assert r.status_code == 200
        foofile = zipfile.ZipFile(io.BytesIO(r.content))
        foofile.extractall(os.path.join(self.root, self.path_name))

    def adj(self, datatype: str = "torch.sparse"):
        # assert str(type(self.adj_)) == "<class 'torch.Tensor'>"
        if self.adj_ is None:
            return self.adj_
        if datatype == "torch.sparse":
            return self.adj_
        elif datatype == "scipy.sparse":
            return sp.coo_matrix(self.adj.to_dense())
        elif datatype == "np.array":
            return self.adj_.to_dense().numpy()
        else:
            raise ValueError(
                "datatype should be torch.sparse, tf.sparse, np.array, or scipy.sparse"
            )

    def features(self, datatype: str = "torch.tensor"):
        if self.is_normalize and self.features_ is not None:
            self.features_ = feature_norm(self, self.features_)

        if self.features is None:
            return self.features_
        if datatype == "torch.tensor":
            return self.features_
        elif datatype == "np.array":
            return self.features_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def labels(self, datatype: str = "torch.tensor"):
        if self.labels_ is None:
            return self.labels_
        if datatype == "torch.tensor":
            return self.labels_
        elif datatype == "np.array":
            return self.labels_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_val(self, datatype: str = "torch.tensor"):
        if self.idx_val_ is None:
            return self.idx_val_
        if datatype == "torch.tensor":
            return self.idx_val_
        elif datatype == "np.array":
            return self.idx_val_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_train(self, datatype: str = "torch.tensor"):
        if self.idx_train_ is None:
            return self.idx_train_
        if datatype == "torch.tensor":
            return self.idx_train_
        elif datatype == "np.array":
            return self.idx_train_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_test(self, datatype: str = "torch.tensor"):
        if self.idx_test_ is None:
            return self.idx_test_
        if datatype == "torch.tensor":
            return self.idx_test_
        elif datatype == "np.array":
            return self.idx_test_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens(self, datatype: str = "torch.tensor"):
        if self.sens_ is None:
            return self.sens_
        if datatype == "torch.tensor":
            return self.sens_
        elif datatype == "np.array":
            return self.sens_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens_idx(self):
        if self.sens_idx_ is None:
            self.sens_idx_ = -1
        return self.sens_idx_


def mx_to_torch_sparse_tensor(sparse_mx, is_sparse=False, return_tensor_sparse=True):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not is_sparse:
        sparse_mx = sp.coo_matrix(sparse_mx)
    else:
        sparse_mx = sparse_mx.tocoo()
    if not return_tensor_sparse:
        return sparse_mx

    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class Facebook(Dataset):
    def __init__(
        self,
        path: str = "./dataset/facebook/",
        is_normalize: bool = False,
        root: str = "./dataset",
    ) -> None:
        super().__init__(is_normalize=is_normalize, root=root)
        self.path_name = "facebook"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(os.path.join(self.root, self.path_name, "107.edges")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/facebook/facebook/107.edges"
            filename = "107.edges"
            self.download(url, filename)
        if not os.path.exists(os.path.join(self.root, self.path_name, "107.feat")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/facebook/facebook/107.feat"
            filename = "107.feat"
            self.download(url, filename)
        if not os.path.exists(os.path.join(self.root, self.path_name, "107.featnames")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/facebook/facebook/107.featnames"
            filename = "107.featnames"
            self.download(url, filename)

        edges_file = open(os.path.join(self.root, self.path_name, "107.edges"))
        edges = []
        for line in edges_file:
            edges.append([int(one) for one in line.strip("\n").split(" ")])

        feat_file = open(os.path.join(self.root, self.path_name, "107.feat"))
        feats = []
        for line in feat_file:
            feats.append([int(one) for one in line.strip("\n").split(" ")])

        feat_name_file = open(os.path.join(self.root, self.path_name, "107.featnames"))
        feat_name = []
        for line in feat_name_file:
            feat_name.append(line.strip("\n").split(" "))
        names = {}
        for name in feat_name:
            if name[1] not in names:
                names[name[1]] = name[1]

        feats = np.array(feats)

        node_mapping = {}
        for j in range(feats.shape[0]):
            node_mapping[feats[j][0]] = j

        feats = feats[:, 1:]

        sens = feats[:, 264]
        labels = feats[:, 220]

        feats = np.concatenate([feats[:, :264], feats[:, 266:]], -1)

        feats = np.concatenate([feats[:, :220], feats[:, 221:]], -1)

        edges = np.array(edges)
        # edges=torch.tensor(edges)
        # edges=torch.stack([torch.tensor(one) for one in edges],0)

        node_num = feats.shape[0]
        adj = np.zeros([node_num, node_num])

        for j in range(edges.shape[0]):
            adj[node_mapping[edges[j][0]], node_mapping[edges[j][1]]] = 1

        idx_train = np.random.choice(
            list(range(node_num)), int(0.8 * node_num), replace=False
        )
        idx_val = list(set(list(range(node_num))) - set(idx_train))
        idx_test = np.random.choice(idx_val, len(idx_val) // 2, replace=False)
        idx_val = list(set(idx_val) - set(idx_test))

        self.features_ = torch.FloatTensor(feats)
        self.sens_ = torch.FloatTensor(sens)
        self.idx_train_ = torch.LongTensor(idx_train)
        self.idx_val_ = torch.LongTensor(idx_val)
        self.idx_test_ = torch.LongTensor(idx_test)
        self.labels_ = torch.LongTensor(labels)

        self.features_ = torch.cat([self.features_, self.sens_.unsqueeze(-1)], -1)
        self.adj_ = mx_to_torch_sparse_tensor(adj)
        self.sens_idx_ = -1


class Nba(Dataset):
    def __init__(
        self,
        dataset_name="nba",
        predict_attr_specify=None,
        return_tensor_sparse=True,
        is_normalize: bool = False,
        root: str = "./dataset",
    ):
        super().__init__(is_normalize=is_normalize, root=root)
        if dataset_name != "nba":
            if dataset_name == "pokec_z":
                dataset = "region_job"
            elif dataset_name == "pokec_n":
                dataset = "region_job_2"
            else:
                dataset = None
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path = "./dataset/pokec/"
            test_idx = False
        else:
            dataset = "nba"
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
            path = "./dataset/NBA"
            test_idx = True

        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
            sens,
            idx_sens_train,
        ) = self.load_pokec(
            dataset,
            sens_attr,
            predict_attr if predict_attr_specify == None else predict_attr_specify,
            path=path,
            label_number=label_number,
            sens_number=sens_number,
            seed=seed,
            test_idx=test_idx,
        )

        # adj=adj.todense()
        adj = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )
        labels[labels > 1] = 1

        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = -1

    def load_pokec(
        self,
        dataset,
        sens_attr,
        predict_attr,
        path="../dataset/pokec/",
        label_number=1000,
        sens_number=500,
        seed=19,
        test_idx=False,
    ):
        """Load data"""

        self.path_name = "nba"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(os.path.join(self.root, self.path_name, "nba.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/NBA/nba.csv"
            filename = "nba.csv"
            self.download(url, filename)
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "nba_relationship.txt")
        ):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/NBA/nba_relationship.txt"
            filename = "nba_relationship.txt"
            self.download(url, filename)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "nba.csv")
        )
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, "nba_relationship.txt"),
            dtype=np.int64,
        )

        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        import random

        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)) :]

        sens = idx_features_labels[sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


class Bail(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset"):
        super(Bail, self).__init__(is_normalize=is_normalize, root=root)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_bail("bail")

        node_num = features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = sens_idx

    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def load_bail(
        self,
        dataset,
        sens_attr="WHITE",
        predict_attr="RECID",
        path="./dataset/bail/",
        label_number=100,
    ):
        # print('Loading {} dataset from {}'.format(dataset, path))
        self.path_name = "bail"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))

        if not os.path.exists(os.path.join(self.root, self.path_name, "bail.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail.csv"
            file_name = "bail.csv"
            self.download(url, file_name)
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "bail_edges.txt")
        ):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail_edges.txt"
            file_name = "bail_edges.txt"
            self.download(url, file_name)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "{}.csv".format(dataset))
        )
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)

        # build relationship

        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random

        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 0


class Oklahoma(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset"):
        super(Oklahoma, self).__init__(is_normalize=is_normalize, root=root)
        dataset_name = "oklahoma"
        from scipy.io import loadmat

        if dataset_name == "oklahoma":
            dataset_name = "Oklahoma97"
        elif dataset_name == "unc28":
            dataset_name = "UNC28"
        self.path_name = "oklahoma"

        self.url = "https://drive.google.com/u/0/uc?id=1tNcxgtEQX3dtDKqwDMswJEvxpKBpov75&export=download"
        self.destination = os.path.join(self.root, self.path_name, "oklahoma.zip")
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(
            os.path.join(
                self.root, self.path_name, "Oklahoma97/{}_feat.pkl".format(dataset_name)
            )
        ):
            self.download_zip(self.url)
        if not os.path.exists(
            os.path.join(
                self.root,
                self.path_name,
                "Oklahoma97/{}_user_sen.pkl".format(dataset_name),
            )
        ):
            self.download_zip(self.url)
        if not os.path.exists(
            os.path.join(
                self.root,
                self.path_name,
                "Oklahoma97/{}_train_items.pkl".format(dataset_name),
            )
        ):
            self.download_zip(self.url)
        if not os.path.exists(
            os.path.join(
                self.root,
                self.path_name,
                "Oklahoma97/{}_test_set.pkl".format(dataset_name),
            )
        ):
            self.download_zip(self.url)

        feats = pkl.load(
            open(
                join(
                    self.root,
                    self.path_name,
                    "Oklahoma97/{}_feat.pkl".format(dataset_name),
                ),
                "rb",
            )
        )
        sens = pkl.load(
            open(
                join(
                    self.root,
                    self.path_name,
                    "Oklahoma97/{}_user_sen.pkl".format(dataset_name),
                ),
                "rb",
            )
        )
        sens = [sens[idx] for idx in range(feats.shape[0])]
        train_items = pkl.load(
            open(
                join(
                    self.root,
                    self.path_name,
                    "Oklahoma97/{}_train_items.pkl".format(dataset_name),
                ),
                "rb",
            )
        )
        test_items = pkl.load(
            open(
                join(
                    self.root,
                    self.path_name,
                    "Oklahoma97/{}_test_set.pkl".format(dataset_name),
                ),
                "rb",
            )
        )

        adj = np.zeros([feats.shape[0], feats.shape[0]])

        for item in [train_items, test_items]:
            for key, value in item.items():
                for one in value:
                    adj[key][one] = 1

        self.adj_ = mx_to_torch_sparse_tensor(adj)
        features = torch.FloatTensor(feats)
        sens = torch.FloatTensor(sens)
        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        train_items = train_items
        test_items = test_items

        self.features_ = features
        self.train_items_ = train_items
        self.test_items_ = test_items
        self.sens_ = sens
        self.sens_idx_ = -1

        # random train_idx
        self.idx_train_ = np.random.choice(
            list(train_items.keys()), int(len(features) * 0.8), replace=False
        )
        # random val and test
        self.idx_val_ = np.random.choice(
            list(set(train_items.keys()) - set(self.idx_train_)),
            int(len(features) * 0.1),
            replace=False,
        )
        self.idx_test_ = list(
            set(train_items.keys()) - set(self.idx_train_) - set(self.idx_val_)
        )

        # random label
        self.labels_ = np.zeros(len(features))
        self.pos_idx_ = np.random.choice(
            np.arange(len(features)), int(len(features) * 0.1), replace=False
        )
        self.labels_[self.pos_idx_] = 1

        # to torch
        self.labels_ = torch.tensor(self.labels_, dtype=torch.long)
        self.idx_train_ = torch.tensor(self.idx_train_, dtype=torch.long)
        self.idx_val_ = torch.tensor(self.idx_val_, dtype=torch.long)
        self.idx_test_ = torch.tensor(self.idx_test_, dtype=torch.long)

    def adj(self, datatype: str = "torch.sparse"):
        if self.adj_ is None:
            return self.adj_
        if datatype == "torch.sparse":
            return self.adj_
        elif datatype == "np.array":
            return self.adj_.to_dense().numpy()
        else:
            raise ValueError("datatype should be torch.sparse or np.array")

    def features(self, datatype: str = "torch.tensor"):
        if self.features_ is None:
            return self.features_
        if datatype == "torch.tensor":
            return self.features_
        elif datatype == "np.array":
            return self.features_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor or np.array")

    def train_items(self, datatype: str = "dict"):
        if self.train_items_ is None:
            return self.train_items_
        if datatype == "dict":
            return self.train_items_
        else:
            raise ValueError("datatype should be torch.tensor or np.array")

    def test_items(self, datatype: str = "dict"):
        if self.test_items_ is None:
            return self.test_items_
        if datatype == "dict":
            return self.test_items_
        else:
            raise ValueError("datatype should be dict")

    def sens(self, datatype: str = "torch.tensor"):
        if self.sens_ is None:
            return self.sens_
        if datatype == "torch.tensor":
            return self.sens_
        elif datatype == "np.array":
            return self.sens_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor or np.array")


class German(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset"):
        super(German, self).__init__(is_normalize=is_normalize, root=root)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_german("german")

        node_num = features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)

        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = sens_idx

    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def load_german(
        self,
        dataset,
        sens_attr="Gender",
        predict_attr="GoodCustomer",
        path="./dataset/german/",
        label_number=100,
    ):
        # print('Loading {} dataset from {}'.format(dataset, path))
        self.path_name = "german"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))

        if not os.path.exists(os.path.join(self.root, self.path_name, "german.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/german/german.csv"
            file_name = "german.csv"
            self.download(url, file_name)
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "german_edges.txt")
        ):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/german/german_edges.txt"
            file_name = "german_edges.txt"
            self.download(url, file_name)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "{}.csv".format(dataset))
        )
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("OtherLoansAtStore")
        header.remove("PurposeOfLoan")

        # Sensitive Attribute
        idx_features_labels["Gender"][idx_features_labels["Gender"] == "Female"] = 1
        idx_features_labels["Gender"][idx_features_labels["Gender"] == "Male"] = 0

        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))

        labels = torch.LongTensor(labels)

        import random

        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 0
