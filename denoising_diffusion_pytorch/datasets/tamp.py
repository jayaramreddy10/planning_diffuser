import collections
import numpy as np
import pdb

import torch
from torch.utils.data import Dataset
from glob import glob


def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)


class TampDataset(Dataset):

    conditions = [
        # ([], 1), ## none
        ([0], 1), ## first
        ([-1], 1), ## last
        ([0,-1], 1), ## first and last
    ]

    def __init__(self, H, max_path_length=1000, max_n_episodes=4000):
        dataset = "/data/vision/billf/scratch/yilundu/pddlstream/output_5/*.npy"
        datasets = sorted(glob(dataset))
        obs_dim = 63

        conditions_k, conditions_p = zip(*self.conditions)

        self.conditions_k = np.array(conditions_k, dtype=np.object)
        self.conditions_p = np.array(conditions_p) / sum(conditions_p)

        qstates = np.zeros((max_n_episodes, max_path_length, obs_dim))
        path_lengths = np.zeros(max_n_episodes, dtype=np.int)

        for i, dataset in enumerate(datasets):
            qstate = np.load(dataset)
            print(qstate.max(), qstate.min())
            # qstate[np.isnan(qstate)] = 0.0
            path_length = len(qstate)

            if path_length > max_path_length:
                qstates[i, :max_path_length] = qstate[:max_path_length]
                path_length = max_path_length
            else:
                qstates[i, :path_length] = qstate
            path_lengths[i] = path_length
        qstates = qstates[:i+1]
        path_lengths = path_lengths[:i+1]

        ## make indices
        print('Making indices')
        indices = []
        for i, path_length in enumerate(path_lengths):
            for start in range(path_length - H + 1):
                end = start + H
                indices.append((i, start, end))
        indices = np.array(indices)

        self.obs_dim = obs_dim
        self.qstates = qstates
        self.path_lengths = path_lengths
        self.indices = indices

        self.normalize()

        print(f'[ TampDataset ] qstates: {qstates.shape}')

    def normalize(self):
        '''
            normalizes to [-1, 1]
        '''
        # dataset = self.env.get_dataset()
        # X = np.concatenate([dataset['infos/qpos'], dataset['infos/qvel']], axis=-1)
        mins = self.mins = self.qstates.min(axis=0).min(axis=0)
        maxs = self.maxs = self.qstates.max(axis=0).max(axis=0)
        ## [ 0, 1 ]
        self.qstates = (self.qstates - mins) / (maxs - mins + 1e-5)
        ## [ -1, 1 ]
        self.qstates = self.qstates * 2 - 1

    def unnormalize(self, x):
        '''
            x : [ 0, 1 ]
        '''
        assert x.max() <= 1 and x.min() >= 0, f'x range: ({x.min():.4f}, {x.max():.4f})'
        mins = to_tensor(self.mins, dtype=x.dtype, device=x.device)
        maxs = to_tensor(self.maxs, dtype=x.dtype, device=x.device)
        return x * (maxs - mins) + mins

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-7):
        path_ind, start, end = self.indices[idx]
        qstates = self.qstates[path_ind, start:end]
        assert qstates.max() <= 1.0 + eps and qstates.min() >= -1.0 - eps, f'qstates range: ({qstates.min():.4f}, {qstates.max():.4f})'

        cond = np.random.choice(self.conditions_k, p=self.conditions_p)
        qstates = to_tensor(qstates[None])
        mask = torch.zeros_like(qstates)
        for t in cond:
            mask[:, t] = 1

        return qstates, mask


class KukaDataset(Dataset):

    conditions = [
        # ([], 1), ## none
        ([0], 1), ## first
        ([-1], 1), ## last
        ([0,-1], 1), ## first and last
    ]

    def __init__(self, H, max_path_length=300, max_n_episodes=15600):
        dataset = "kuka_dataset/*.npy"
        datasets = sorted(glob(dataset))     #11649 trajectories (.npy files)
        obs_dim = 39

        conditions_k, conditions_p = zip(*self.conditions)   #([0], [-1], [0, -1]),      (1, 1, 1)

        self.conditions_k = np.array(conditions_k, dtype=np.object)    #array([list([0]), list([-1]), list([0, -1])], dtype=object)
        self.conditions_p = np.array(conditions_p) / sum(conditions_p)   #array([0.33333333, 0.33333333, 0.33333333])

        qstates = np.zeros((max_n_episodes, max_path_length, obs_dim))   #(15600, 300, 39)
        path_lengths = np.zeros(max_n_episodes, dtype=np.int)    #(size: (15600, ))

        for i, dataset in enumerate(datasets):
            qstate = np.load(dataset)   #say (369, 39)   
        # (The first axis of the dataset) The 369 is timestep t
        # (The second axis of the dataset) 7 variables from the 39 dimension are the joint variables of the robot. The rest of the parameters (32 parameters) is the position as well as the orientation of the 4 cubes ( 3 pos and 4 rot for each cube)

        # data has the trajectory points of each cube as well as the joint values for the robot in each timestep.
            qstate = qstate[::2]   #we  get every second item from above array 
            print(qstate.max(), qstate.min())
            # qstate[np.isnan(qstate)] = 0.0
            path_length = len(qstate)

            if path_length > max_path_length:
                qstates[i, :max_path_length] = qstate[:max_path_length]
                path_length = max_path_length
            else:
                qstates[i, :path_length] = qstate
            path_lengths[i] = path_length
        qstates = qstates[:i+1]   #(11649, 300, 39)
        path_lengths = path_lengths[:i+1]    #size: (11649, )

        ## make indices
        print('Making indices')
        indices = []
        for i, path_length in enumerate(path_lengths):   #for every traj/path i
            for start in range(path_length - H + 1):
                end = start + H
                indices.append((i, start, end))   #start, end are seperated by H. (0, H), (1, H+1) .... (path_lenghts[i] - H, path_lenghts[i])
        indices = np.array(indices)   #shape: (716930, 3)

        self.obs_dim = obs_dim  #39
        self.qstates = qstates
        self.path_lengths = path_lengths
        self.indices = indices    #empty numpy array

        self.normalize()     #normalizes qstates to [-1, 1]

        print(f'[ TampDataset ] qstates: {qstates.shape}')

    def normalize(self):
        '''
            normalizes to [-1, 1]
        '''
        # dataset = self.env.get_dataset()
        # X = np.concatenate([dataset['infos/qpos'], dataset['infos/qvel']], axis=-1)
        mins = self.mins = self.qstates.min(axis=0).min(axis=0)
        maxs = self.maxs = self.qstates.max(axis=0).max(axis=0)
        ## [ 0, 1 ]
        self.qstates = (self.qstates - mins) / (maxs - mins + 1e-5)
        ## [ -1, 1 ]
        self.qstates = self.qstates * 2 - 1

    def unnormalize(self, x):
        '''
            x : [ 0, 1 ]
        '''
        assert x.max() <= 1 and x.min() >= 0, f'x range: ({x.min():.4f}, {x.max():.4f})'
        mins = to_tensor(self.mins, dtype=x.dtype, device=x.device)
        maxs = to_tensor(self.maxs, dtype=x.dtype, device=x.device)
        return x * (maxs - mins) + mins

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-7):
        path_ind, start, end = self.indices[idx]
        qstates = self.qstates[path_ind, start:end]   #(128, 39)
        assert qstates.max() <= 1.0 + eps and qstates.min() >= -1.0 - eps, f'qstates range: ({qstates.min():.4f}, {qstates.max():.4f})'

        cond = np.random.choice(self.conditions_k, p=self.conditions_p)  #randomly selects element from self.conditions
        qstates = to_tensor(qstates)
        # mask = qstates[-1]
        # for t in cond:
        #     mask[:, t] = 1
        mask = torch.zeros_like(qstates[..., -1])   #size: 128
        for t in cond:    #either ([0], [-1], [0, -1])
            mask[t] = 1

        return qstates, mask


class KukaDatasetReward(Dataset):

    conditions = [
        # ([], 1), ## none
        ([0], 1), ## first
        ([-1], 1), ## last
        ([0,-1], 1), ## first and last
    ]

    def __init__(self, H, max_path_length=1000, max_n_episodes=12000):
        dataset = "kuka_dataset/*.npy"
        datasets = sorted(glob(dataset))
        obs_dim = 39

        conditions_k, conditions_p = zip(*self.conditions)

        self.conditions_k = np.array(conditions_k, dtype=np.object)
        self.conditions_p = np.array(conditions_p) / sum(conditions_p)

        qstates = np.zeros((max_n_episodes, max_path_length, obs_dim))
        path_lengths = np.zeros(max_n_episodes, dtype=np.int)

        for i, dataset in enumerate(datasets):
            qstate = np.load(dataset)
            qstate = qstate[::2]
            print(qstate.max(), qstate.min())
            # qstate[np.isnan(qstate)] = 0.0
            path_length = len(qstate)

            if path_length > max_path_length:
                qstates[i, :max_path_length] = qstate[:max_path_length]
                path_length = max_path_length
            else:
                qstates[i, :path_length] = qstate
            path_lengths[i] = path_length
        qstates = qstates[:i+1]
        path_lengths = path_lengths[:i+1]

        ## make indices
        print('Making indices')
        indices = []
        for i, path_length in enumerate(path_lengths):
            for start in range(path_length - H + 1):
                end = start + H
                indices.append((i, start, end))
        indices = np.array(indices)

        self.obs_dim = obs_dim

        positions = []
        for i in range(4):
            pos = qstates[:, :, 7+i*8:10+i*8]
            positions.append(pos)

        labels = []

        for i in range(4):
            for j in range(4):
                if i == j:
                    continue

                pos_i = positions[i]
                pos_j = positions[j]

                pos_stack = np.linalg.norm(pos_i[..., :2] - pos_j[..., :2], axis=-1) < 0.1
                height_stack = pos_i[..., 2] > pos_j[..., 2]

                stack = pos_stack & height_stack
                labels.append(stack)

        self.labels = np.stack(labels, axis=-1)

        self.qstates = qstates
        self.path_lengths = path_lengths
        self.indices = indices

        self.normalize()

        print(f'[ TampDataset ] qstates: {qstates.shape}')

    def normalize(self):
        '''
            normalizes to [-1, 1]
        '''
        # dataset = self.env.get_dataset()
        # X = np.concatenate([dataset['infos/qpos'], dataset['infos/qvel']], axis=-1)
        mins = self.mins = self.qstates.min(axis=0).min(axis=0)
        maxs = self.maxs = self.qstates.max(axis=0).max(axis=0)
        ## [ 0, 1 ]
        self.qstates = (self.qstates - mins) / (maxs - mins + 1e-5)
        ## [ -1, 1 ]
        self.qstates = self.qstates * 2 - 1

    def unnormalize(self, x):
        '''
            x : [ 0, 1 ]
        '''
        assert x.max() <= 1 and x.min() >= 0, f'x range: ({x.min():.4f}, {x.max():.4f})'
        mins = to_tensor(self.mins, dtype=x.dtype, device=x.device)
        maxs = to_tensor(self.maxs, dtype=x.dtype, device=x.device)
        return x * (maxs - mins) + mins

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-7):
        path_ind, start, end = self.indices[idx]
        qstates = self.qstates[path_ind, start:end]
        assert qstates.max() <= 1.0 + eps and qstates.min() >= -1.0 - eps, f'qstates range: ({qstates.min():.4f}, {qstates.max():.4f})'

        target = self.labels[path_ind, start:end]

        return qstates, target
