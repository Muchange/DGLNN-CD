import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torch.autograd import Variable
import config
import numpy as np

class EnetGnn(nn.Module):
    def __init__(self, mlp_num_layers,use_gpu):
        super().__init__()
        self.g_rnn_layers = nn.ModuleList([nn.Linear(3,3) for l in range(mlp_num_layers)])
        self.g_rnn_actfs = nn.ModuleList([nn.PReLU() for l in range(mlp_num_layers)])
        self.q_rnn_layer = nn.Linear(6, 3)
        self.q_rnn_actf = nn.ReLU()
        self.use_gpu = use_gpu

    def get_knn_indices(self, batch_mat, k):
        r = torch.bmm(batch_mat, batch_mat.permute(0, 2, 1)) 
        N = r.size()[0]
        HW = r.size()[1]
        if self.use_gpu:
            batch_indices = torch.zeros((N, HW, k)).cuda()
        else:
            batch_indices = torch.zeros((N, HW, k))
        for idx, val in enumerate(r):
            diag = val.diag().unsqueeze(0)
            diag = diag.expand_as(val)
            D = (diag + diag.t() - 2 * val).sqrt()
            topk, indices = torch.topk(D, k=k, largest=False)
            batch_indices[idx] = indices.data
        return batch_indices

    def forward(self, cnn_encoder_output, gnn_iterations, k, proj_3d):
        # extract for convenience
        N = cnn_encoder_output.size()[0]
        C = cnn_encoder_output.size()[1]
        H = cnn_encoder_output.size()[2]
        W = cnn_encoder_output.size()[3]
        K = k
        knn = self.get_knn_indices(proj_3d, k=K)
        knn = knn.view(N, H*W*K).long()  # N HWK
        # prepare CNN encoded features for RNN
        h = cnn_encoder_output 
        # N C H W
        h = h.permute(0, 2, 3, 1).contiguous()  # N H W C
        h = h.view(N, (H * W), C)  # N HW C
        # aggregate and iterate messages in m, keep original CNN features h for later
        m = h.clone()  # N HW C
        # loop over timestamps to unroll
        for i in range(gnn_iterations):
            # do this for every  samplein batch, not nice, but I don't know how to use index_select batchwise
            for n in range(N):
                # N is bacth_size
                # fetch features from nearest neighbors
                neighbor_features = torch.index_select(h[n], 0, Variable(knn[n])).view(H*W, K, C)  # H*W K C
                #neighbor_features.shape-------------256,7,128
                # run neighbor features through MLP g and activation function
                for idx, g_layer in enumerate(self.g_rnn_layers):
                    neighbor_features = self.g_rnn_actfs[idx](g_layer(neighbor_features))  # HW K C
                # average over activated neighbors
                m[n] = torch.mean(neighbor_features, dim=1)  # HW C
            # concatenate current state with messages
            concat = torch.cat((h, m), 2)  # N HW 2C
            # get new features by running MLP q and activation function
            h = self.q_rnn_actf(self.q_rnn_layer(concat))  # N HW C
            knn = self.get_knn_indices(h, k=K)
            knn = knn.view(N, H*W*K).long()
        h = h.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()  # N C H W
        return h

class Model(nn.Module):

    def __init__(self, nclasses, mlp_num_layers,use_gpu):
        super().__init__()
        self.gnn = EnetGnn(mlp_num_layers,use_gpu)
        self.fc1 = nn.Linear(147,128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,2)

    def forward(self, input, gnn_iterations, k, proj_3d, use_gnn, only_encode=False):

        if use_gnn:
            x = self.gnn.forward(input, gnn_iterations, k, proj_3d)

        x = x.view(-1,147)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.relu1(x)
        x = self.fc4(x)
        return x