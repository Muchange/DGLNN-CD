import cv2
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import cd_data
from models_knn import Model
import config
from testresults import testResult
import argparse 
from tqdm import tqdm
import scipy.io as sio
import scipy
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image 
import warnings

def setup_seed(seed):
    torch.manual_seed(seed)                           
    torch.cuda.manual_seed_all(seed)           
    np.random.seed(seed)                       
    torch.backends.cudnn.deterministic = True

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('DGLNN')
    parser.add_argument('--num_epochs', default=100,type=int,help='Number of epoch')
    parser.add_argument('--batchsize', type=int, default=32,help='batch size in training')
    parser.add_argument('--test_batchsize', type=int, default=512,help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0',help='specify gpu device')
    return parser.parse_args()

setup_seed(42)

def main(args):
    print("Loading data...")
    dataset_tr = cd_data.trainDatasets(datasets='./GCN-data/train/7YA+YB+SB_train.npy',three_d = './GCN-data/train/7YA+YB+SB_train_point.npy',label='./GCN-data/train/7YA+YB+SB_train_Lab.npy')
    dataloader_tr = DataLoader(dataset_tr, batch_size=args.batchsize, shuffle=True,num_workers=config.workers_tr)
    dataset_va = cd_data.testDatasets(datasets='./GCN-data/test/7SA_test.npy',three_d = './GCN-data/test/7SA_test_point.npy')
    dataloader_va = DataLoader(dataset_va, batch_size=args.test_batchsize, shuffle=False,num_workers=config.workers_va)
    cv2.setNumThreads(config.workers_tr)
    print("Preparing model...")
    model = Model(config.nclasses, config.mlp_num_layers,config.use_gpu)
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    if config.use_gpu:
        model = nn.DataParallel(model,device_ids=[0,1])
        model.to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': config.gnn_initial_lr}])

    def test():
        global best_kappa
        model.eval()
        Label = []
        for batch_idx, data in tqdm(enumerate(dataloader_va),total = len(dataloader_va),smoothing=0.9):
            x = data[0]
            proj_3d = data[1]
            x = x.float()
            proj_3d = proj_3d.float()
            input = x.permute(0, 3, 1, 2).contiguous()
            input = input.type(torch.FloatTensor)

            if config.use_gpu:
                input = input.cuda()
                proj_3d = proj_3d.cuda()

            optimizer.zero_grad()
            output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, proj_3d=proj_3d, use_gnn=config.use_gnn)
            _, predicted = torch.max(output, 1)
            predicted.cpu().numpy()
            predicted.cuda()
            Label.extend(predicted)
        outputLabel = np.array(Label)
        outputImg = outputLabel.reshape((560,549),order = 'F')
        outputImg = Image.fromarray(np.uint8(outputImg*255.0))
        outputImg = outputImg.convert('L')
        refImg  = plt.imread('./ref_img/Sendai-A-ref.png')
        testResults = testResult(outputImg,refImg)
        FA, MA = testResults.FA_MA()
        refLabel = refImg.reshape(-1,)
        kappa = testResults.KappaCoef(FA, MA)
        
        if kappa >= best_kappa:
            outputImg.save('./Sendai-A-result.png') 
            best_kappa = kappa
            torch.save(model,'./SA_result')
            with open('SA_result.txt','a') as file_handle:  
                file_handle.write("epoch = {}\tFA = {}\tMA = {}\tKappa = {}\n".format(epoch,FA,MA,kappa)) 
                
        print('FA: {}\t MA: {}\t kappa: {:.4f}\n'.format(FA,MA,kappa))
        return

    print("Number of epochs: %d"%args.num_epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.num_epochs + 1):
        for batch_idx, data in  enumerate(dataloader_tr):
            x = data[0]
            target = data[1].long()
            proj_3d = data[2]
            x = x.float()
            proj_3d = proj_3d.float()
            input = x.permute(0, 3, 1, 2).contiguous()
            input = input.type(torch.FloatTensor)

            if config.use_gpu:
                input = input.cuda()
                proj_3d = proj_3d.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, proj_3d=proj_3d, use_gnn=config.use_gnn)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                    epoch, batch_idx * len(input), len(dataloader_tr.dataset),
                    100. * batch_idx / len(dataloader_tr), loss.data.item(),
                    optimizer.param_groups[0]['lr']))
        test()

if __name__ == '__main__':
    args = parse_args()
    best_kappa = 0
    main(args)