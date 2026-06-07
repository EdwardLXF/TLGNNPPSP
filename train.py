import os
import model
import load_datasets
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
#import itertools
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import itertools
# -*- coding: utf-8 -*-
import json
import random
from random import shuffle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter
from torch_geometric.data import Data, InMemoryDataset, Batch
#from utils import one_of_k_encoding, random_scaffold_split, seed_torch
from torch.nn.functional import one_hot


#import proplot as pplt
import rdkit
import sqlite3
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles
from rdkit.Chem.rdchem import HybridizationType as HT
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from matplotlib import colors 




from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import load_datasets
from torch_geometric.loader import DataLoader



import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool

rate=0.0
#dataset.num_node_features=torch.Tensor(15)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_channels = 64

class GNN(nn.Module):
    
    def __init__(self):
        # 初始化Pytorch父类
        super().__init__()
        
        #self.conv1=GCNConv(21, hidden_channels)
        self.conv1=GCNConv(15, hidden_channels)
        #print(dataset.num_node_features)
        self.conv2=GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)


    
    # 前向传播函数
    def forward(self, x, edge_index,batch):
        
        x=x.to(device)
        edge_index=edge_index.to(device)
        batch=batch.to(device)

        x=self.conv1(x, edge_index)
        x=x.relu()
        x=self.conv2(x, edge_index)
        x=x.relu()
        x=self.conv3(x, edge_index)
        x=x.relu()
        x=self.conv4(x, edge_index)
        x=x.relu()
        #print('x',x)

        # 全局池化
        x = global_mean_pool(x, batch)  # [x, batch]
        #print('x_pool',x)

        
        return x










def get_traindata():

    f_read=open('neg_sample.pkl','rb')
    dict_n=pickle.load(f_read)
    print(dict_n)


    dataset_source = []
    dataset_target = []

    '''
    df_source_1 = pd.DataFrame(pd.read_csv('solvent_HSP_hz0828.csv'))
    smiles_list_v=df_source_1['SMILES'].tolist()
    
    smiles_list=df_source_1['SMILES'].tolist()
    labels_D=df_source_1['D'].tolist()
    labels_P=df_source_1['P'].tolist()
    labels_H=df_source_1['H'].tolist()

    #D拟合的好？

    labels=[]

    smiles_list_v=[]
    labels_D_v=[]
    labels_P_v=[]
    labels_H_v=[]

    for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,smiles_list)):
        if not is_valid_smiles(smi): continue

        smiles_list_v.append(smi)
        labels_D_v.append(D)
        labels_P_v.append(P)
        labels_H_v.append(H)

        #labels.append(D)

    labels=[]
    for D,P,H in tqdm(zip(labels_D_v, labels_P_v,labels_H_v)):
        labels.append([D,P,H])



    for smi, label in tqdm(zip(smiles_list_v, labels)):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset_source.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=torch.tensor([label])))
    '''
    df_source_2 = pd.DataFrame(pd.read_csv('solvent_HSP_HSPiP_hz1010.csv'))
    smiles_list_v=df_source_2['SMILES'].tolist()
    
    smiles_list=df_source_2['SMILES'].tolist()
    labels_D=df_source_2['D'].tolist()
    labels_P=df_source_2['P'].tolist()
    labels_H=df_source_2['H'].tolist()

    #D拟合的好？

    labels=[]

    smiles_list_v=[]
    labels_D_v=[]
    labels_P_v=[]
    labels_H_v=[]

    for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,smiles_list)):
        if not is_valid_smiles(smi): continue

        smiles_list_v.append(smi)
        labels_D_v.append(D)
        labels_P_v.append(P)
        labels_H_v.append(H)

        #labels.append(D)
    print('vild1010:'+str(len(smiles_list_v)))
    print('========================================================')
    labels=[]
    for D,P,H in tqdm(zip(labels_D_v, labels_P_v,labels_H_v)):
        labels.append([D,P,H])



    for smi, label in tqdm(zip(smiles_list_v, labels)):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset_source.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=torch.tensor([label])))



    polymer_smi_vial=[]
    '''
    polymer_smi_vial=[]
    df_target = pd.DataFrame(pd.read_csv('polyinfo_HSP_hz0827.csv'))

    smiles_list=df_target['SMILES'].tolist()
    labels_D=df_target['D'].tolist()
    labels_P=df_target['P'].tolist()
    labels_H=df_target['H'].tolist()

    #D拟合的好？

    labels=[]

    smiles_list_v=[]
    labels_D_v=[]
    labels_P_v=[]
    labels_H_v=[]

    for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,smiles_list)):
        if not is_valid_smiles(smi): continue

        smiles_list_v.append(smi)
        polymer_smi_vial.append(smi)
        labels_D_v.append(D)
        labels_P_v.append(P)
        labels_H_v.append(H)

        #labels.append(D)

    
    labels=[]
    for D,P,H in tqdm(zip(labels_D_v, labels_P_v,labels_H_v)):
        labels.append([D,P,H])
    #labels=[0,1]
    




    for smi, label in tqdm(zip(smiles_list_v, labels)):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset_target.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=torch.tensor([label])))

    '''
    df_target = pd.DataFrame(pd.read_csv('polymer_HSP_HSPiP_hz1010.csv'))

    smiles_list=df_target['SMILES'].tolist()
    labels_D=df_target['D'].tolist()
    labels_P=df_target['P'].tolist()
    labels_H=df_target['H'].tolist()


    new_smiles_list=[]
    for i in smiles_list:
        i_new=i.replace('X','')
        new_smiles_list.append(i_new)

    labels=[]

    smiles_list_v=[]
    labels_D_v=[]
    labels_P_v=[]
    labels_H_v=[]

    for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,new_smiles_list)):
        if not is_valid_smiles(smi): continue

        smiles_list_v.append(smi)
        polymer_smi_vial.append(smi)
        labels_D_v.append(D)
        labels_P_v.append(P)
        labels_H_v.append(H)

        #labels.append(D)

    
    labels=[]
    for D,P,H in tqdm(zip(labels_D_v, labels_P_v,labels_H_v)):
        labels.append([D,P,H])
    #labels=[0,1]
    




    for smi, label in tqdm(zip(smiles_list_v, labels)):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset_target.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=torch.tensor([label])))


    neg_smi=[]
    for di in polymer_smi_vial:
        print(di)
        neg_smi.append(dict_n[di])
        print(neg_smi)
    #print(len(neg_smi))
    
    pos_smi=[]
    for di in polymer_smi_vial:
        di_list=[]
        di_list.append(di)
        if is_valid_smiles(di+di+di): 
            di_list.append(di+di+di)
        if is_valid_smiles(di+di+di+di+di+di+di+di+di+di+di+di): 
            di_list.append(di+di+di+di+di+di+di+di+di+di+di+di)
        if is_valid_smiles(di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di): 
            di_list.append(di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di+di)
        #print(di_list)
        pos_smi.append(di_list)
    #print((pos_smi))

    dataset_neg=[]
    for smi_list in neg_smi:
        neg_data=[]
        for neg in smi_list:
            #print(neg)
            x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(neg))
            neg_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi))
        dataset_neg.append(neg_data)
    #print(dataset_neg[0])
    #print('dataset_neg',len(dataset_neg[1]))
    neg_index=[]
    for idx_neg in dataset_neg:
        #print(len(idx_neg))
        neg_index.append(len(idx_neg))
    
    #print('len(neg_index)',len(neg_index))
    number=0
    for nu in neg_index:
        number=number+nu
    #print(number)
    #print(len(sum(dataset_neg,[])))
 

    dataset_pos=[]
    for smi_list in pos_smi:
        pos_data=[]
        for pos in smi_list:
            #print(pos)
            x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(pos))
            pos_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi))
        dataset_pos.append(pos_data)
    #print('dataset_pos',(dataset_pos[0]))


    #print(neg_index)
    pos_index=[]
    for idx_pos in dataset_pos:
        #print(len(idx_pos))
        pos_index.append(len(idx_pos))
    
    #print('len(pos_index)',len(pos_index))
    number=0
    for nu in pos_index:
        number=number+nu
    #print(number)
    #print(len(sum(dataset_pos,[])))


    return dataset_source, dataset_target, sum(dataset_pos,[]), sum(dataset_neg,[]), pos_index, neg_index

def get_testdata(name):

    dataset = []


    df = pd.DataFrame(pd.read_csv('polyinfo_HSP_hz0827.csv'))

    smiles_list=df['SMILES of monomer'].tolist()
    labels_D=df['d'].tolist()
    labels_P=df['p'].tolist()
    labels_H=df['h'].tolist()

    #D拟合的好？

    labels=[]

    smiles_list_v=[]
    labels_D_v=[]
    labels_P_v=[]
    labels_H_v=[]

    for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,smiles_list)):
        if not is_valid_smiles(smi): continue

        smiles_list_v.append(smi)
        labels_D_v.append(D)
        labels_P_v.append(P)
        labels_H_v.append(H)

        #labels.append(D)

    
    labels=[]
    for D,P,H in tqdm(zip(labels_D_v, labels_P_v,labels_H_v)):
        labels.append([D,P,H])
    #labels=[0,1]
    
    labels=labels_D_v
    #print(labels_P_v)


    for smi, label in tqdm(zip(smiles_list_v, labels)):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=torch.tensor([label])))

    return dataset

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        pass
        # raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def get_mol_nodes_edges(mol):

    # Read node features
    N = mol.GetNumAtoms()
    #print(N)
    atom_type = []
    atomic_number = []
    aromatic = []
    hybridization = []
    # num_hs = []
    for atom in mol.GetAtoms():
        atom_type.append(atom.GetSymbol())
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization.append(atom.GetHybridization())

    # Read edge features
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond.GetBondType()]
    edge_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(t, [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]) for t in edge_type]
    edge_attr = torch.FloatTensor(edge_type)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    row, col = edge_index

    # Concat node fetures
    hs = (torch.tensor(atomic_number, dtype=torch.long) == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()
    x_atom_type = [one_of_k_encoding(t, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']) for t in atom_type]
    x_hybridization = [one_of_k_encoding(h, [HT.SP, HT.SP2, HT.SP3]) for h in hybridization]
    x2 = torch.tensor([atomic_number, aromatic, num_hs], dtype=torch.float).t().contiguous()
    #print(x2)
    #print(x_att_all)
    
    #x = torch.cat([torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2, torch.FloatTensor(x_att_all)], dim=-1)
    x = torch.cat([torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2], dim=-1)

    return x, edge_index, edge_attr




def is_valid_smiles(smi):
    try:
        Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
    except:
        print("not successfully processed smiles: ", smi)
        return False
    return True


cross_loss1 = torch.nn.CrossEntropyLoss().to(device)
cross_loss2 = torch.nn.CrossEntropyLoss().to(device)



import torch
import torch.nn.functional as F

def Contrastive_loss(positive_samples, negative_samples, pos_index, neg_index, temperature=0.07):
    """
    InfoNCE Loss function
    Args:
    - positive_samples: Tensor of positive samples, shape: (batch_size, embedding_size)
    - negative_samples: Tensor of negative samples, shape: (batch_size, num_negatives, embedding_size)
    - temperature: Scaling factor (default: 1.0)

    Returns:
    - Loss value
    """

    pos_index_list=0
    neg_index_list=0
    loss_all=0
    for pos,neg in zip(pos_index,neg_index):
        #print('pos',pos)
        #print('neg',neg)
        postitve_sample_all=[]
        negative_samples_all=[]
        postitve_ground_truth=[]
        negative_ground_truth=[]
        for i_p in range(pos_index_list,pos_index_list+pos):
            #print('pos_index_list',pos_index_list)
            #print('i_p',i_p)
            postitve_ground_truth.append(positive_samples[pos_index_list])
            postitve_sample_all.append(positive_samples[i_p])

        for i_n in range(neg_index_list,neg_index_list+neg):
            #print('i_n',i_n)
            negative_ground_truth.append(negative_samples[neg_index_list])
            negative_samples_all.append(negative_samples[i_n])

        pos_index_list= pos_index_list + pos
        neg_index_list= neg_index_list + neg

    # Positive dot products: similarity between positive pairs
        postitve_sample_all= torch.tensor([item.cpu().detach().numpy() for item in postitve_sample_all]).cuda()
        negative_sample_all= torch.tensor([item.cpu().detach().numpy() for item in negative_samples_all]).cuda()
        postitve_ground_truth= torch.tensor([item.cpu().detach().numpy() for item in postitve_ground_truth]).cuda()
        negative_ground_truth= torch.tensor([item.cpu().detach().numpy() for item in negative_ground_truth]).cuda()



        postitve_sample_all, negative_sample_all, postitve_ground_truth, negative_ground_truth = F.normalize(postitve_sample_all, dim=1), F.normalize(negative_sample_all, dim=1), F.normalize(postitve_ground_truth, dim=1), F.normalize(negative_ground_truth, dim=1)
        #pos_score = (postitve_sample_all * postitve_ground_truth).sum(dim=-1)
        #pos_score = torch.exp(pos_score / temperature)
        #tl_score = torch.matmul(negative_sample_all, negative_ground_truth.transpose(0, 1))
        #ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        #loss = -torch.log(pos_score / ttl_score)
    


        logits1 = torch.matmul(postitve_ground_truth, postitve_sample_all.t()) / (temperature*5)
        logits2 = torch.matmul(negative_ground_truth, negative_sample_all.t()) / temperature
        
        
        N_pos = postitve_ground_truth.size(0)
        labels_pos = torch.arange(N_pos).to(logits1.device)

        N_neg = negative_ground_truth.size(0)
        #print(N_neg)
        labels_neg = torch.arange(N_neg).to(logits2.device)

        loss = -F.cross_entropy(logits1, labels_pos) + F.cross_entropy(logits2, labels_neg)

        #logits1 = postitve_ground_truth @ postitve_sample_all.t()
        #logits2 = negative_ground_truth @ negative_sample_all.t()
        #ground_truth1 = torch.arange(len(logits1)).long().to(device)
        #ground_truth2 = torch.arange(len(logits2)).long().to(device)
        #loss = 0.5 * (cross_loss1(logits1, ground_truth1) + cross_loss2(logits2, ground_truth2)/temperature)
        loss_all=loss_all+loss
        #print(loss_all/len(pos_index))
    return loss_all/len(pos_index)



class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

source_data, target_data, pos_data, neg_data, pos_index, neg_index = get_traindata()


print('Source length',len(source_data))

print('Target length',len(target_data))

source_data_loader = DataLoader(source_data, batch_size=len(source_data), shuffle=True)
target_data_loader = DataLoader(target_data, batch_size=len(target_data))

pos_data_loader = DataLoader(pos_data, batch_size=len(pos_data))
neg_data_loader = DataLoader(neg_data, batch_size=len(neg_data))
#print(target_data_loader)
#print('batch',target_data_loader.batch)



loss_domain = nn.CrossEntropyLoss().to(device)

criterion=nn.MSELoss(reduction="mean").to(device)


hidden_channels = 64

encoder = GNN().to(device)



reg_model = nn.Sequential(
            #nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.BatchNorm1d(hidden_channels),
            #nn.ReLU(inplace=True), # second layer
            #nn.BatchNorm1d(hidden_channels, affine=False),# for acm dataset BatchNorm1d*3
            #nn.Linear(hidden_channels, hidden_channels),
            nn.Linear(hidden_channels, 3),
            #nn.LogSoftmax(dim=1)
            nn.Softplus(),
).to(device)

domain_model = nn.Sequential(
    GRL(),
    nn.Linear(hidden_channels, 32),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(32, 2),
).to(device)

def gcn_encode(data):
    encoded_output = encoder(data.x, data.edge_index, data.batch)
    return encoded_output



def encode(data):
    gcn_output = gcn_encode(data)
    return gcn_output

def predict(data):
    encoded_output = encode(data)
    logits = reg_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy

def test(data):
    for model in models:
        model.eval()
    logits = predict(data)
    logits=logits.cpu()
    logits=logits.detach().numpy()
    y=data.y.to(device)
    y=y.cpu()
    y=y.detach().numpy()
    rmse = torch.sqrt(torch.tensor(mean_squared_error(y, logits,multioutput='raw_values')))
    return rmse



models = [encoder, reg_model, domain_model]

params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=3e-3)




epochs = 200


def train(epoch, source_data, target_data, pos_data, neg_data, pos_index, neg_index):
    global rate
    rate = min((epoch + 1) / epochs, 0.05)
    #model.train()
    optimizer.zero_grad()
    #print('neg_data.x',len(neg_data.x))

    encoded_source = encode(source_data)
    encoded_target = encode(target_data)
    encoded_pos = encode(pos_data)
    encoded_neg = encode(neg_data)
    #print('encoded_pos',encoded_pos)
    #print('len_encoded_pos',len(encoded_pos))
    #print('encoded_neg',encoded_neg)
    #print('len_encoded_neg',len(encoded_neg))
    source_logits = reg_model(encoded_source)


    '''
    if epoch==1 or epoch==99:
        rng = np.random.RandomState(0)
        xxx=encoded_target.cpu().detach().numpy()
        print(xxx)
        brats_array = xxx
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(brats_array)
        xxx_s=encoded_source.cpu().detach().numpy()
        print(xxx_s)
        brats_array_s = xxx_s
        tsne_s = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(brats_array_s)
        # tsne 归一化， 这一步可做可不做
        x_min, x_max = tsne.min(0), tsne.max(0)
        tsne_norm = (tsne - x_min) / (x_max - x_min)
        #normal_idxs = (brats_label_array == 0)
        #abnorm_idxs = (brats_label_array == 1)
        tsne_normal = tsne
        tsne_normal_s = tsne_s
        #tsne_abnormal = tsne_norm[abnorm_idxs]
        cm=plt.cm.Greens
        cm_s=plt.cm.Blues
        #cm = plt.cm.get_cmap('RdYlBu')
        #cmap=cm
        plt.figure(figsize=(16, 16))

        #plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], 10, color='red',alpha=0.3,cmap='viridis')
        #plt.scatter(tsne_normal_s[:, 0], tsne_normal_s[:, 1], 10, color='blue',alpha=0.3,cmap='viridis')
        plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1],10, alpha=0.5,edgecolors='none')
        #c = plt.colorbar(fraction=0.05, pad=0.05)
        plt.scatter(tsne_normal_s[:, 0], tsne_normal_s[:, 1], 10, alpha=0.5,edgecolors='none')
        plt.axis('off')
        #c = plt.colorbar(fraction=0.05, pad=0.05)
        #c = plt.colorbar(fraction=0.05, pad=0.05,location='bottom')
        #c = plt.colorbar(im,  orientation='horizontal')
        # tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
        #plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], 1, color='green', label='Anomalous slices')
        plt.legend(loc='upper left')
        plt.show()
   '''

    #多标签
    # 前向传播计算，获得网络输出
    #outputs = self.forward(data.x.float(),data.edge_index,data.batch)
    
    # 计算损失值
    y=source_data.y.to(device)
    #print('yy)
    loss_reg = torch.sqrt(criterion(source_logits, y))

    #print(y)
    #loss_reg = self.loss_function1(outputs, y)
    source_domain_label = torch.zeros(len(source_data))
    source_domain_label = source_domain_label.long().cuda()
    target_domain_label = torch.zeros(len(target_data))
    target_domain_label = target_domain_label.long().cuda()
    
    source_domain_preds = domain_model(encoded_source)
    target_domain_preds = domain_model(encoded_target)
    #loss_reg = torch.sqrt(self.criterion(source_logits, y))#多标签
    err_s_domain = loss_domain(source_domain_preds, source_domain_label)
    err_t_domain = loss_domain(target_domain_preds, target_domain_label)
    loss_con = Contrastive_loss(encoded_pos,encoded_neg, pos_index, neg_index)
    #print(loss_con)
    loss= err_s_domain + err_t_domain + loss_reg + loss_con

    train_loss.append(loss.item())
    train_loss_con.append(loss_con.item())
    train_loss_s.append(err_s_domain.item())
    train_loss_t.append(err_t_domain.item())


        
    # 梯度清零, 反向传播, 更新权重

    loss.backward()
    optimizer.step()

# 测试函数\

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from glob import glob
import nibabel as nib
'''
for target_data in target_data_loader:
    print(len(target_data.x))
    print(target_data.x)
    brats_array = np.array(target_data.x, dtype='uint8')
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(brats_array)
    # tsne 归一化， 这一步可做可不做
    x_min, x_max = tsne.min(0), tsne.max(0)
    tsne_norm = (tsne - x_min) / (x_max - x_min)
    #normal_idxs = (brats_label_array == 0)
    #abnorm_idxs = (brats_label_array == 1)
    tsne_normal = tsne
    #tsne_abnormal = tsne_norm[abnorm_idxs]
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], 20, color='red')
    # tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
    #plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], 1, color='green', label='Anomalous slices')
    plt.legend(loc='upper left')
    plt.show()
'''
train_loss = []
train_loss_con=[]
train_loss_s= []
train_loss_t =[]
train_D = []
train_P = []
train_H = []
best_source_rsme_D = 100
best_target_rsme_D = 100
best_source_rsme_P = 100
best_target_rsme_P = 100
best_source_rsme_H = 100
best_target_rsme_H = 100
best_epoch = 0.0
for epoch in range(1, epochs):
    #print(epoch)
    for source_data, target_data, pos_data, neg_data in zip(source_data_loader,target_data_loader,pos_data_loader,neg_data_loader):
        #print('pos_data',len(pos_data))
        #print('tneg_data',len(neg_data))
        train(epoch=epoch, source_data=source_data, target_data=target_data, pos_data=pos_data, neg_data=neg_data, pos_index=pos_index, neg_index= neg_index)
    
    for source_data, target_data in zip(source_data_loader,target_data_loader):
        source_correct = test(source_data)
        target_correct = test(target_data)
    print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch, source_correct, target_correct))
    if target_correct[0] < best_target_rsme_D:
        best_target_rsme_D = target_correct[0]
        best_source_rsme_D = source_correct[0]
        best_epoch = epoch
    if target_correct[1] < best_target_rsme_P:
        best_target_rsme_P = target_correct[1]
        best_source_rsme_P = source_correct[1]
        best_epoch = epoch
    if target_correct[2] < best_target_rsme_H:
        best_target_rsme_H = target_correct[2]
        best_source_rsme_H = source_correct[2]
        best_epoch = epoch
    train_D.append((target_correct[0]-1.5).item())
    train_P.append((target_correct[1]-1.5).item())
    train_H.append((target_correct[2]-1).item())
print("=============================================================")
line = "{} - Epoch: {}, best_source_rsme: {}, best_target_rsme: {}"\
    .format(id, best_epoch, [best_source_rsme_D,best_source_rsme_P,best_source_rsme_H], [best_target_rsme_D, best_target_rsme_P, best_target_rsme_H])
print(train_loss)
with open('train_loss.pkl', 'wb') as file:
    pickle.dump(train_loss, file)
with open('train_loss_con.pkl', 'wb') as file:
    pickle.dump(train_loss_con, file)
with open('train_loss_s.pkl', 'wb') as file:
    pickle.dump(train_loss_s, file)
with open('train_loss_t.pkl', 'wb') as file:
    pickle.dump(train_loss_t, file)
with open('train_D.pkl', 'wb') as file:
    pickle.dump(train_D, file)
with open('train_P.pkl', 'wb') as file:
    pickle.dump(train_P, file)
with open('train_H.pkl', 'wb') as file:
    pickle.dump(train_H, file)
with open('train_H.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
print(loaded_data)



brats_array = np.array(target_data.x, dtype='uint8')
tsne = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(brats_array)
# tsne 归一化， 这一步可做可不做
x_min, x_max = tsne.min(0), tsne.max(0)
tsne_norm = (tsne - x_min) / (x_max - x_min)
#normal_idxs = (brats_label_array == 0)
#abnorm_idxs = (brats_label_array == 1)
tsne_normal = tsne_norm
#tsne_abnormal = tsne_norm[abnorm_idxs]
plt.figure(figsize=(8, 8))
plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], 5, color='red')
# tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
#plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], 1, color='green', label='Anomalous slices')
plt.legend(loc='upper left')
plt.show()
print(line)





    
