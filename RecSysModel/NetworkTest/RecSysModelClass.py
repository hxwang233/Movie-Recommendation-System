#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
from torch.nn import init
import torch.utils.data as data_utils
from torch.autograd import Variable
import numpy as np
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# In[ ]:


class GMF(nn.Module):
    def __init__(self, n_factors, n_user, n_item, activation = torch.relu, batch_normalization = False, n_output = 1):
        super(GMF, self).__init__()
        self.activation = activation
        self.do_bn = batch_normalization
        parameter_LeCun = np.sqrt(n_factors)
        
        #self.bn_userInput = nn.BatchNorm1d(1)   # for input data
        #self.bn_itemInput = nn.BatchNorm1d(1)   # for input data
        
        self.gmf_user_embedding_layer = nn.Embedding(n_user, n_factors)
        self._set_normalInit(self.gmf_user_embedding_layer, hasBias = False) 
        self.gmf_item_embedding_layer = nn.Embedding(n_item, n_factors)
        self._set_normalInit(self.gmf_item_embedding_layer, hasBias = False) 
        
        #self.bn_user_elayer = nn.BatchNorm1d(mlp_embedding_size) 
        #self.bn_item_elayer = nn.BatchNorm1d(mlp_embedding_size)     

        self.predict = nn.Linear(n_factors, n_output)         # output layer
        self._set_uniformInit(self.predict, parameter = parameter_LeCun)            # parameters initialization
        return

    def _set_normalInit(self, layer, parameter = [0.0, 0.01], hasBias=True):
        init.normal_(layer.weight, mean = parameter[0], std = parameter[1])
        if hasBias:
            init.normal_(layer.bias, mean = parameter[0], std = parameter[1])
        return
    
    def _set_uniformInit(self, layer, parameter = 5, hasBias = True):
        init.uniform_(layer.weight, a = - parameter, b = parameter)
        if hasBias:
            init.uniform_(layer.bias, a = - parameter, b = parameter)
        return
    
    def _set_heNormalInit(self, layer, hasBias=True):
        init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if hasBias:
            init.kaiming_normal_(layer.bias, nonlinearity='relu')
        return
    
    def _set_heUniformInit(self, layer, hasBias=True):
        init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if hasBias:
            init.kaiming_uniform_(layer.bias, nonlinearity='relu')
        return

    def forward(self, x1, x2):
        #if self.do_bn: 
            #x1 = self.bn_userInput(x1)     # input batch normalization
            #x2 = self.bn_itemInput(x2)
        x1 = self.gmf_user_embedding_layer(x1)
        x2 = self.gmf_item_embedding_layer(x2)
        x3 = torch.mul(x1, x2)
        #print(x3.data.numpy().shape)
        x  = torch.flatten(x3, start_dim=1)
        #print(x.data.numpy().shape)
        out = torch.sigmoid(self.predict(x))
        return out


class NCF(nn.Module):
    def __init__(self, gmf_n_factors, layers,  n_user, n_item, activation = torch.relu, batch_normalization = False, n_output = 1):
        super(NCF, self).__init__()
        self.activation = activation
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        self.n_layer  = len(layers)
        parameter_LeCun = np.sqrt(gmf_n_factors + layers[-1])

        #self.bn_userInput = nn.BatchNorm1d(1)   # for input data
        #self.bn_itemInput = nn.BatchNorm1d(1)   # for input data
        
        self.mlp_user_embedding_layer = nn.Embedding(n_user, int(layers[0]/2))
        self._set_normalInit(self.mlp_user_embedding_layer, hasBias = False) 
        self.mlp_item_embedding_layer = nn.Embedding(n_item, int(layers[0]/2))
        self._set_normalInit(self.mlp_item_embedding_layer, hasBias = False) 
        
        self.gmf_user_embedding_layer = nn.Embedding(n_user, gmf_n_factors)
        self._set_normalInit(self.gmf_user_embedding_layer, hasBias = False) 
        self.gmf_item_embedding_layer = nn.Embedding(n_item, gmf_n_factors)
        self._set_normalInit(self.gmf_item_embedding_layer, hasBias = False) 
        
        for i in range(1, self.n_layer):               # build hidden layers and BN layers
            fc = nn.Linear(layers[i-1], layers[i])
            self._set_normalInit(fc)                  # parameters initialization
            setattr(self, 'fc%i' % i, fc)       # IMPORTANT set layer to the Module
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(layers[i])
                setattr(self, 'bn%i' % i, bn)   # IMPORTANT set layer to the Module
                self.bns.append(bn)

        self.predict = nn.Linear(gmf_n_factors + layers[-1], n_output)         # output layer
        self._set_uniformInit(self.predict, parameter = parameter_LeCun)            # parameters initialization
        return

    def _set_normalInit(self, layer, parameter = [0.0, 0.01], hasBias=True):
        init.normal_(layer.weight, mean = parameter[0], std = parameter[1])
        if hasBias:
            init.normal_(layer.bias, mean = parameter[0], std = parameter[1])
        return
    
    def _set_uniformInit(self, layer, parameter = 5, hasBias = True):
        init.uniform_(layer.weight, a = - parameter, b = parameter)
        if hasBias:
            init.uniform_(layer.bias, a = - parameter, b = parameter)
        return
    
    def _set_heNormalInit(self, layer, hasBias=True):
        init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if hasBias:
            init.kaiming_normal_(layer.bias, nonlinearity='relu')
        return
    
    def _set_heUniformInit(self, layer, hasBias=True):
        init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if hasBias:
            init.kaiming_uniform_(layer.bias, nonlinearity='relu')
        return

    def forward(self, x1, x2):
        #if self.do_bn: 
            #x1 = self.bn_userInput(x1)     # input batch normalization
            #x2 = self.bn_itemInput(x2)
        mlp_x1 = self.mlp_user_embedding_layer(x1)
        mlp_x2 = self.mlp_item_embedding_layer(x2)
        
        gmf_x1 = self.gmf_user_embedding_layer(x1)
        gmf_x2 = self.gmf_item_embedding_layer(x2)
        
        mlp_x3 = torch.cat((mlp_x1, mlp_x2), dim=1)
        mlp_x  = torch.flatten(mlp_x3, start_dim=1)        
        for i in range(1, self.n_layer):
            mlp_x = self.fcs[i-1](mlp_x)
            if self.do_bn: 
                mlp_x = self.bns[i-1](mlp_x)   # batch normalization
            mlp_x = self.activation(mlp_x)
        
        gmf_x3 = torch.mul(gmf_x1, gmf_x2)
        gmf_x  = torch.flatten(gmf_x3, start_dim=1)

        x = torch.cat((mlp_x, gmf_x), dim=1)
        out = torch.sigmoid(self.predict(x))
        return out


# In[ ]:


class ConvNCF(nn.Module):
    def __init__(self, fm_sizes, n_user, n_item, n_fm, drop_out, myStride=2, n_output=1):
        ''' e.g.--> fm_sizes = [64,32,16,8,4,2,1] '''
        super(ConvNCF, self).__init__()
        self.convs = list()
        self.dropout = nn.Dropout(p=drop_out)
        self.user_embedding_layer = nn.Embedding(n_user, fm_sizes[0])
        self._set_normalInit(self.user_embedding_layer, hasBias = False) 
        #self._set_xavierInit(self.user_embedding_layer, hasBias = False)
        #self._set_heInit(self.user_embedding_layer, hasBias = False) 
        self.item_embedding_layer = nn.Embedding(n_item, fm_sizes[0])
        self._set_normalInit(self.item_embedding_layer, hasBias = False) 
        #self._set_xavierInit(self.item_embedding_layer, hasBias = False)
        #self._set_heInit(self.item_embedding_layer, hasBias = False) 
        for i in range(1, len(fm_sizes)):
            inChannel = 1 if i == 1 else n_fm
            #conv = nn.Conv2d(in_channels=inChannel, out_channels=32, kernel_size=fm_sizes[i]+myStride, stride=myStride)
            conv = nn.Conv2d(in_channels=inChannel, out_channels=n_fm, kernel_size=4, stride=myStride, padding=1)
            #self._set_normalInit(conv)
            #self._set_xavierInit(conv)
            self._set_heInit(conv)
            setattr(self, 'conv%i' % i, conv)
            self.convs.append(conv)

        self.predict = nn.Linear(n_fm, n_output)         # output layer
        self._set_xavierInit(self.predict)            # parameters initialization
        return
    
    def _set_xavierInit(self, layer, hasBias = True):
        init.xavier_uniform_(layer.weight)
        if hasBias:
            init.constant_(layer.bias, 0.01)
        return
    
    def _set_heInit(self, layer, hasBias = True):
        init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if hasBias:
            init.constant_(layer.bias, 0.01)
        return
    
    def _set_normalInit(self, layer, parameter = [0.0, 0.1], hasBias = True):
        init.normal_(layer.weight, mean = parameter[0], std = parameter[1])
        if hasBias:
            init.constant_(layer.bias, 0.01)
        return
    
    def _set_uniformInit(self, layer, parameter = 1, hasBias = True):
        init.uniform_(layer.weight, a = 0, b = parameter)
        if hasBias:
            init.uniform_(layer.bias, a = 0, b = parameter)
        return
    
    def forward(self, user, item_pos, item_neg, train = True):
        user = self.user_embedding_layer(user)
        item_pos = self.item_embedding_layer(item_pos)
        if train:
            item_neg = self.item_embedding_layer(item_neg)
        x1, x2 = None, None
        temp1, temp2 = list(), list() 
        out1, out2 = None, None
        for i in range(user.size()[0]):
            temp1.append(torch.mm(user[i].T, item_pos[i]))
            if train:
                temp2.append(torch.mm(user[i].T, item_neg[i]))
        x1 = torch.stack(temp1)
        x1 = x1.view(x1.size()[0], -1, x1.size()[1], x1.size()[2])
        if train:
            x2 = torch.stack(temp2)
            x2 = x2.view(x2.size()[0], -1, x2.size()[1], x2.size()[2])
        ''' ## conv2d -input  (batch_size, channel, weight, height) '''
        for conv in self.convs:
            x1 = torch.relu(conv(x1))
            if train:
                x2 = torch.relu(conv(x2))
        ''' ## conv2d -output (batch_size, out_channel, out_weight, out_height) '''
        x1 = torch.flatten(x1, start_dim = 1)
        x1 = self.dropout(x1)
        if train:
            x2 = torch.flatten(x2, start_dim = 1)
            x2 = self.dropout(x2)
        #out1 = torch.sigmoid(self.dropout(self.predict(x1)))
        out1 = torch.sigmoid(self.predict(x1))
        if train:
            #out2 = torch.sigmoid(self.dropout(self.predict(x2)))
            out2 = torch.sigmoid(self.predict(x2))
        return out1, out2


# In[ ]:


class ENMF(nn.Module):
    def __init__(self, emb_size, n_user, n_item, neg_weight, drop_out, count, c0=512, x=0.6):
        super().__init__()
        self.c0 = c0
        self.x  = x
        self.count = count
        self.n_user = n_user
        self.n_item = n_item
        self.neg_weight = neg_weight
        self.emb_size   = emb_size
        self.user_embs = nn.Embedding(n_user, emb_size)
        self.item_embs = nn.Embedding(n_item+1, emb_size)
        self.h = nn.Parameter(torch.randn(emb_size, 1))
        self.dropout = nn.Dropout(p=drop_out)
        self.freq = self.calcu_freq()
        self._reset_para()
        return
    
    def _reset_para(self):
        nn.init.xavier_normal_(self.user_embs.weight)
        nn.init.xavier_normal_(self.item_embs.weight)
        nn.init.constant_(self.h, 0.01)
        return
    
    def calcu_freq(self):
        freq_items = sorted(self.count.keys())
        freq_count = [self.count[k] for k in freq_items]
        freq = np.zeros(self.item_embs.weight.shape[0])
        freq[freq_items] = freq_count       
        #freq = freq/np.sum(freq)
        freq = np.power(freq, self.x)
        freq = self.c0 * freq/np.sum(freq)
        freq = torch.from_numpy(freq).type(torch.float).cuda()
        return freq
    
    def forward(self, uids, pos_iids):
        '''
        uids: B
        u_iids: B * L
        '''
        u_emb = self.dropout(self.user_embs(uids))
        pos_embs = self.item_embs(pos_iids)

        # torch.einsum("ab,abc->abc")
        # B * L * D
        mask = (~(pos_iids.eq(self.n_item))).float()
        pos_embs = pos_embs * mask.unsqueeze(2)

        # torch.einsum("ac,abc->abc")
        # B * L * D
        pq = u_emb.unsqueeze(1) * pos_embs
        # torch.einsum("ajk,kl->ajl")
        # B * L
        hpq = pq.matmul(self.h).squeeze(2)

        # loss
        pos_data_loss = torch.sum((1 - self.neg_weight) * hpq.square() - 2.0 * hpq)

        # torch.einsum("ab,ac->abc")
        part_1 = self.item_embs.weight.unsqueeze(2).bmm(self.item_embs.weight.unsqueeze(1))
        part_2 = u_emb.unsqueeze(2).bmm(u_emb.unsqueeze(1))

        # D * D
        part_1 = part_1.sum(0)
        part_2 = part_2.sum(0)
        part_3 = self.h.mm(self.h.t())
        all_data_loss = torch.sum(part_1 * part_2 * part_3)

        loss = self.neg_weight * all_data_loss + pos_data_loss
        return loss
    
    def rank(self, uid):
        '''
        uid: Batch_size
        '''
        uid_embs = self.user_embs(uid)
        user_all_items = uid_embs.unsqueeze(1) * self.item_embs.weight
        items_score = user_all_items.matmul(self.h).squeeze(2)
        return items_score
    
'''    def rank(self, user):
        res = self.user_embs(user).unsqueeze(0)
        res = res * self.item_embs.weight
        res = res.matmul(self.h).squeeze(1)
        return res'''

