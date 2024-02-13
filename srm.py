#-*-coding:utf-8-*-

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import grad
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def data_generation(n1, n2, ps, pvb, pv, r, scramble=0):
    S = np.random.normal(0, 2, [n1, ps])
    V = np.random.normal(0, 2, [n1, pvb + pv])
    Z = np.random.normal(0, 1, [n1, ps + 1])
    for i in range(ps):
        S[:, i:i + 1] = 0.8 * Z[:, i:i + 1] + 0.2 * Z[:, i + 1:i + 2]
 
    beta = np.zeros((ps, 1))
    for i in range(ps):
        beta[i] = (-1) ** i * (i % 3 + 1) * 1.0/2.0

    noise = np.random.normal(0, 1.0, [n1, 1])
    Y = np.dot(S, beta) + noise + 1.0 * S[:, 0:1] * S[:, 1:2] * S[:, 2:3]
    index_pre = np.ones([n1, 1], dtype=bool)
    for i in range(pvb):
        D = np.abs(V[:, pv + i:pv + i + 1] * sign(r) - Y)
        pro = np.power(np.abs(r), -D * 5)
        selection_bias = np.random.random([n1, 1])
        index_pre = index_pre & (
                    selection_bias < pro)
    index = np.where(index_pre == True)
    S_re = S[index[0], :]
    V_re = V[index[0], :]
    Y_re = Y[index[0]]
    n, p = S_re.shape
    index_s = np.random.permutation(n)

    X_re = np.hstack((S_re, V_re))
    beta_X = np.vstack((beta, np.zeros((pv + pvb, 1))))

    X = torch.from_numpy(X_re[index_s[0:n2], :]).float()
    y =  torch.from_numpy(Y_re[index_s[0:n2], :]).float()
    
    return X, y

class MLP(nn.Module):
    def __init__(self, input_size=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 1)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        nn.init.normal_(self.layer1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.layer1.bias, val=0)
        nn.init.normal_(self.layer2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.layer2.bias, val=0)
        nn.init.normal_(self.layer3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.layer3.bias, val=0)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)
    
    def features(self, x):
        out = self.relu1(self.layer1(x))
        return self.relu2(self.layer2(out))


class Linear(nn.Module):
    def __init__(self, input_size=10, hidden_size=50):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_size, 1, bias=False)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
    def forward(self, x):
        return self.fc(x)

class RegularizerSolver():
    def __init__(self, trainX, trainY, alpha0, model='linear'):
        self.device = torch.device("cuda:2")
        self.trainX = trainX
        self.trainY = trainY
        self.init_weight = (torch.ones(trainY.shape[0],dtype=float)/trainX.shape[0]).to(self.device)
        self.weight = (torch.ones(trainY.shape[0],dtype=float)/trainX.shape[0]).to(self.device)
        self.source_model = None
        self.target_model = None
        self.alpha0 = alpha0
        self.model_type=model
        self.threshold = None
        self.gap = None
        
    def fetch_model(self):
        if self.model_type == 'mlp':
            return MLP(input_size=self.trainX.shape[1]).to(self.device)
        elif self.model_type == 'linear':
            return Linear(input_size=self.trainX.shape[1]).to(device)

    def optimize_model(self, weight, epochs):
        model = self.fetch_model()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss = 1e3
        for i in range(1,1+epochs):
            if self.threshold is not None and loss < self.threshold:
                break
            pred_y = model(self.trainX)
            loss = torch.dot(weight.float().reshape(-1), ((pred_y-self.trainY)**2).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return model, loss

    def optimize_weight(self):
        pred_source = self.source_model(self.trainX)
        pred_target = self.target_model(self.trainX)
        loss_source = (pred_source-self.trainY)**2
        loss_target = (pred_target-self.trainY)**2
        gap = loss_source-loss_target
        index = torch.argsort(gap.reshape(-1), descending=True)
        self.weight = torch.zeros(self.trainY.shape[0], dtype=float).to(self.device)
        self.weight[index[:int(self.alpha0*self.trainY.shape[0])]] = 1.0/int(self.alpha0*self.trainY.shape[0])
        return torch.mean(gap[index[:int(self.alpha0*self.trainY.shape[0])]]).detach()
    
    def learn_weights(self, whole_epochs=5, epochs=10000):
        self.source_model, self.threshold = self.optimize_model(self.init_weight, epochs)
        pred_source = self.source_model(self.trainX)
        loss_source = (pred_source-self.trainY)**2
        index = torch.argsort(loss_source.reshape(-1), descending=True)
        self.weight = torch.zeros(self.trainY.shape[0]).to(self.device)
        select_sum = int(self.alpha0*self.trainY.shape[0])
        self.weight[index[:select_sum]] = 1.0/select_sum
        V_list = []
        gap_list = []
        for i in range(whole_epochs):
            self.target_model, _ = self.optimize_model(self.weight, epochs)
            self.gap = self.optimize_weight()
            gap_list.append(self.gap)
            
        return self.weight.cpu().detach()


def SRM(trainX, trainY, epochs, lam = 0.1, ratio=0.1, delta=0.8):
    device = torch.device("cuda")
    model = MLP(input_size=trainX.shape[1]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    trainX = trainX.to(device)
    trainY = trainY.to(device)
    
    learned_weight = None
    penalty = 0.0
    for i in range(1, epochs):
        model.train()
        if i == 1000:
            phis = model.features(trainX).clone().detach()
            solver = RegularizerSolver(phis, trainY, ratio, model='mlp')
            learned_weight = solver.learn_weights(whole_epochs=1, epochs=300).float().to(device)
            if solver.gap < delta:
                return model.cpu()

        if learned_weight is not None:
            pred = model(trainX)
            error = (pred-trainY)**2
            loss_1 = torch.mean(error)
            grad_1 = grad(torch.mean(error), model.parameters(), create_graph=True, allow_unused=True)[0]
            loss_2 = torch.dot(learned_weight.reshape(-1), error.reshape(-1))
            grad_2 = grad(torch.dot(learned_weight.reshape(-1), error.reshape(-1)), model.parameters(), create_graph=True)[0]
            penalty = (grad_1-grad_2).pow(2).mean()
            loss = (loss_1 + loss_2) / 2 + lam * penalty
        else:
            pred = model(trainX)
            error = (pred-trainY)**2
            loss = torch.mean(error)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model.cpu()


