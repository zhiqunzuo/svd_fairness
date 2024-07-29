import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data

from models import weights_init_normal, test_model
from FairBatchSampler import FairBatch, CustomDataset
from scipy.optimize import fsolve
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
from collections import defaultdict

def run_epoch(model, train_features, labels, optimizer, criterion):
    optimizer.zero_grad()
    label_predicted = model.forward(train_features)
    loss = criterion((F.tanh(label_predicted.squeeze()) + 1) / 2, (labels.squeeze() + 1) / 2)
    loss.backward()
    optimizer.step()
    return loss.item()

def get_sigma_fair(sigma, k, lam):
    tem = (9 * k * sigma * lam ** 2 + np.sqrt(3) * np.sqrt(2 * k ** 2 * lam ** 3 + 27 * k ** 2 * sigma ** 2 * lam ** 4)) ** (1 / 3)
    return -k / (6 ** (1 / 3) * tem) + tem / (6 ** (2 / 3) * lam)

def func(lam, k, sigma, c):
    return sum([get_sigma_fair(sigma[i], k[i], lam) ** 4 for i in range(k.shape[0])]) - c

def get_lambda_(k, sigma, c):
    lambad_initial_guess = 1.0
    lambda_solution = fsolve(func, lambad_initial_guess, args=(k, sigma, c))
    lambda_optimal = lambda_solution[0]
    return lambda_optimal

def whiten(x):
    mean_x = torch.mean(x, dim=0)
    x_centered = x - mean_x

    cov_x = torch.mm(x_centered.t(), x_centered) / (x_centered.shape[0] - 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov_x)

    whitening_matrix = eigenvectors @ torch.diag(torch.pow(eigenvalues + 1e-7, -0.5)) @ eigenvectors.t()

    x_whitened = torch.mm(x_centered, whitening_matrix)

    return x_whitened

def select_k_similar_pairs(x1, x2, k):
    distances = pairwise_euclidean_distance(x1, x2)

    flatten_distances = distances.flatten()

    top_k_indices = torch.topk(flatten_distances, k, largest=False)[1]

    selected_x1 = torch.zeros((k, x1.size()[1]))
    selected_x2 = torch.zeros((k, x2.size()[1]))

    for i in range(k):
        selected_x1[i] = x1[int(top_k_indices[i] // distances.size()[1])]
        selected_x2[i] = x2[int(top_k_indices[i] % distances.size()[1])]

    return selected_x1, selected_x2

def normalize_data(xz_train, z_train, y_train, xz_test):
    #center_1 = torch.mean(xz_train[(z_train == 0.0) & (y_train == 1.0)], dim=0).unsqueeze(0)
    #center_2 = torch.mean(xz_train[(z_train == 1.0) & (y_train == 1.0)], dim=0).unsqueeze(0)
    center_1 = torch.mean(xz_train[z_train == 0.0], dim=0).unsqueeze(0)
    center_2 = torch.mean(xz_train[z_train == 1.0], dim=0).unsqueeze(0)
    
    std_1 = torch.std(xz_train[z_train == 0.0], dim=0).unsqueeze(0)
    std_2 = torch.std(xz_train[z_train == 1.0], dim=0).unsqueeze(0)
    
    for i in range(xz_train.size()[0]):
        if z_train[i] == 0.0:
            xz_train[i] = xz_train[i] - center_1
            xz_train[i] = xz_train[i] / (std_1 + 1e-7)
        else:
            xz_train[i] = xz_train[i] - center_2
            xz_train[i] = xz_train[i] / (std_2 + 1e-7)
    
    for i in range(xz_test.size()[0]):
        if z_test[i] == 0.0:
            xz_test[i] = xz_test[i] - center_1
            xz_test[i] = xz_test[i] / (std_1 + 1e-7)
        else:
            xz_test[i] = xz_test[i] - center_2
            xz_test[i] = xz_test[i] / (std_2 + 1e-7)
    
    return xz_train, xz_test

df = pd.read_csv("compas.csv")

df = df.dropna(subset=["days_b_screening_arrest"]) 
df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30) & (df.is_recid != -1) & (df.c_charge_degree != 'O') & (df.score_text != 'N/A') ]
df.reset_index(inplace=True, drop=True) 

FEATURES_CLASS = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"] 
CONTI_FEATURE = ["priors_count"] 
CLASS_FEATURE = "two_year_recid"
SENSITIVE_FEATURE = "sex"

data = df.to_dict('list')
for k in data.keys():
		data[k] = np.array(data[k])
  
y = data[CLASS_FEATURE]
y[y==0] = -1

X = np.array([]).reshape(len(y), 0)
x_control = defaultdict(list)
feature_names = []
for attr in FEATURES_CLASS:
        vals = data[attr]
        if attr in CONTI_FEATURE:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance  
            vals = np.reshape(vals, (len(y), -1))

        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

		# add to sensitive features dict
        if attr in SENSITIVE_FEATURE:
            x_control[attr] = vals


		# add to learnable features
        X = np.hstack((X, vals))

        if attr in CONTI_FEATURE: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

# convert the sensitive feature to 1-d array
x_control = dict(x_control)
for k in x_control.keys():
		#assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
		x_control[k] = np.array(x_control[k]).flatten()
	
#print(x_control)
#print(X.shape)
#print(y.shape)
#print(feature_names)
z = x_control["sex"]

xz_train = X[:int(X.shape[0] * 0.8)]
y_train = y[:int(X.shape[0] * 0.8)]
z_train = z[:int(X.shape[0] * 0.8)]

xz_test = X[int(X.shape[0] * 0.8):]
y_test = y[int(X.shape[0] * 0.8):]
z_test = z[int(X.shape[0] * 0.8):]

#xz_train = np.load('./synthetic_data/xz_train.npy')
#y_train = np.load('./synthetic_data/y_train.npy') 
#z_train = np.load('./synthetic_data/z_train.npy')

#xz_test = np.load('./synthetic_data/xz_test.npy')
#y_test = np.load('./synthetic_data/y_test.npy') 
#z_test = np.load('./synthetic_data/z_test.npy')

xz_train = torch.FloatTensor(xz_train)
y_train = torch.FloatTensor(y_train)
z_train = torch.FloatTensor(z_train)

xz_test = torch.FloatTensor(xz_test)
y_test = torch.FloatTensor(y_test)
z_test = torch.FloatTensor(z_test)

"""
full_tests = []
# Set the train data
train_data = CustomDataset(xz_train, y_train, z_train)

seeds = [0,1,2,3,4,5,6,7,8,9]
for seed in seeds:
    
    print("< Seed: {} >".format(seed))
    
    # ---------------------
    #  Initialize model, optimizer, and criterion
    # ---------------------
    
    #model = LogisticRegression(3,1)
    model = nn.Sequential(
        nn.Linear(12, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    torch.manual_seed(seed)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()

    losses = []
    
    # ---------------------
    #  Define FairBatch and DataLoader
    # ---------------------
    
    sampler = FairBatch (model, train_data.x, train_data.y, train_data.z, batch_size = 100, alpha = 0.005, target_fairness = 'dp', replacement = False, seed = seed)
    train_loader = torch.utils.data.DataLoader (train_data, sampler=sampler, num_workers=0)

    # ---------------------
    #  Model training
    # ---------------------
    for epoch in range(450):

        tmp_loss = []
        
        for batch_idx, (data, target, z) in enumerate (train_loader):
            loss = run_epoch (model, data, target, optimizer, criterion)
            tmp_loss.append(loss)
            
        losses.append(sum(tmp_loss)/len(tmp_loss))
        
    tmp_test = test_model(model, xz_test, y_test, z_test)
    full_tests.append(tmp_test)
    
    print("  Test accuracy: {}, DP disparity: {}".format(tmp_test['Acc'], tmp_test['DP_diff']))
    print("----------------------------------------------------------------------")

tmp_acc = []
tmp_dp = []
for i in range(len(seeds)):
    tmp_acc.append(full_tests[i]['Acc'])
    tmp_dp.append(full_tests[i]['DP_diff'])

print("Test accuracy (avg): {}".format(sum(tmp_acc)/len(tmp_acc)))
print("DP disparity  (avg): {}".format(sum(tmp_dp)/len(tmp_dp)))
"""

xz_train, xz_test = normalize_data(xz_train, z_train, y_train, xz_test)

full_tests = []
dp_tests = []

train_data = CustomDataset(xz_train, y_train, z_train)

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#seeds = [2]

for seed in seeds:
    print("< Seed: {} >".format(seed))
    
    model = nn.Sequential(
        nn.Linear(12, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    
    torch.manual_seed(seed)
    model.apply(weights_init_normal)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()
    
    losses = []
    
    sampler = FairBatch(model, train_data.x, train_data.y, train_data.z, batch_size=100, alpha=0.005, target_fairness="original", replacement=False, seed=seed)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, num_workers=0)
    
    for epoch in range(450):
        tmp_loss = []
        for batch_idx, (data, target, z) in enumerate(train_loader):
            loss = run_epoch(model, data, target, optimizer, criterion)
            tmp_loss.append(loss)
        
        losses.append(sum(tmp_loss) / len(tmp_loss))
    
    tmp_test = test_model(model, xz_test, y_test, z_test)
    full_tests.append(tmp_test)
    
    print("  Test accuracy: {}, DP disparity: {}".format(tmp_test['Acc'], tmp_test['DP_diff']))
    print("----------------------------------------------------------------------")
    
    weight = model[0].weight.data
    
    #x1 = xz_train[(z_train == 0.0) & (y_train == 1.0)]
    #x2 = xz_train[(z_train == 1.0) & (y_train == 1.0)]
    x1 = xz_train[(z_train == 0.0) & (y_train == 1.0)]
    x2 = xz_train[(z_train == 1.0) & (y_train == 1.0)]
    
    x1 = x1 - torch.mean(x1, dim=0).unsqueeze(0)
    x2 = x2 - torch.mean(x2, dim=0).unsqueeze(0)
    
    xxt = (1 / (x1.size()[0] - 1)) * torch.mm(x1.T, x1) - (1 / (x2.size()[0] - 1)) * torch.mm(x2.T, x2)
    print("xxt = {}".format(xxt))
    try:
        s = torch.linalg.cholesky(xxt + torch.eye(xxt.size()[0]) * 1)
    except:
        s = torch.linalg.cholesky(-xxt + torch.eye(xxt.size()[0]) * 1)
    ws = torch.mm(weight, s)
    u, sig, v = torch.linalg.svd(ws, full_matrices=False)
    print("sig = {}".format(sig))
    
    #k = torch.zeros(sig.size(0))
    #for i in range(sig.size(0)):
    #    k[i] = torch.mm(v[i].unsqueeze(0), torch.mm(torch.linalg.inv(s), 
    #                                                torch.mm(torch.linalg.inv(s).T, v[i].unsqueeze(0).T)))
    #print("k = {}".format(k))
    #lambda_ = get_lambda_(k.clone().detach().numpy(), sig.clone().detach().numpy(), c=(torch.sum(sig ** 4) / 200).item())
    
    #sig_fair = (k * sig) / (k + lambda_)
    #print("sig fair = {}".format(sig_fair))
    k = torch.zeros(sig.size()[0])
    for i in range(sig.size()[0]):
        k[i] = torch.mm(v[i].unsqueeze(0), 
                        torch.mm(torch.linalg.inv(s),
                                 torch.mm(xz_train.T,
                                          torch.mm(xz_train,
                                                   torch.mm(torch.linalg.inv(s).T, v[i].unsqueeze(0).T)))))
    lam = get_lambda_(k.clone().detach().numpy(), sig.clone().detach().numpy(), (torch.sum(sig ** 4) / 50).item())
    sig_fair = torch.zeros(sig.size()[0])
    for i in range(sig.size()[0]):
        sig_fair[i] = get_sigma_fair(sig[i], k[i], lam)
    print("sig_fair = {}".format(sig_fair))
    
    #sig[0] = sig[0] * 1 / 8
    #print("sig summation after SVD = {}".format(torch.sqrt(torch.sum(sig_fair ** 2))))
    #print("sig sum = {}".format(torch.sqrt(torch.sum(sig ** 2))))
    #ws = torch.mm(u, torch.mm(torch.diag(sig), v))
    ws = torch.mm(u, torch.mm(torch.diag(sig_fair), v))
    weight = torch.mm(ws, torch.linalg.inv(s))
    
    model[0].weight.data = weight
    
    dp_test = test_model(model, xz_test, y_test, z_test)
    dp_tests.append(dp_test)
    

    #out_x1 = model[2](model[1](model[0](x1)))
    #out_x2 = model[2](model[1](model[0](x2)))
    #diff = torch.sqrt(torch.sum((out_x1 - out_x2) ** 2))
    #print("diff = {}".format(diff))
    
    print("After SVD Test accuracy: {}, DP disparity: {}".format(dp_test['Acc'], dp_test['DP_diff']))
    print("----------------------------------------------------------------------")
    

tmp_acc = []
tmp_dp = []
tmp_eo = []
for i in range(len(seeds)):
    tmp_acc.append(full_tests[i]["Acc"])
    tmp_dp.append(full_tests[i]["DP_diff"])
    tmp_eo.append(full_tests[i]['EO_Y1_diff'])

print("Test accuracy (avg): {}".format(sum(tmp_acc)/len(tmp_acc)))
print("DP disparity  (avg): {}".format(sum(tmp_dp)/len(tmp_dp)))
print("EO disparity  (avg): {}".format(sum(tmp_eo)/len(tmp_eo)))

dp_acc = []
dp_dp = []
dp_eo = []
for i in range(len(seeds)):
    dp_acc.append(dp_tests[i]["Acc"])
    dp_dp.append(dp_tests[i]["DP_diff"])
    dp_eo.append(dp_tests[i]['EO_Y1_diff'])

print("Test accuracy (avg): {}".format(sum(dp_acc)/len(dp_acc)))
print("DP disparity  (avg): {}".format(sum(dp_dp)/len(dp_dp)))
print("EO disparity  (avg): {}".format(sum(dp_eo)/len(dp_eo)))