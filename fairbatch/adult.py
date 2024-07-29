import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import weights_init_normal, test_model
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from FairBatchSampler import FairBatch, CustomDataset
from scipy.optimize import fsolve

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

def oneHotCatVars(df, df_cols):
    df_1 = df.drop(columns=df_cols, axis=1)
    df_2 = pd.get_dummies(df[df_cols])
    return pd.concat([df_1, df_2], axis=1, join="inner")

def normalize_data(xz_train, z_train, y_train, xz_test, z_test):
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

# read the data
column_names = ["age", "workclass", "fnlwgt", "education", "educational-num", "matrial-status", "occupation",
                "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week",
                "native-country", "income"]

train = pd.read_csv("adult_data.txt", sep=",\s", header=None, names=column_names, engine="python")
test = pd.read_csv("adult_test.txt", sep=",\s", header=None, names=column_names, engine="python")
test["income"].replace(regex=True, inplace=True, to_replace=r"\.", value=r"")

adult = pd.concat([test, train])
adult.reset_index(inplace=True, drop=True)

for col in set(adult.columns) - set(adult.describe().columns):
    adult[col] = adult[col].astype("category")

# predict the missing values - workclass
test_data = adult[(adult.workclass.values == "?")].copy()
test_label = test_data.workclass

train_data = adult[(adult.workclass.values != "?")].copy()
train_label = train_data.workclass

train_data.drop(columns=["workclass"], inplace=True)
test_data.drop(columns=["workclass"], inplace=True)

train_data = oneHotCatVars(train_data, train_data.select_dtypes("category").columns)
test_data = oneHotCatVars(test_data, test_data.select_dtypes("category").columns)

log_reg = LogisticRegression()
log_reg.fit(train_data, train_label)
log_reg_pred = log_reg.predict(test_data)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_label)
clf_pred = clf.predict(test_data)

r_forest = RandomForestClassifier(n_estimators=10)
r_forest.fit(train_data, train_label)
r_forest_pred = r_forest.predict(test_data)

majority_class = adult.workclass.value_counts().index[0]

pred_df =  pd.DataFrame({'RFor': r_forest_pred, 'DTree' : clf_pred, 'LogReg' : log_reg_pred})
overall_pred = pred_df.apply(lambda x: x.value_counts().index[0] if x.value_counts().iloc [0] > 1 else majority_class, axis = 1)

adult.loc[(adult.workclass.values == '?'),'workclass'] = overall_pred.values

test_data = adult[(adult.occupation.values == '?')].copy()
test_label = test_data.occupation

train_data = adult[(adult.occupation.values != '?')].copy()
train_label = train_data.occupation

test_data.drop(columns = ['occupation'], inplace = True)
train_data.drop(columns = ['occupation'], inplace = True)

train_data = oneHotCatVars(train_data, train_data.select_dtypes('category').columns)
test_data = oneHotCatVars(test_data, test_data.select_dtypes('category').columns)

log_reg = LogisticRegression()
log_reg.fit(train_data, train_label)
log_reg_pred = log_reg.predict(test_data)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_label)
clf_pred = clf.predict(test_data)

r_forest = RandomForestClassifier(n_estimators=10)
r_forest.fit(train_data, train_label)
r_forest_pred = r_forest.predict(test_data)


majority_class = adult.occupation.value_counts().index[0]

pred_df =  pd.DataFrame({'RFor': r_forest_pred, 'DTree' : clf_pred, 'LogReg' : log_reg_pred})
overall_pred = pred_df.apply(lambda x: x.value_counts().index[0] if x.value_counts().iloc[0] > 1 else majority_class, axis = 1)

adult.loc[(adult.occupation.values == '?'),'occupation'] = overall_pred.values

test_data = adult[(adult['native-country'].values == '?')].copy()
test_label = test_data['native-country']

train_data = adult[(adult['native-country'].values != '?')].copy()
train_label = train_data['native-country']

test_data.drop(columns = ['native-country'], inplace = True)
train_data.drop(columns = ['native-country'], inplace = True)

train_data = oneHotCatVars(train_data, train_data.select_dtypes('category').columns)
test_data = oneHotCatVars(test_data, test_data.select_dtypes('category').columns)

log_reg = LogisticRegression()
log_reg.fit(train_data, train_label)
log_reg_pred = log_reg.predict(test_data)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_label)
clf_pred = clf.predict(test_data)

r_forest = RandomForestClassifier(n_estimators=10)
r_forest.fit(train_data, train_label)
r_forest_pred = r_forest.predict(test_data)


majority_class = adult['native-country'].value_counts().index[0]

pred_df =  pd.DataFrame({'RFor': r_forest_pred, 'DTree' : clf_pred, 'LogReg' : log_reg_pred})
overall_pred = pred_df.apply(lambda x: x.value_counts().index[0] if x.value_counts().iloc[0] > 1 else majority_class, axis = 1)

adult.loc[(adult['native-country'].values == '?'),'native-country'] = overall_pred.values

adult["workclass"] = adult["workclass"].cat.remove_categories("?")
adult['occupation'] = adult['occupation'].cat.remove_categories('?')
adult['native-country'] = adult['native-country'].cat.remove_categories('?')

income_mapping = {"<=50K": -1, ">50K": 1}

adult_data = adult.drop(columns=["income"])
adult_label = adult.income.map(income_mapping)

adult_cat_onehot = pd.get_dummies(adult_data.select_dtypes("category")).astype(int)
adult_non_cat = adult_data.select_dtypes(exclude="category")
adult_data_one_hot = pd.concat([adult_non_cat, adult_cat_onehot], axis=1, join="inner")

sensitive_attributes = adult_data_one_hot["gender_Male"]

xz_train, xz_test, y_train, y_test, z_train, z_test = train_test_split(
    np.array(adult_data_one_hot), np.array(adult_label), np.array(sensitive_attributes),
    test_size=0.4, random_state=42
)
xz_valid, xz_test, y_valid, y_test, z_valid, z_test = train_test_split(
    xz_test, y_test, z_test, test_size=0.5, random_state=42
)

xz_train = torch.FloatTensor(xz_train)
y_train = torch.FloatTensor(y_train)
z_train = torch.FloatTensor(z_train)

xz_valid = torch.FloatTensor(xz_valid)
y_valid = torch.FloatTensor(y_valid)
z_valid = torch.FloatTensor(z_valid)

xz_test = torch.FloatTensor(xz_test)
y_test = torch.FloatTensor(y_test)
z_test = torch.FloatTensor(z_test)

xz_train, xz_valid = normalize_data(xz_train, z_train, y_train, xz_valid, z_valid)
xz_train, xz_test = normalize_data(xz_train, z_train, y_train, xz_test, z_test)

full_tests = []
dp_tests = []

train_data = CustomDataset(xz_train, y_train, z_train)

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#seeds = [2]

for seed in seeds:
    print("< Seed: {} >".format(seed))
    
    model = nn.Sequential(
        nn.Linear(105, 32),
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
    x1 = xz_train[z_train == 0.0]
    x2 = xz_train[z_train == 1.0]
    
    x1 = x1 - torch.mean(x1, dim=0).unsqueeze(0)
    x2 = x2 - torch.mean(x2, dim=0).unsqueeze(0)
    
    xxt = (1 / (x1.size()[0] - 1)) * torch.mm(x1.T, x1) - (1 / (x2.size()[0] - 1)) * torch.mm(x2.T, x2)
    print("xxt = {}".format(xxt))
    try:
        s = torch.linalg.cholesky(xxt + torch.eye(xxt.size()[0]) * 10)
    except:
        s = torch.linalg.cholesky(-xxt + torch.eye(xxt.size()[0]) * 10)
    ws = torch.mm(weight, s)
    u, sig, v = torch.linalg.svd(ws, full_matrices=False)
    print("sig = {}".format(sig))
    print("sig summation = {}".format(torch.sum(sig ** 4)))
    
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
    lam = get_lambda_(k.clone().detach().numpy(), sig.clone().detach().numpy(), (torch.sum(sig ** 4) / 100).item())
    sig_fair = torch.zeros(sig.size()[0])
    for i in range(sig.size()[0]):
        sig_fair[i] = get_sigma_fair(sig[i], k[i], lam)
    print("sig_fair = {}".format(sig_fair))
    print("summation sig_fair = {}".format(torch.sum(sig_fair ** 4)))
    
    sig[0] = sig[0] * 4 / 8
    #print("sig summation after SVD = {}".format(torch.sqrt(torch.sum(sig_fair ** 2))))
    #print("sig sum = {}".format(torch.sqrt(torch.sum(sig ** 2))))
    ws = torch.mm(u, torch.mm(torch.diag(sig), v))
    #ws = torch.mm(u, torch.mm(torch.diag(sig_fair), v))
    weight = torch.mm(ws, torch.linalg.inv(s))
    
    model[0].weight.data = weight
    
    dp_test = test_model(model, xz_test, y_test, z_test)
    dp_tests.append(dp_test)
    
    """
    out_x1 = model[2](model[1](model[0](x1)))
    out_x2 = model[2](model[1](model[0](x2)))
    diff = torch.sqrt(torch.sum((out_x1 - out_x2) ** 2))
    print("diff = {}".format(diff))
    """
    
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