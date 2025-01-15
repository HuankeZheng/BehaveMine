import torch
import torch.nn as nn
import os
import re
import numpy as np
import importlib

import src.dataloader as dataloader
import src.model as model

importlib.reload(model)
importlib.reload(dataloader)

model_name = 'BehaveMine'


def train(train_data, test_data, epoch, lr, batch_size, window_size, model_params, dataset_type):
    train_model = model.BehaveMine(input_dim=model_params['input_dim'], b_hidden_dim=model_params['b_hidden_dim'],
                                   s_hidden_dim=model_params['s_hidden_dim'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model = train_model.to(device)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr)

    train_dateset = dataloader.DataLoader(train_data, window_size, model_params['input_dim'])
    test_dataset = dataloader.DataLoader(test_data, window_size, model_params['input_dim'])
    train_loader = torch.utils.data.DataLoader(train_dateset, batch_size=batch_size, collate_fn=dataloader.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=dataloader.collate_fn)

    save_model_dir = f'result/{model_name}/{dataset_type}'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    criterion1 = nn.CrossEntropyLoss()

    best_loss = float('inf')
    flag = False

    best_epoch = 0
    for epoch in range(epoch):
        train_loss_all = 0
        loss_act_all = 0
        for input_data, act_target, lengths in train_loader:
            input_data = tuple(input_data.to(device) for input_data in input_data)
            act_target = act_target.to(device)

            optimizer.zero_grad()

            act_pred, _ = train_model(input_data, lengths)
            act_pred = act_pred.squeeze()

            loss = criterion1(act_pred, act_target)

            train_loss_all += loss.item()
            loss.backward()
            optimizer.step()
        train_loss_all /= len(train_loader)

        for input_data, act_target, lengths in test_loader:
            input_data = tuple(input_data.to(device) for input_data in input_data)
            act_target = act_target.to(device)

            optimizer.zero_grad()

            act_pred, _ = train_model(input_data, lengths)
            act_pred = act_pred.squeeze()

            loss = criterion1(act_pred, act_target)
            loss_act_all += loss.item()

        loss_act_all /= len(test_loader)

        loss_all = loss_act_all
        if loss_all < best_loss:
            best_loss = loss_all
            best_epoch = epoch
            best_model = train_model.state_dict()
            flag = True

        if epoch % 10 == 0:
            if flag:
                torch.save(best_model, save_model_dir + f'/loss{best_loss:.4f}.pth')
                print(f'epoch:{epoch}, best model, loss:{loss_all}')
                flag = False

        if epoch % 50 == 0:
            print(f'epoch:{epoch}, save model, loss:{loss_all}')
            torch.save(train_model.state_dict(), save_model_dir + f'/loss{loss_all:.4f}.pth')

        if epoch - best_epoch > 20:
            print(f'early stop')
            break
    return


def eval(eval_data, window_size, model_params, dataset_type):
    model_dir = f'result/{model_name}/{dataset_type}'
    loss_min = 100
    for filename in os.listdir(model_dir):
        match = re.search(r'loss(\d+\.\d+)', filename)
        if match is None:
            continue
        loss_value = float(match.group(1))
        if loss_value < loss_min:
            loss_min = loss_value
            filename_min = filename

    eval_model_dict = torch.load(model_dir + f"/{filename_min}")

    eval_model = model.BehaveMine(input_dim=model_params['input_dim'], b_hidden_dim=model_params['b_hidden_dim'],
                                  s_hidden_dim=model_params['s_hidden_dim'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_model = eval_model.to(device)
    eval_model.load_state_dict(eval_model_dict)

    eval_dataset = dataloader.DataLoader(eval_data, window_size, model_params['input_dim'])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, collate_fn=dataloader.collate_fn)

    record = []

    for input_data, act_target, lengths in eval_loader:
        input_data = tuple(input_data.to(device) for input_data in input_data)
        act_target = act_target.to(device)

        act_pred, embed = eval_model(input_data, lengths)
        act_target = act_target[:, -1, :].squeeze()
        act_pred = act_pred[:, -1, :].squeeze()
        embed = embed[:, -1, :].squeeze()

        act_target = act_target.cpu().detach().numpy()
        act_pred = act_pred.cpu().detach().numpy()
        embed = embed.cpu().detach().numpy()

        top5_indices = np.argsort(act_pred)[-5:]
        top5_values = act_pred[top5_indices]

        act_target = act_target.argmax(axis=0)
        act_pred = act_pred.argmax(axis=0)

        item = {'act_target': act_target, 'act_pred': act_pred, 'top5_indices': top5_indices,
                'top5_values': top5_values, 'embed': embed}
        record.append(item)

    return record
