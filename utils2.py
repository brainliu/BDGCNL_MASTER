# _*_coding:utf-8_*_
# Name:Brian
# Create_time:2021/2/3 13:47
# file: utils2.py
# location:chengdu
# number:610000
import os
import numpy as np
import torch
import logging
import time
from metrics import *
import matplotlib.pyplot as plt
def get_logger(root, name=None, debug=True):
    time_lag=time.time()
    #when debug is true, show DEBUG and INFO in screen
    #when debug is false, show DEBUG in file and info in both screen&file
    #INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = os.path.join(root, 'run_%s.log'%time_lag)
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger


def train_epoch(train_loader, adj_mx,net,optimizer,epoch,logger,loss_criterion,device="cuda",log_step=10):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    epoch_training_losses = []
    for batch_idx, (X_data, y_batch) in enumerate(train_loader):
        net.cuda()
        net.train()
        optimizer.zero_grad()
        y_batch=y_batch.to(device)
        X_batch=X_data[:,0:1,:,:].permute(2,0,3,1).to(device)   #T B  N C z
        mask_missing=X_data[:,1:2,:,:].permute(2,0,3,1).to(device)
        out = net(X_batch,adj_mx,mask_missing)
        del X_batch
        loss = loss_criterion(out, y_batch)
        # loss = loss_criterion(out, y_batch.to(device))
        loss.backward()
        optimizer.step()
        del out
        del y_batch
        if batch_idx % log_step == 0:
            logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader), loss.item()))
        epoch_training_losses.append(loss.detach().cpu().numpy())
    train_epoch_loss=sum(epoch_training_losses) / len(epoch_training_losses)
    logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
    del train_loader
    return train_epoch_loss


def val_epoch(val_loader,net,adj_mx,epoch,logger,loss_criterion,device="cuda",log_step=20):
    epoch_val_losses = []
    with torch.no_grad():
        for batch_idx, (X_data, y_batch) in enumerate(val_loader):
            y_batch = y_batch.to(device)
            X_batch = X_data[:, 0:1, :, :].permute(2, 0, 3, 1).to(device)  # T B  N C z
            mask_missing = X_data[:, 1:2, :, :].permute(2, 0, 3, 1).to(device)
            out = net(X_batch,adj_mx,mask_missing)
            # y_batch=y_batch.to(device)
            val_loss = loss_criterion(out, y_batch)
            del y_batch
            del out
            if batch_idx % log_step == 0:
                logger.info('val Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, len(val_loader), val_loss.item()))
            epoch_val_losses.append(val_loss.detach().cpu().numpy())
        train_epoch_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        logger.info('**********val Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
        del val_loader
        return train_epoch_loss

def test_all(adj_mx, test_loader,model,max_speed,logger,epochs,device="cuda"):
    model.to(device)
    model.eval()
    mae_list = []
    mape_list= []
    rmse_list=[]
    with torch.no_grad():
        for batch_idx, (X_data, y_batch) in enumerate(test_loader):
            y_batch = y_batch.to(device)
            y_batch=y_batch*max_speed
            mask_missing = X_data[:, 1:2, :, :].permute(2, 0, 3, 1).to(device)
            X_batch = X_data[:, 0:1, :, :].permute(2, 0, 3, 1).to(device)  # T B  N C z
            output = model(X_batch,adj_mx,mask_missing)*max_speed


            # target_unnormalized = y_batch.detach().cpu().numpy()
            mae, rmse, mape, _, _ = All_Metrics(output, y_batch, None, 0.)
            logger.info("test_eopch_old:{}/{}:, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%".format(
                batch_idx,len(test_loader), mae, rmse, mape*100 ))

            mape_list.append(mape)
            mae_list.append(mae)
            rmse_list.append(rmse)
            logger.info("test_eopch_all index:{}/{}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%".format(
                batch_idx, len(test_loader), mae, rmse, mape*100))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.plot(mae_list, "b", label="mae_list")
        plt.plot(rmse_list, "g", label="rmse_list")
        ax1.set_ylabel('MSE and RMSE values')
        ax1.set_title("Result of test epochs-> %s "%epochs)
        ax1.set_ylim(0,100)
        ax2 = ax1.twinx()  # this is the important function
        plt.plot(mape_list, "r", label="MAPE")
        ax2.set_ylabel('MAPE values')
        ax2.set_ylim(0, 0.9)
        ax1.legend()
        ax2.legend()
        plt.show()
        mae_ave = sum(mae_list) / len(mae_list)
        mape_ave = sum(mape_list) / len(mape_list)
        rmse_ave = sum(rmse_list) / len(rmse_list)
        logger.info("test_eopch average： MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            mae_ave, rmse_ave, mape_ave*100))



