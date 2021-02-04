# _*_coding:utf-8_*_
# Name:Brian
# Create_time:2021/2/1 16:49
# file: basci_train.py
# location:chengdu
# number:610000
import  torch.nn as nn
import matplotlib.pyplot as plt
from data_load import *
from GCNM import  *
from utils import *
# if torch.cuda.is_available():
#     torch.cuda.set_device(0)

root="./PEMS-BAY/"
if os.path.isdir(root) == False:
    os.makedirs(root, exist_ok=True)
    print("creat new files")
logger = get_logger(root, debug=False)
logger.info('Experiment log path in: {}'.format("./PEMS-BAY/"))


######model parameters

num_of_vertices= 325
num_of_features= 1
device="cuda"
speed_matrix=pd.read_hdf('./data/PEMS-BAY/pems-bay.h5')[1:30000]
mask_ones_proportion=0.5 #不缺失的值的比率
seq_len=12
pred_len=12
slide_length=8
shuffle=True
number_of_filters=42
train_propotion=0.6
valid_propotion=0.2
BATCH_SIZE=24
adj_data=np.load("./data/PEMS-BAY/pems-bay_adj.npy")
adj_two=construct_adj_double(adj_data,steps=2)
adj_two=torch.Tensor(adj_two).to(device) #转化为tensor
train_dataloader, valid_dataloader, test_dataloader, max_speed, X_mean=prepare_dataset(speed_matrix,mask_ones_proportion=mask_ones_proportion,seq_len=seq_len,BATCH_SIZE=BATCH_SIZE)
model_bdgcnm=BDGCNM_model(num_of_vertices,num_of_features,slide_length,seq_len,pred_len,number_of_filters).to(device)

optimizer = torch.optim.Adam(model_bdgcnm.parameters(), lr=1e-3)
# for para in model_bdgcnm.parameters():
#     print(para)
loss_criterion = nn.MSELoss()
training_losses = []
validation_losses=[]


best_model = None
best_loss = float('inf')
not_improved_count = 10
start_epoch = 3
epochs=20
RESUME=1
if RESUME:
    path_checkpoint = "./models/checkpoint/ckpt_best_%s.pth"%start_epoch  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点

    model_bdgcnm.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    training_losses = checkpoint["training_losses"]
    validation_losses = checkpoint["validation_losses"]

for epoch in range(start_epoch, epochs + 1):
    loss1=train_epoch(train_dataloader, adj_two, model_bdgcnm, optimizer, epoch, logger, loss_criterion, device="cuda", log_step=20)
    val_epoch_loss = val_epoch(valid_dataloader, model_bdgcnm, adj_two, epoch, logger, loss_criterion, device="cuda",
                               log_step=20)
    validation_losses.append(val_epoch_loss)
    training_losses.append(loss1)
    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        not_improved_count = 0
        best_state = True
    else:
        not_improved_count += 1
        best_state = False
        # save the best state
    if best_state == True:
        logger.info('*********************************Current best model saved!')
        best_model = copy.deepcopy(model_bdgcnm.state_dict())

    logger.info('**********Training loss: {},Validation loss: {}'.format(training_losses[-1],
                                                                         validation_losses[-1]))
    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.xlabel('epoches number')
    plt.ylabel('loss values')
    plt.title('the result of epoch: %s' % epoch)
    plt.legend()
    plt.show()

    checkpoint = {
        "net": model_bdgcnm.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
    }
    if not os.path.isdir("./models/checkpoint"):
        os.makedirs("./models/checkpoint")
    torch.save(checkpoint, './models/checkpoint/ckpt_best_%s.pth' % (str(epoch)))
    save_path = os.path.join(root, 'best_model.pth')
    torch.save(best_model, save_path)
    logger.info("Saving current best model to " + save_path)
    # print(loss1)

    test_all(adj_two, test_dataloader, model_bdgcnm, max_speed, logger, device="cuda")