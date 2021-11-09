import os
import json
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from alexnet import AlexNet

BATCH_SIZE = 64  # 论文128
LR = 0.0001  # 论文 0.01
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
EPOCHS = 10  # 论文90

DATASET_PATH = 'data'
MODEL = 'AlexNet.pth'


def train_device(device='cpu'):
    # 只考虑单卡训练
    if device == 'gpu':
        cuda_num = torch.cuda.device_count()
        if cuda_num >= 1:
            print('device:gpu')
            return torch.device(f'cuda:{0}')
    else:
        print('device:cpu')
        return torch.device('cpu')


def dataset_loader(dataset_path):
    dataset_path = os.path.join(os.getcwd(), dataset_path)
    assert os.path.exists(dataset_path), f'[{dataset_path}] does not exist.'
    train_dataset_path = os.path.join(dataset_path, 'train')
    val_dataset_path = os.path.join(dataset_path, 'val')
    # 训练集图片随机裁剪224x224区域，以0.5的概率水平翻转
    # 由于torchvision没有封装PCA jitter，所以用Corlor jitter模拟RGB通道强度的变化（不够严谨...）
    # alexnet中训练样本分布为零均值分布，这里采用了常用的均值为0方差为1的标准正态分布
    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(size=224),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'val': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=val_dataset_path, transform=data_transform['val'])
    return train_dataset, val_dataset


def idx2class_json(train_dataset):
    class2idx_dic = train_dataset.class_to_idx
    idx2class_dic = dict((val, key) for key, val in class2idx_dic.items())
    # json.dumps()把python对象转换成json格式的字符串
    json_str = json.dumps(idx2class_dic)
    with open('class_idx.json', 'w') as json_file:
        json_file.write(json_str)
    print('write class_idx.json complete.')


def evaluate_val_accuracy(net, val_dataset_loader, val_dataset_num, device=torch.device('cpu')):
    # ==============================================
    # isinstance()与type()区别：
    # type()不会认为子类是一种父类类型，不考虑继承关系。
    # isinstance()会认为子类是一种父类类型，考虑继承关系。
    # 如果要判断两个类型是否相同推荐使用isinstance()
    # ==============================================
    if isinstance(net, nn.Module):
        net.eval()
    val_correct_num = 0
    for i, (val_img, val_label) in enumerate(val_dataset_loader):
        val_img, val_label = val_img.to(device), val_label.to(device)
        output = net(val_img)
        _, idx = torch.max(output.data, dim=1)
        val_correct_num += torch.sum(idx == val_label)
    val_correct_rate = val_correct_num / val_dataset_num
    return val_correct_rate


def train(net, train_dataset, val_dataset, device=torch.device('cpu')):
    train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
    print(f'[{len(train_dataset)}] images for training, [{len(val_dataset)}] images for validation.')
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)  # 论文使用的优化器
    # optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # 学习率调整策略
    # 论文中，alexnet将错误率（应该指的是验证集）作为指标，当错误率一旦不再下降的时候降低学习率。alexnet训练了大约90个epoch，学习率下降3次
    # 第一种策略，每30个epoch降低一次学习率（不严谨）
    # lr_scheduler=optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1)
    # 第二种策略，错误率不再下降的时候降低学习率，我们后面会计算验证集的准确率，错误率不再下降和准确率不再提高是一个意思,所以mode为max，但是
    # 实测的时候
    # ==================================================================================================================
    # 注意：在PyTorch 1.1.0之前的版本，学习率的调整应该被放在optimizer更新之前的。如果我们在 1.1.0 及之后的版本仍然将学习率的调整
    # （即 scheduler.step()）放在 optimizer’s update（即 optimizer.step()）之前，那么 learning rate schedule 的第一个值将
    # 会被跳过。所以如果某个代码是在 1.1.0 之前的版本下开发，但现在移植到 1.1.0及之后的版本运行，发现效果变差，
    # 需要检查一下是否将scheduler.step()放在了optimizer.step()之前。
    # ==================================================================================================================
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=1,
                                                        min_lr=0.00000001)
    # 在训练的过程中会根据验证集的最佳准确率保存模型
    best_val_correct_rate = 0.0
    for epoch in range(EPOCHS):
        net.train()
        # 可视化训练进度条
        train_bar = tqdm(train_dataset_loader)
        # 计算每个epoch的loss总和
        loss_sum = 0.0
        for i, (train_img, train_label) in enumerate(train_bar):
            optimizer.zero_grad()
            train_img, train_label = train_img.to(device), train_label.to(device)
            output = net(train_img)
            loss = loss_function(output, train_label)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            train_bar.desc = f'train epoch:[{epoch + 1}/{EPOCHS}], loss:{loss:.5f}'
        # 测试验证集准确率
        val_correct_rate = evaluate_val_accuracy(net, val_dataset_loader, len(val_dataset), device)
        # 根据验证集准确率更新学习率
        lr_scheduler.step(val_correct_rate)
        print(
            f'epoch:{epoch + 1}, '
            f'train loss:{(loss_sum / len(train_dataset_loader)):.5f}, '
            f'val correct rate:{val_correct_rate:.5f}')
        if val_correct_rate > best_val_correct_rate:
            best_val_correct_rate = val_correct_rate
            # 保存模型
            torch.save(net.state_dict(), MODEL)
    print('train finished.')


if __name__ == '__main__':
    # 这里数据集只有5类
    alexnet = AlexNet(num_classes=5)
    device = train_device('gpu')
    train_dataset, val_dataset = dataset_loader(DATASET_PATH)
    # 保存类别对应索引的json文件，预测用
    idx2class_json(train_dataset)
    train(alexnet, train_dataset, val_dataset, device)
