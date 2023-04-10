import os.path

from torch.utils.tensorboard import SummaryWriter
from ReadData import BoneData
from torch import device, nn, cuda, optim
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from DynamicPoseModule import *

# could be Dynamic or Start
mode = "Dynamic"
root_dir = ""
if mode == 'Dynamic':
    root_dir = "DynamicData"
else:
    root_dir = "StartData"

device = device("cuda" if cuda.is_available() else "cpu")
pose = "Smash"
batch_size = 4
num_epoch = 50

ds = BoneData(root_dir, pose)
train_rt, val_rt = 0.7, 0.1
train_len, val_len = round(train_rt * len(ds)), round(val_rt * len(ds))
test_len = len(ds) - train_len - val_len
train_ds, val_ds, test_ds = random_split(ds, [train_len, val_len, test_len])

train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

if mode == 'Dynamic':
    poseModule = DynamicPoseModule().to(device)
else:
    poseModule = StartPoseModule().to(device)

loss_fn = nn.MSELoss().to(device)

learning_rate = 1e-2
optimizer = optim.SGD(poseModule.parameters(), lr=learning_rate)

writer = SummaryWriter("logs")

train_step = 0

for epoch in range(num_epoch):
    print(f"number of epoch: {epoch + 1}")

    # training
    for in_data, target in train_dl:
        in_data = in_data.to(device)
        target = target.to(device)

        output = poseModule(in_data)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1
        if train_step % 10 == 0:
            print(f"step:{train_step}, Loss: {loss.item()}")
        writer.add_scalar("train_loss", loss.item(), train_step)

    # validating
    total_val_loss = 0
    with torch.no_grad():
        for in_data, target in val_dl:
            in_data = in_data.to(device)
            target = target.to(device)

            output = poseModule(in_data)
            loss = loss_fn(output, target)
            total_val_loss += loss.item()

    print(f"total validation loss: {total_val_loss}")
    writer.add_scalar("validation_loss", total_val_loss, epoch + 1)
    print("")

    try:
        os.makedirs(os.path.join("trained_models_" + mode, pose))
    except:
        pass
    torch.save(poseModule.state_dict(), os.path.join("trained_models_" + mode, pose, f"poseModule{epoch + 1}.pth"))

total_test_loss = 0
with torch.no_grad():
    for in_data, target in test_dl:
        in_data = in_data.to(device)
        target = target.to(device)

        output = poseModule(in_data)
        loss = loss_fn(output, target)
        total_test_loss += loss.item()
print(f"total test loss: {total_test_loss}")