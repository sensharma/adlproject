import torch  # , torchvision
# from torchvision import datasets, transforms
from torch import optim
from torch.nn import functional as F

from models import cnn
import os
import data.dataloaders as dl
from paths import MNIST_CNN_PATH as PATH
# batch_size = 128

# PATH = os.path.abspath('./saved_models/test_exp')
num_epochs = 20
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

train_loader = dl.mnist_loader(train=True, train_limit=60000, batch_size=128)
test_loader = dl.mnist_loader(train=False, test_limit=10000, batch_size=128)
model = cnn.Net().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def save_mod_opt(epoch, loss, path=PATH):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, f'{path}/checkpoint.{epoch}')


def train(model, device, train_loader, optimizer, epoch, save=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data/255, target/255
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output.log(), target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if save:
            save_mod_opt(epoch, loss)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data/255, target/255
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            # test_loss += F.nll_loss(output.log(), target).item()
            test_loss += F.nll_loss(output, target).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
