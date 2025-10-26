from torch.utils.data import DataLoader
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.models import EEGNet


dataset = DEAPDataset(root_path='./data_preprocessed_python',
                      online_transform=transforms.Compose([
                          transforms.To2d(),
                          transforms.ToTensor(),
                      ]),
                      label_transform=transforms.Compose([
                          transforms.Select('valence'),
                          transforms.Binary(5.0),
                      ]))

model = EEGNet(chunk_size=128,
               num_electrodes=32,
               dropout=0.5,
               kernel_1=64,
               kernel_2=16,
               F1=8,
               F2=16,
               D=2,
               num_classes=2)

x, y = next(iter(DataLoader(dataset, batch_size=64)))
model(x)