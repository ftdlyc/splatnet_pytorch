import torch
from data.shapenet import ShapeNetDataset
from models.splatnet import SplatNetSegment

trainset = ShapeNetDataset('/opt/shapenetcore_partanno_segmentation_benchmark_v0', split='train', argumentation=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testset = ShapeNetDataset('/opt/shapenetcore_partanno_segmentation_benchmark_v0', split='test', argumentation=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)
valset = ShapeNetDataset('/opt/shapenetcore_partanno_segmentation_benchmark_v0', split='val', argumentation=False)
valloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)

net = SplatNetSegment(trainset.class_nums, trainset.category_nums)
for epcho in range(1, 400):
    net.fit(trainloader, epcho)
    if epcho % 20 == 0:
        net.score(valloader)
net.score(testloader)
torch.save(net.state_dict(), 'model.pkl')
