from models.resnet import QuantizedResNet18, QuantizedResNet34, QuantizedResNet50, QuantizedResNet101, QuantizedResNet152
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from tqdm import tqdm

def train(model, batch_size=240, num_workers=8, dataset_folder_path='/home/marcelo/datasets/ILSVRC2012/ILSVRC2012_train'):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(dataset_folder_path, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    else:
        device = "cpu"
    
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    model = model.train()

    for epoch in range(2):
        running_loss = 0.0
        for i, (inputImage, target) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            target = target.to(device)
            inputImage = inputImage.to(device)

            model.module.quantize()

            outputs = model(inputImage)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0 and i != 0:
                tqdm.write("[%d, %d] loss: %.5f" % (epoch+1, i+1, running_loss/500))
                running_loss = 0.0
    
    model = model.module
    correct1, correct5, total = test(model)
    print(correct1/total, correct5/total)

def test(model, batch_size=2048, num_workers=8, dataset_folder_path='/home/marcelo/datasets/ILSVRC2012/ILSVRC2012_val'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(dataset_folder_path, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers = num_workers,
        pin_memory = True
    )

    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    else:
        device = "cpu"

    model = model.to(device)
    model.module.quantize()
    model.eval()
    correct1, correct5, total = 0, 0, 0
    with torch.no_grad():
        for i, (inputImage, target) in enumerate(tqdm(data_loader)):
            target = target.to(device)
            inputImage = inputImage.to(device)

            outputs = model(inputImage)
            _, predicted = torch.max(outputs.data, 1)
            a = torch.argsort(outputs.data, 1, True)[:, 0:5]

            total += target.size(0)
            correct1+=(predicted==target).sum().item()
            correct5+=(a==target.unsqueeze(1)).sum().item()

    return correct1, correct5, total


from DSConv.nn.dsconv2d import DSConv2d
def counting_dsconv(model):
    count = 0
    for m in model.modules():
        if isinstance(m, DSConv2d):
            count+=1
    return count

if __name__=="__main__":
    model = QuantizedResNet50(4, 32, pretrained=True)
    print(model)
    train(model)
    correct1, correct5, total = test(model)
    print(correct1/total, correct5/total)







