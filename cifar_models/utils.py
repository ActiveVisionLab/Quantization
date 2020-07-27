import torchvision
import torchvision.transforms as transforms
import torch
from tqdm import tqdm


def train(model, epochs):
    model.train()

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=8
    )

    if not hasattr(model, "module"):
        if torch.cuda.device_count() >= 1:
            model = torch.nn.DataParallel(model)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"
        model.to(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    initial_lr = 0.0001  ## This is the last lr of the training script
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(
        model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4
    )

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(trainloader)):
            images = images.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()

            model.module.quantize()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

            if i % 100 == 0 and i != 0:
                tqdm.write(
                    "[%d, %d] loss: %.5f" % (epoch + 1, i + 1, running_loss / 500)
                )
                running_loss = 0.0

    return model


def test(model):
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=8
    )

    if not hasattr(model, "module"):
        if torch.cuda.device_count() >= 1:
            model = torch.nn.DataParallel(model)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"
        model.to(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.module.quantize()
    total, correct1, correct5 = 0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            sorted_logits = torch.argsort(outputs.data, 1, True)[:, :5]
            total += labels.size(0)
            correct1 += (predicted == labels).sum().item()
            correct5 += (sorted_logits == labels.unsqueeze(1)).sum().item()

    accuracy1 = correct1 / total
    accuracy5 = correct5 / total

    return (accuracy1, accuracy5)
