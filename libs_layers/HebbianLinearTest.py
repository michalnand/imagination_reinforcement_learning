import torch
import torch.nn as nn
import HebbianLinear as Hebbian
import torchvision

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(Model, self).__init__()

        self.layers = [
                        Flatten(),
                        Hebbian.HebbianLinear(input_features, 256),
                        nn.ReLU(),
                        Hebbian.HebbianLinear(256, 64),
                        nn.ReLU(),
                        Hebbian.HebbianLinear(64, output_features)
        ]

        self.model = nn.Sequential(*self.layers)


        print(self.model)

    def forward(self, x):
        return self.model.forward(x)

    def reset(self):
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "reset"):
                self.layers[i].reset()



if __name__ == "__main__":

    batch_size = 128

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./dataset_files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./dataset_files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])), batch_size=batch_size, shuffle=True)


    input_features  = 28*28
    output_features = 10
    model = Model(input_features, output_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    accuraccy_filtered = 0
    epoch_count = 10
    for epoch in range(epoch_count):
        
        for batch_idx, (input_x, target_y_) in enumerate(train_loader):

            batch_size_ = input_x.shape[0]
            target_y = torch.zeros(batch_size_, output_features)
            target_y.scatter_(1, target_y_.unsqueeze(1), 1)
            
            predicted_y = model(input_x)
            loss = (target_y - predicted_y)**2
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hit  = torch.argmax(target_y, dim = 1) == torch.argmax(predicted_y, dim = 1)
            miss = torch.argmax(target_y, dim = 1) != torch.argmax(predicted_y, dim = 1)

            hit  = hit.sum().detach().numpy()
            miss = miss.sum().detach().numpy()

            
            accuraccy = 100.0*hit/(hit + miss)
            accuraccy_filtered = 0.9*accuraccy_filtered + 0.1*accuraccy
            
            print(epoch, batch_idx, loss, round(accuraccy_filtered, 2))

        model.reset()

    print("program done")