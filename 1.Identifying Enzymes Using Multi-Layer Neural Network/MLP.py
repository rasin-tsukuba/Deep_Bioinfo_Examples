import pickle
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# Construct Network
class MLP(torch.nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output, drop_p):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_outpupt = n_output
        self.drop_p = drop_p

        self.input_layer = torch.nn.Linear(self.n_input, self.n_hidden_1)
        self.dropout_layer_1 = torch.nn.Dropout(p=self.drop_p)
        self.hidden_layer_1 = torch.nn.Linear(self.n_hidden_1, self.n_hidden_2)
        self.dropout_layer_2 = torch.nn.Dropout(p=self.drop_p)
        self.output_layer = torch.nn.Linear(self.n_hidden_2, self.n_outpupt)

    def forward(self, x):
        x = self.dropout_layer_1(torch.nn.functional.relu(self.input_layer(x)))
        x = self.dropout_layer_2(torch.nn.functional.relu(self.hidden_layer_1(x)))
        x = F.softmax(self.output_layer(x))

        return x

def load_Pfam(name_list_pickle, model_name_list_pickle):
    with open(name_list_pickle, 'rb') as f:
        name_list = pickle.load(f)

    with open(model_name_list_pickle, 'rb') as f:
        model_list = pickle.load(f)

    encoding = []

    # Go through every name
    for i in range(len(name_list)):
        if i % 10000 == 0:
            print('Processing %dth sequence.' % i)
        # A sparce 16306D vector
        single_encoding = np.zeros(16306)
        if name_list[i]:
            for single_name in name_list[i]:
                # One-hot manner
                single_encoding[model_list.index(single_name)] = 1

        encoding.append(single_encoding)
    return encoding




def train(model, device, data, opt, e):
    model.train()

    for batch_idx, (feature, label) in enumerate(data):

        feature, label = feature.to(device), label.long().to(device)
        #print(feature.shape, feature.device, label.shape, label.device)
        opt.zero_grad()
        output = model.forward(feature)
        #print(output.shape, output.device, label)
        loss = F.nll_loss(output, label)
        loss.backward()
        opt.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e,
                batch_idx * len(feature),
                len(data.dataset),
                100. * batch_idx / len(data),
                np.exp(loss.item())))


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.long().to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        np.exp(test_loss), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    use_cuda = torch.cuda.is_available()
    print('Use CUDA:', use_cuda)
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    device = torch_device
    
    if device == "cpu":
        print("Using CPU")
    else:
        print("Using" , device)

    # how much data for training and how much data for testing
    test_ratio = 0.1
    # total number of classes, useful for define network structure
    number_class = 2
    # total number of feature, useful for define network structure
    number_features = 16306
    # stochastic gradient descent, training batch size
    batch_size = 32
    # training epoches
    epochs = 100

    # data loading
    enzyme_feature = load_Pfam('Pfam_name_list_new_data.pickle', 'Pfam_model_names_list.pickle')
    non_enzyme_feature = load_Pfam('Pfam_name_list_non_enzyme.pickle', 'Pfam_model_names_list.pickle')

    # make features
    features = np.concatenate(
        [enzyme_feature, non_enzyme_feature],
        axis=0
    )

    # make labels
    labels = np.concatenate(
        [np.ones([22168, 1]), np.zeros([22168, 1])],
        axis=0
    ).flatten().astype(np.int64)

    print("data loaded")
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=test_ratio,
                                                        random_state=0)

    print("train and test data prepared")
    tensor_x_train = torch.Tensor(x_train)  # transform to torch tensor
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)

    train_data = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(TensorDataset(tensor_x_test, tensor_y_test))
    print("dataloader done")
    # Instantiate
    model = MLP(n_input=number_features,
              n_hidden_1=batch_size // 2,
              n_hidden_2=batch_size,
              n_output=2,
              drop_p=0.3).cuda()
    print(model)
    print("model instantiated")
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        print("Epoch ", epoch, ":")
        train(model, device, data=train_data, opt=optimizer, e=epoch)
        test(model, device, test_data)

    torch.save(model.state_dict(), 'enzymes_classification.pt')
    print("model saved")

if __name__ == '__main__':
    main()
