from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.notmnist import NotMNIST
from data_loader.dermofit import Dermofit

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    provides transformed 32x32 grayscale images
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        print("Mnist is used")
        self.data_dir = data_dir + "MNIST/"
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
class CIFAR_10DataLoader(BaseDataLoader):
    """
    CIFAR-10 data loading demo using BaseDataLoader
    Provides RGB Images of 32x32
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        print("Cifar is used")
        self.data_dir = data_dir + "CIFAR10/"
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super(CIFAR_10DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class FashionMnistDataLoader(BaseDataLoader):
    """
    Fashion Mnist data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        print("Fashion Mnist is used")
        self.data_dir = data_dir + "FashionMNIST/"
        self.dataset = datasets.FashionMNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(FashionMnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
class NotMNISTDataLoader(BaseDataLoader):
    """
    NotMnist data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        print("NotMnist is used")
        self.data_dir = data_dir + "NotMNIST/"
        
        trsfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
       
        self.dataset = NotMNIST(self.data_dir, train = training, download = True, transform = trsfm)
        super(NotMNISTDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
          

class SVHNDataLoader(BaseDataLoader):
    """
    SVHN data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training='train'):
        print("SVHN is used")
        self.data_dir = data_dir + "SVHN/"
        
        trsfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        self.dataset = datasets.SVHN(self.data_dir, split = training, download = True, transform = trsfm)
        super(SVHNDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class DermofitDataLoader(BaseDataLoader):
    """
    Dermofit data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        print("Dermofit is used")
        self.data_dir = data_dir + "dermofit"
        
        trsfm_train = transforms.Compose([
                      #transforms.RandomHorizontalFlip(),
                      #transforms.RandomRotation(degrees=60),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      ])
        trsfm_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                     ])
        if (training == True):
            self.dataset = Dermofit(self.data_dir, train = training, transform = trsfm_train, process=True)
            super(DermofitDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        elif (training == False):
            self.dataset = Dermofit(self.data_dir, train = training, transform = trsfm_test, process=False)
            super(DermofitDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class HAMDataLoader(BaseDataLoader):
    """
    HAM data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        print("HAM is used")
        self.data_dir = data_dir + "HAM"
        
        trsfm_train = transforms.Compose([
                      #transforms.RandomHorizontalFlip(),
                      #transforms.RandomRotation(degrees=60),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      ])
        trsfm_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                     ])
        if (training == True):
            self.dataset = Dermofit(self.data_dir, train = training, transform = trsfm_train, process=True)
            super(HAMDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        elif (training == False):
            self.dataset = Dermofit(self.data_dir, train = training, transform = trsfm_test, process=False)
            super(HAMDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def channel_transformation(x):
    '''
    transforms a uni-channeled image into multi-channeled image
    '''
    return x.repeat(3, 1, 1)
        
        
        
        