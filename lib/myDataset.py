from torch.utils.data.dataset import Dataset

class myDatasetClass(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = len(datasets)

    def __getitem__(self, index):
        img, label = self.datasets[index]                #PIL.Image(128, 512)
        img = img.mean(0)                                #(128,512)
        img = img.t()                                    #(512, 128)
        #img = 1 - img                                    #(512, 128)
        img.sub_(0.5).mul_(2)
        #print(img)
        return img, label

    def __len__(self):
        return self.length