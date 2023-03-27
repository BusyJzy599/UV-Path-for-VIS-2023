from torch.utils.data import  Dataset
import torchvision.transforms as transforms
import h5py

from PIL import Image



class HDF5Dataset(Dataset):

    def __init__(self, path, test=False, train=False, val=False):
        self.file_path = path
        self.train = train
        self.test = test
        self.val = val
        self.dataset = None
        # file = h5py.File(path, "r")
        # self.dataset_len = len(file['imgs'])
        self.transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def __getitem__(self, index):
        # f = h5py.File(self.file_path, 'r')
        # for group in f.keys():
        #     print (group)
        if self.dataset is None:
            if self.train:
                self.train_img = h5py.File(self.file_path, 'r')['train_img']
                # print(self.train_img.shape)
                self.train_labels = h5py.File(self.file_path, 'r')['train_labels']
                # print(self.train_labels[1:10])
                train_cur_img = self.train_img[index]
                # path = self.paths[index].decode('UTF-8')
                train_PIL_image = Image.fromarray(np.uint8(train_cur_img)).convert('RGB')
                train_img = self.transformations(train_PIL_image)
                train_label = self.train_labels[index]

                return (train_img, train_label, index)
            if self.test:
                self.test_imgs = h5py.File(self.file_path, 'r')['test_img']
                self.test_labels = h5py.File(self.file_path, 'r')['test_labels']
                test_cur_img = self.test_imgs[index]
                # path = self.paths[index].decode('UTF-8')
                test_PIL_image = Image.fromarray(np.uint8(test_cur_img)).convert('RGB')
                test_img = self.transformations(test_PIL_image)
                test_label = self.test_labels[index]
                return (test_img, test_label,index)
            if self.val:
                self.val_img = h5py.File(self.file_path, 'r')['val_img']
                self.val_labels = h5py.File(self.file_path, 'r')['val_labels']
                val_cur_img = self.val_img[index]
                # path = self.paths[index].decode('UTF-8')
                val_PIL_image = Image.fromarray(np.uint8(val_cur_img)).convert('RGB')
                val_img = self.transformations(val_PIL_image)
                val_label = self.val_labels[index]
                return (val_img, val_label, index)

    def __len__(self):
        return self.dataset_len


class Preprocessor(Dataset):
    def __init__(self, dataset):
        super(Preprocessor, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
            return self._get_item(indices)

    def _get_item(self, index):
        x, label, index_,img_idx_,global_idx = self.dataset[index]
        return x, label-1, index_,img_idx_,global_idx


