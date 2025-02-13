import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

from numpy import dtype
from torch.utils.data import Dataset
import torch
import pandas
import os
from skimage import io
import cv2
from glob import glob

class graspDataSet(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pandas.read_csv(csv_file, header = None)
        print(f"Data: {self.annotations}")
        # print("Annotations: " + str(self.annotations.__len__()))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(image_path, as_gray=True)
        image = image.astype('float32')
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

def generate_images(directory, shape):
    if os.path.exists(os.path.join(directory, 'prepped/labels.csv')):
        return

    # Put all the files from different classes into their own globs for iteration
    files_yellow = glob(directory + '/yellow/*.jpg')
    files_orange = glob(directory + '/orange/*.jpg')

    # Put each class glob into an array for iteration
    files = [files_yellow, files_orange]
    classes = ['yellow', 'orange']

    classification = 0

    save_dir = os.path.join(directory, 'prepped')
    os.chdir(save_dir)
    fil = open(os.path.join(save_dir, 'labels.csv'), 'a')

    for file_list in files:
        i = 0
        for file in file_list:
            im_name = str('s' + classes[classification] + str(i) + '.jpg')
            if not os.path.exists(os.path.join(save_dir, im_name)):
                image = cv2.imread(file)
                image = cv2.resize(image, (shape,shape), interpolation=cv2.INTER_AREA)
                cv2.imwrite(im_name, image)
            fil.write(str('s' + classes[classification] + str(i) + '.jpg ,' + str(classification) + ',\n'))

            i += 1
        
        
        classification += 1

    fil.close()


'''
from glob import glob
import cv2

files_black = glob('Training_Boxes/images/boxes_train/black/*.jpg')
files_card = glob('Training_Boxes/images/boxes_train/card/*.jpg')
files_clear = glob('Training_Boxes/images/boxes_train/clear/*.jpg')
files_styro = glob('Training_Boxes/images/boxes_train/styro/*.jpg')

files = [files_black, files_card, files_clear, files_styro]

classification= 0

fil = open('trainlabels.csv', 'a')

for file_list in files:
    classification += 1

    for file in file_list:
        f = file.split("s")[-1]
        image = cv2.imread(file)

        cv2.imwrite('Training_Boxes/images/boxes_train/all/s'+f, image)
        fil.write('s'+f+','+str(classification)+',\n')

fil.close()
'''