from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import cv2
def test_model(model, criterion):
    since = time.time()
    model.train(False)  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    i=-1
    for data in dataloders:
        i=i+1
        # get the inputs
        inputs, label = data
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(label.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)
        print(i)
        for j in range(labels.size()[0]):
            p=preds[j]
            t=label[j]
            if(p!=t):
                a=inputs[j].cpu().data.numpy()
                a=a.transpose((1,2,0))
                print(image_datasets.imgs[i])
                #cv2.imwrite('a.jpg',a)
        #print('label')
        #print([class_names[label[j]] for j in range (labels.size()[0])])
        #print('predict')
        #print([class_names[preds[j]] for j in range(preds.size()[0])])
    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects / dataset_sizes
    print(' Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Test Acc: {:4f}'.format(epoch_acc))


if __name__ == '__main__':

    # data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'test': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # your image data file
    data_dir = './data'
    image_datasets =datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                              data_transforms['val']) 
    # wrap your data and label into Tensor
    print(image_datasets.imgs)
    dataloders = torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=4)

    dataset_sizes = len(image_datasets) 
    class_names = image_datasets.classes
    print(class_names)
    # use gpu or not
    use_gpu = True
    #torch.cuda.is_available()

    # get model and replace the original fc layer with your fc layer
    model_ft = torch.load('best_model_res.pkl')
    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    #test_model(model=model_ft,
                           #criterion=criterion)
