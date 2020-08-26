#%% Housekeeping

import os, copy, random, shutil
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

torch.manual_seed(0)
print('Using PyTorch version', torch.__version__)

#%% Prepare training and test sets

class_names = ['normal', 'viral', 'covid']
root_dir = os.path.join(os.getcwd(),'COVID-19 Radiography Database')
source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']

if os.path.isdir(os.path.join(root_dir, source_dirs[1])): # if 'NORMAL' exists in root dir, make a test folder within root
    os.mkdir(os.path.join(root_dir, 'test'))

    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i])) # rename source dirs to class names

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c)) # make a dir for each class within test

    for c in class_names:   # load test dirs by class
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')] # for each .png in each source class dir
        selected_images = random.sample(images, 30) # sample 30 images 
        for image in selected_images: # for each image in the sample
            source_path = os.path.join(root_dir, c, image) # copy images over to test folder
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path) # move batch of files
            
#%% Initialize datasets

class cxr_dataset(torch.utils.data.Dataset): # in python, classes consist of both data and functions
    def __init__(self, image_dirs, transform): # boilerplate code, initialize source dirs + transform operations
        def get_images(class_name): # grab .pngs for input class and return images
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images
        
        self.images = {} # create empty set (unordered collection with no duplicates)
        self.class_names = ['normal', 'viral', 'covid']
        
        for class_name in self.class_names: # grab images for each class
            self.images[class_name] = get_images(class_name)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    def __len__(self): # No. of images in each class
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    
    def __getitem__(self, index): # for each image, convert to RGB, apply transformations, not sure what the index is for here
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)  

#%% Transformation parameters

# resize, modulate, convert to tensor, and normalize
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#%% Prepare training dataset       
train_dirs = { # dictionary to define dirpaths for each class
    'normal': 'COVID-19 Radiography Database/normal',
    'viral': 'COVID-19 Radiography Database/viral',
    'covid': 'COVID-19 Radiography Database/covid'
}
train_dataset = cxr_dataset(train_dirs, train_transform) # set training dataset

# Prepare testing dataset   
test_dirs = {
    'normal': 'COVID-19 Radiography Database/test/normal',
    'viral': 'COVID-19 Radiography Database/test/viral',
    'covid': 'COVID-19 Radiography Database/test/covid'
}
test_dataset = cxr_dataset(test_dirs, test_transform) # set testing dataset

# Set up data loaders
batch_size = 6
dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('Number of training batches', len(dl_train))
print('Number of test batches', len(dl_test))

#%% Show sample images
class_names = train_dataset.class_names

def show_images(images, labels, preds):
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0)) # rearrange x/y pixels to 0/2 dimensions
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean # unnormalize
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
            
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

images, labels = next(iter(dl_train))
show_images(images, labels, labels)

images, labels = next(iter(dl_test))
show_images(images, labels, labels)
#%% Load and modify pre-trained model

resnet18 = torchvision.models.resnet18(pretrained=True)
print(resnet18)
#resnet18.conv1.requires_grad = False
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3) # adjust I/Os for the last fc layer
loss_fn = torch.nn.CrossEntropyLoss() # define loss function
optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-4) # define Adam optimizer and learning rate

def show_preds():
    resnet18.eval()
    images, labels = next(iter(dl_test)) # grab images and true labels from the batch
    outputs = resnet18(images)
    _, preds = torch.max(outputs, 1) # torch.max second output is the predicted class
    show_images(images, labels, preds)
    
show_preds()
#%% Training

def train(epochs):
    print('Starting training..')
    for e in range(0, epochs):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)
        
        running_loss_history = []
        val_running_loss_history = []
        train_loss = 0.
        val_loss = 0.

        resnet18.train() # set model to training phase

        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad() # reset gradients
            outputs = resnet18(images) 
            loss = loss_fn(outputs, labels) # calculate loss function
            loss.backward() # backward propagation
            optimizer.step() # adjust weights
            
            train_loss += loss.item()
            running_loss_history.append(train_loss)     

            if train_step % 20 == 0: # print training progress every 20 epochs? 20 images?
                print('Evaluating at step', train_step)

                accuracy = 0

                resnet18.eval() # set model to eval phase

                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = resnet18(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    val_running_loss_history.append(val_loss)

                    _, preds = torch.max(outputs, 1)
                    accuracy += sum((preds == labels).numpy())

                val_loss /= (val_step + 1)
                accuracy = accuracy/len(test_dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

                show_preds()
                resnet18.train() # set back to training mode

                if accuracy >= 0.98:
                    print('Performance condition satisfied, stopping..')
                    best_model_wts = copy.deepcopy(resnet18.state_dict()) # save model weights
                    return best_model_wts, accuracy, e, running_loss_history, val_running_loss_history
        
        train_loss /= (train_step + 1)

    print(f'Training Loss: {train_loss:.4f}')
    print('Training complete..')

best_model, best_acc, no_epoch, training_loss, val_loss = train(epochs=10) # run training and return model weights, highest accuracy, # of epochs

resnet18.load_state_dict(best_model)
torch.save(resnet18, './covid_resnet18_epoch%d.pt' %no_epoch)
print('Saved trained model')
#%%
show_preds()

#%% Plot validation loss

plt.plot(training_loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.legend()