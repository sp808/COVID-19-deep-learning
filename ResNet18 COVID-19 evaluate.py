#%%

from __future__ import print_function
import torch, os, time
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sn

#%%

def find_sens_spec( covid_prob, noncovid_prob, thresh):
    sensitivity= (covid_prob >= thresh).sum()   / (len(covid_prob)+1e-10)
    specificity= (noncovid_prob < thresh).sum() / (len(noncovid_prob)+1e-10)
    print("sensitivity= %.3f, specificity= %.3f" %(sensitivity,specificity))
    return sensitivity, specificity

class_names = ['covid', 'viral', 'normal']

#%% Load trained model

model_name= os.path.join(os.getcwd(),'covid_resnet18_epoch0.pt') # fill in model filename here
model = torch.load(model_name, map_location='cpu') 
model.eval()

#%% Define image transofmrations and loader
imsize= 224
loader = transforms.Compose([transforms.Resize(imsize), 
                             transforms.CenterCrop(224), 
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])

def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # resnet returns an error if this isn't here
    return image

sm = torch.nn.Softmax(dim=1)

#%% Grab predicted probabilities for all samples
# Should probably convert this section to functions

test_covid_path = os.path.join(os.getcwd(), 'COVID-19 Radiography Database', 'covid')
test_pneum_path = os.path.join(os.getcwd(), 'COVID-19 Radiography Database', 'viral')
test_normal_path = os.path.join(os.getcwd(), 'COVID-19 Radiography Database', 'normal')
test_covid = []
test_pneum = []
test_normal = []

for x in os.listdir(test_covid_path): #os.listdir will generate an iterable list of files
    if x.lower().endswith('png'):
        test_covid.append(x)
        
for x in os.listdir(test_pneum_path): 
    if x.lower().endswith('png'):
        test_pneum.append(x)        
        
for x in os.listdir(test_normal_path): 
    if x.lower().endswith('png'):
        test_normal.append(x)

covid_pred= np.zeros([len(test_covid),1]).astype(int)
pneum_pred= np.zeros([len(test_pneum),1]).astype(int)
normal_pred  = np.zeros([len(test_normal),1]).astype(int)

covid_prob= np.zeros([len(test_covid),1])
pneum_prob= np.zeros([len(test_pneum),1])
normal_prob   = np.zeros([len(test_normal),1])

for i in range(len(test_covid)):
    cur_img= image_loader(os.path.join(test_covid_path, test_covid[i])) # load image
    model_output= model(cur_img) # evaluate image in trained model
    cur_pred = model_output.max(1, keepdim=True)[1] # grab prediction
    cur_prob = 1 - sm(model_output) # outputs seem to be inverted, not sure why, 1 - is a bandage fix
    covid_pred[i,:] = cur_pred.data.numpy()[0,0]
    covid_prob[i,:]= cur_prob.data.numpy()[0,0] 
#    print("%03d Covid predicted label:%s" %(i, class_names[int(cur_pred.data.numpy())]) )

for i in range(len(test_pneum)):
    cur_img= image_loader(os.path.join(test_pneum_path, test_pneum[i])) # load image
    model_output= model(cur_img) # evaluate image in trained model
    cur_pred = model_output.max(1, keepdim=True)[1] # grab prediction
    cur_prob = 1 - sm(model_output) # outputs seem to be inverted, not sure why, 1 - is a bandage fix
    pneum_pred[i,:] = cur_pred.data.numpy()[0,0]
    pneum_prob[i,:]= cur_prob.data.numpy()[0,0] 
#    print("%03d Covid predicted label:%s" %(i, class_names[int(cur_pred.data.numpy())]) )

for i in range(len(test_normal)):
    cur_img= image_loader(os.path.join(test_normal_path, test_normal[i]))
    model_output= model(cur_img)
    cur_pred = model_output.max(1, keepdim=True)[1]  
    cur_prob = 1 - sm(model_output) # outputs seem to be inverted, not sure why, 1 - is a bandage fix
    normal_pred[i,:] = cur_pred.data.numpy()[0,0]
    normal_prob[i,:]= cur_prob.data.numpy()[0,0]
#    print("%03d Non-Covid predicted label:%s" %(i, class_names[int(cur_pred.data.numpy())]) )
    
#%% Find sensitivity and specificity
thresh = 0.2
sensitivity_40, specificity= find_sens_spec( covid_prob, normal_prob, thresh)

# Draw ROC

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

y_test_res18= [1 for i in range(len(covid_prob))]+[0 for i in range(len(normal_prob))]
y_pred_res18= [covid_prob[i] for i in range(len(covid_prob))]+[normal_prob[i] for i in range(len(normal_prob))]

auc_res18 = roc_auc_score(y_test_res18, y_pred_res18)
ns_fpr_res18, ns_tpr_res18, _ = roc_curve(y_test_res18, y_pred_res18)

plt.figure()
pyplot.plot(ns_fpr_res18, ns_tpr_res18,  color='darkgreen', linewidth=2,   label='ResNet18,        AUC= %.3f' %auc_res18)
pyplot.ylim([0,1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title("ROC Curve")
# show the legend
pyplot.legend(loc='lower right')
plt.savefig('./ROC_covid19.png', dpi=200) 

#%% Confusion matrix
from sklearn.metrics import confusion_matrix

covid_list= [int(covid_pred[i]) for i in range(len(covid_pred))]
covid_count = [(x, covid_list.count(x)) for x in set(covid_list)]

pneum_list= [int(pneum_pred[i]) for i in range(len(pneum_pred))]
pneum_count = [(x, pneum_list.count(x)) for x in set(pneum_list)]

normal_list= [int(normal_pred[i]) for i in range(len(normal_prob))]
normal_count = [(x, normal_list.count(x)) for x in set(normal_list)]

y_pred_list= covid_list+normal_list+pneum_list
y_test_list= [2 for i in range(len(covid_list))]+[0 for i in range(len(normal_list))]+[1 for i in range(len(pneum_list))]

y_pred= np.asarray(y_pred_list, dtype=np.int64)
y_test= np.asarray(y_test_list, dtype=np.int64)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)   

# Plot normalized confusion matrix
df_cm = pd.DataFrame(cnf_matrix, index = [i for i in class_names],
                                 columns = [i for i in class_names])

ax = sn.heatmap(df_cm, cmap=plt.cm.RdPu, annot=True, cbar=True, fmt='g', xticklabels= ['Normal','CAP', 'COVID-19'], yticklabels= ['Normal','CAP', 'COVID-19'])
ax.set_title("Confusion matrix")
plt.savefig('./confusion_matrix.png', dpi=200) 
    
