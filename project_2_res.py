import os , shutil
from pathlib import Path 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.models import ResNet50_Weights,resnet50
import torch.optim as optim 
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm.auto import tqdm
import torch , torchvision
from PIL import Image
import cv2



transform = transforms.Compose(
    [
        transforms.Resize((224,225)),
        transforms.ToTensor()
    ]
)


train_image_path = Path(r"datasets\MVTech\transistor\train")

dataset = ImageFolder(root = train_image_path,transform=transform,target_transform=None)
classes,class_id = dataset.classes,dataset.class_to_idx
print(classes,class_id)
train_dataset,test_dataset = torch.utils.data.random_split(dataset,[0.8,0.2])

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle = True)


class Feature_Extract(nn.Module):
    def __init__(self):
        super(Feature_Extract,self).__init__()
        self.model = resnet50(weights = ResNet50_Weights.DEFAULT)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad=True
        
        def hook(module,input,output)->None:
            self.features.append(output)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
    
    def forward(self,input):
        self.features = []
        if input.dim() == 3:
            input = input.unsqueeze(0) 
        with torch.no_grad():
            _ = self.model(input)

        self.avg = torch.nn.AvgPool2d(3,stride=1)
        fmap_size = self.features[0].shape[-2]
        self.resize = nn.AdaptiveAvgPool2d(fmap_size)
        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps,1)
        return patch
    

image = Image.open(r"D:\internship\Project_2\datasets\MVTech\transistor\test\cut_lead\003.png")
image = transform(image).unsqueeze(0)

res_model = Feature_Extract()
feature = res_model(image)

print(res_model.features[0].shape)
print(res_model.features[1].shape)
print(res_model.features)
plt.imshow(image[0].permute(1,2,0))
plt.show()


indices = torch.randperm(64)[:10]
print(indices)
fig,axes = plt.subplots(2,5,figsize = (15,6))
for i , idx in enumerate(indices):
    row = i//5
    col = i%5
    axes[row,col].imshow(feature[0,idx].detach().cpu(),cmap = 'grey')
    axes[row,col].set_title(f"Feature map{idx}")
    axes[row,col].axis('off')
plt.tight_layout()
# plt.show()


class AutoEncoder(nn.Module):
    def __init__(self,in_channels = 1000,latent_dim =  50 ,is_bn = True):
        super(AutoEncoder,self).__init__()
        layers = []
        #encoder 
        layers+=[nn.Conv2d(in_channels,(in_channels+2*latent_dim)//2 ,kernel_size=1,stride=1,padding=0)]
        if is_bn:
            layers+=[nn.BatchNorm2d(num_features=(in_channels+2*latent_dim)//2)]   
        layers+=[nn.ReLU()]
        layers+=[nn.Conv2d((in_channels+2*latent_dim)//2,2*latent_dim,kernel_size=1,stride=1,padding=0)]
        if is_bn:
            layers+=[nn.BatchNorm2d(num_features=2*latent_dim)]
        layers+=[nn.ReLU()]
        layers+=[nn.Conv2d(2*latent_dim,latent_dim,kernel_size=1,stride=1,padding=0)]
        self.encoder = nn.Sequential(*layers)

        #decoder
        layers = []
        layers+=[nn.Conv2d(latent_dim,2*latent_dim,kernel_size=1,stride=1 ,padding = 0)]
        if is_bn:
            layers+=[nn.BatchNorm2d(num_features=(2*latent_dim))]
        layers+=[nn.ReLU()]
        layers+=[nn.Conv2d(2*latent_dim,(in_channels+2*latent_dim)//2,kernel_size=1,stride=1,padding = 0)]
        if is_bn:
            layers+=[nn.BatchNorm2d((in_channels+2*latent_dim)//2)]
        layers+=[nn.ReLU()]
        layers+=[nn.Conv2d((in_channels+2*latent_dim)//2,in_channels,kernel_size=1,stride=1,padding = 0)]
        self.decoder = nn.Sequential(*layers)
    def forward(self,x):
        return self.decoder(self.encoder(x))
    
model = AutoEncoder(in_channels=1536,latent_dim=100).cuda()
res_model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(params = model.parameters(),lr  = 0.001)
checkpoint = torch.load('project_2_model_weights.pth')
model.load_state_dict(checkpoint)
model.eval()

# loss = []
# validation_loss =[]
# epochs = 100
# for epoch in tqdm(range(epochs)):
#     model.train()
#     for data,_ in train_loader:
#         with torch.no_grad():
#             features = res_model(data.cuda())
#         output = model(features)
#         train_loss = criterion(output,features)
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()
#     loss.append(train_loss.item())
#     model.eval()
#     with torch.inference_mode():
#         total_val_loss = 0.0
#         num_batches = 0
#         for data,_ in test_loader:
#             features = res_model(data.cuda())
#             output = model(features)
#             val_loss = criterion(output,features)
#             total_val_loss+=val_loss.item()
#             num_batches+=1
#         avg_val_loss = total_val_loss/num_batches
#         validation_loss.append(avg_val_loss)
#     if epoch%10 ==0 :
#         print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, epochs, train_loss.item(), avg_val_loss))


# plt.plot(loss, label='Training Loss')
# plt.plot(validation_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# torch.save(model.state_dict(),'project_2_model_weights.pth')

image = Image.open(r'D:\internship\Project_2\datasets\MVTech\transistor\test\cut_lead\003.png')
image = transform(image)
with torch.inference_mode():
    features = res_model(image.cuda())
    recon  = model(features)
recon_error =  ((features-recon)**2).mean(axis=(1)).unsqueeze(1)
segmentation_map = nn.functional.interpolate(
    recon_error,
    size = (224,224),
    mode='bilinear'
)

plt.imshow(segmentation_map.squeeze().cpu().numpy(),cmap= 'jet')
plt.show()


def decision_function(segmentation_map):  

    mean_top_10_values = []

    for map in segmentation_map:
        flattened_tensor = map.reshape(-1)
        sorted_tensor, _ = torch.sort(flattened_tensor,descending=True)
        mean_top_10_value = sorted_tensor[:10].mean()
        mean_top_10_values.append(mean_top_10_value)
    return torch.stack(mean_top_10_values)

model.eval()

RECON_ERROR=[]
for data,_ in train_loader:
    
    with torch.no_grad():
        features =  res_model(data.cuda()).squeeze()
        recon = model(features)
    segmentation_map =  ((features-recon)**2).mean(axis=(1))[:,3:-3,3:-3]
    anomaly_score = decision_function(segmentation_map)
    RECON_ERROR.append(anomaly_score)
RECON_ERROR = torch.cat(RECON_ERROR).cpu().numpy()
best_threshold = np.mean(RECON_ERROR) + 3 * np.std(RECON_ERROR)

heat_map_max, heat_map_min = np.max(RECON_ERROR), np.min(RECON_ERROR)


model.eval()
res_model.eval()

test_path = Path(r'D:\internship\Project_2\datasets\MVTech\transistor\test')

for path in test_path.glob('*/*.png'):
    fault_type = path.parts[-2]
    test_image = transform(Image.open(path)).cuda().unsqueeze(0)
    
    with torch.no_grad():
        features =  res_model(test_image)
        recon = model(features)
    
    segmentation_map = ((features - recon)**2).mean(axis=(1))
    y_score_image = decision_function(segmentation_map=segmentation_map)
    
    y_pred_image = 1*(y_score_image >= best_threshold)
    class_label = ['OK','NOTOK']

    if fault_type in ['cut_lead']:

        plt.figure(figsize=(15,5))

        plt.subplot(1,3,1)
        plt.imshow(test_image.squeeze().permute(1,2,0).cpu().numpy())
        plt.title(f'fault type: {fault_type}')

        plt.subplot(1,3,2)
        heat_map = segmentation_map.squeeze().cpu().numpy()
        heat_map = heat_map
        heat_map = cv2.resize(heat_map, (128,128))
        plt.imshow(heat_map, cmap='jet', vmin=heat_map_min, vmax=heat_map_max) 
        plt.title(f'Anomaly score: {y_score_image[0].cpu().numpy() / best_threshold:0.4f} || {class_label[y_pred_image]}')

        plt.subplot(1,3,3)
        plt.imshow((heat_map > best_threshold), cmap='gray')
        plt.title(f'segmentation map')
        
        plt.show()
