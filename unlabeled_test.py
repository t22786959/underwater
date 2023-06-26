import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
from adamp import AdamP
# my import
from model import AIMnet
from dataset_all import TestData ,collate_wrapper
from utils import *
import pyiqa

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


bz = 2
# model_root = 'pretrained/model.pth'
model_root = 'model/ckpt-first-train-finished/model_e200.pth'
Dataset_name = 'Seathru'

input_root = 'data/test/'+ Dataset_name +'/'
save_path = 'result/Seathru'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
checkpoint = torch.load(model_root)
Mydata_ = TestData(input_root)


data_load = data.DataLoader(Mydata_, batch_size=bz)
    
model = AIMnet().cuda()
model = nn.DataParallel(model, device_ids=[0, 1])
optimizer = AdamP(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_dict'])
epoch = checkpoint['epoch']
model.eval()
print('START!')


iqa_metric = pyiqa.create_metric('musiq', as_loss=True).cuda()
score_total =  0


if 1:
    print('Load model successfully!')
    for data_idx, data_ in enumerate(data_load):
        data_input, data_la = data_
        
        data_input = Variable(data_input).cuda()
        data_la = Variable(data_la).cuda()
        print(data_idx)

        if len(data_input) == 1:
            break
        for i in range(bz):
            with torch.no_grad():
                    
                a=data_input[i]
                b=data_la[i]
                       
                if Dataset_name == 'Seathru':
                    a1 = a[:,:400,:600]
                    b1 = b[:,:400,:600]
                    a2 = a[:,400:,:600]
                    b2 = b[:,400:,:600]
                    a3 = a[:,:400,600:]
                    b3 = b[:,:400,600:]
                    a4 = a[:,400:,600:]
                    b4 = b[:,400:,600:]
                    
                    a1=torch.reshape(a1,(1,3,400,600))
                    b1=torch.reshape(b1,(1,3,400,600))
                    a2=torch.reshape(a2,(1,3,400,600))
                    b2=torch.reshape(b2,(1,3,400,600))
                    a3=torch.reshape(a3,(1,3,400,600))
                    b3=torch.reshape(b3,(1,3,400,600))
                    a4=torch.reshape(a4,(1,3,400,600))
                    b4=torch.reshape(b4,(1,3,400,600))
                    result1, _ = model(a1, b1)
                    result2, _ = model(a2, b2)
                    result3, _ = model(a3, b3)
                    result4, _ = model(a4, b4)
                    
                    result = torch.cat((result1,result2), 2)
                    result_temp = torch.cat((result3,result4), 2)
                    result = torch.cat((result,result_temp), 3)
                    
                    score = iqa_metric(result).detach().cpu().numpy()
                    score_total += score
                
                    temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
                    temp_res[temp_res > 1] = 1
                    temp_res[temp_res < 0] = 0
                    temp_res = (temp_res*255).astype(np.uint8)
                    temp_res = Image.fromarray(temp_res)
                    
                    name = Mydata_.A_paths[data_idx*2+i].split('/')[5]
                    temp_res.save('%s/%s' % (save_path, name))
                    
                else:
                    a=torch.reshape(a,(1,3,a.shape[1],a.shape[2]))
                    b=torch.reshape(b,(1,3,b.shape[1],b.shape[2]))
                    print(a.shape)
                    print(b.shape)
                    result, _ = model(a, b)
                    
                    score = iqa_metric(result).detach().cpu().numpy()
                    score_total += score
                    
                    name = Mydata_.A_paths[data_idx*2+i].split('/')[5]
                    print(name)
                    temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
                    temp_res[temp_res > 1] = 1
                    temp_res[temp_res < 0] = 0
                    temp_res = (temp_res*255).astype(np.uint8)
                    temp_res = Image.fromarray(temp_res)
                    temp_res.save('%s/%s' % (save_path, name))
          
        
        print('result saved!')

print(score_total/24)
print('finished!')


