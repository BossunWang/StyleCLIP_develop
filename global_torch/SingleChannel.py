import os

import numpy as np
import torch
from tqdm import tqdm

from PIL import Image
import copy
from manipulate import Manipulator
import argparse

import sys
import clip

def GetImgF(out,model,preprocess):
    imgs=out
    imgs1=imgs.reshape([-1]+list(imgs.shape[2:]))
    
    tmp=[]
    for i in range(len(imgs1)):
        
        img=Image.fromarray(imgs1[i])
        image = preprocess(img).unsqueeze(0).to(device)
        tmp.append(image)
    
    image=torch.cat(tmp)
    with torch.no_grad():
        image_features = model.encode_image(image)
    
    image_features1=image_features.cpu().numpy()
    # print("imgs.shape[:2]:", imgs.shape[:2])
    # print("image_features:", image_features.shape)
    image_features1=image_features1.reshape(list(imgs.shape[:2])+[512])
    
    return image_features1

def GetFs(fs):
    tmp=np.linalg.norm(fs,axis=-1)
    fs1=fs/tmp[:,:,:,None] # fs1 size is (len(mindexs2) * num_c, num_imgs, alpha size, 512)
    fs2=fs1[:,:,1,:]-fs1[:,:,0,:]  # alpha[0]*sigma - alpha[1]* sigma
    fs3=fs2/np.linalg.norm(fs2,axis=-1)[:,:,None] # fs3 size is (len(mindexs2) * num_c, num_imgs, 512)
    fs3=fs3.mean(axis=1) # fs3 size is (len(mindexs2) * num_c, 512)
    fs3=fs3/np.linalg.norm(fs3,axis=-1)[:,None]
    return fs3


def GenFs3(file_path, network_pkl):
    os.makedirs(file_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    device = torch.device('cuda')
    M = Manipulator()
    M.device = device
    G = M.LoadModel(network_pkl, device)
    M.G = G
    M.SetGParameters()
    num_img = 100_000
    M.GenerateS(num_img=num_img)
    M.GetCodeMS()  # get dlatents mean and std

    # M=Manipulator(dataset_name=dataset_name)
    np.set_printoptions(suppress=True)
    # print(M.dataset_name)
    # %%
    img_sindex = 0  # select the first $num_images images
    num_images = 100
    dlatents_o = []
    tmp = img_sindex * num_images
    print("M.dlatents size:", len(M.dlatents))
    for i in range(len(M.dlatents)):
        print(M.dlatents[i].shape)
        tmp1 = M.dlatents[i][tmp:(tmp + num_images)]
        dlatents_o.append(tmp1)
    # %%

    all_f = []
    # M.alpha = [-5, 5]  # ffhq 5
    M.alpha = [-10, 10]  # ffhq 5
    M.step = 2
    M.num_images = num_images
    select = np.array(M.mindexs) <= 16  # below or equal to 128 resolution
    mindexs2 = np.array(M.mindexs)[select]
    print("mindexs2:", mindexs2)

    # edit every index on all channel
    for lindex in mindexs2: #ignore ToRGB layers
        print("lindex:", lindex)
        num_c=M.dlatents[lindex].shape[1]
        print("num_c:", num_c)
        for cindex in tqdm(range(num_c)):

            M.dlatents=copy.copy(dlatents_o)
            M.dlatents[lindex][:,cindex]=M.code_mean[lindex][cindex]

            M.manipulate_layers=[lindex]
            codes,out=M.EditOneC(cindex)
            # out size is (num_images, alpha size, w, h, c)
            image_features1=GetImgF(out,model,preprocess)
            # image_features1 size is (num_images, alpha size, 512)
            all_f.append(image_features1)

    all_f=np.array(all_f)

    fs3=GetFs(all_f)

    #%%
    np.save(file_path+'fs3',fs3)


#%%
if __name__ == "__main__":
    import sys
    sys.path.append("../")

    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--dataset_name',type=str,default='cat',
                    help='name of dataset, for example, ffhq')
    args = parser.parse_args()
    dataset_name=args.dataset_name
    '''
    #%%
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device,jit=False)
    #%%

    # file_path = './npy/ffhq_stylegan_ada/'
    # network_pkl = 'model/ffhq_stylegan_ada.pkl'
    # GenFs3(file_path, network_pkl)

    file_path = './npy/afhq_stylegan_ada/'
    network_pkl = 'model/afhqcat.pkl'
    GenFs3(file_path, network_pkl)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    