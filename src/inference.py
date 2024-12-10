import os
import tqdm
import torch
import warnings
import argparse
import torchvision 
import numpy as np
from PIL import Image
import torch.nn.functional as F

from resnet import WPAL_res50
from densenet import WPAL_dense201
from head import head_1, head_2

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])

# 1 -> WPAL_res50_dr_3_2_1_train_0.07_alpha_0.01
# 2 -> WPAL_dense201_dr_3_2_1_train_0.07
# 3 -> WPAL_dense201_dr_3_2_peta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='Input images folder')
    parser.add_argument('--model', type=str, help='Model path', default='1')
    args = parser.parse_args()

    if args.model == '1':
        model = WPAL_res50()
        model.fc = head_1(model.fc.in_features)
        model.load_state_dict(torch.load('../models/WPAL_res50_dr_3_2_1_train_0.07_alpha_0.01.pt', map_location=device))

    elif args.model == '2':
        model = WPAL_dense201()
        model.fc = head_1(model.fc.in_features)
        model.load_state_dict(torch.load('../models/WPAL_dense201_dr_3_2_1_train_0.07.pt', map_location=device))
    
    # elif args.model == '3':
    #     model = WPAL_dense201()
    #     model.fc = head_1(model.fc.in_features)
    #     model.load_state_dict(torch.load('../models/WPAL_dense201_dr_3_2_peta.pt', map_location=device))

    images = os.listdir(args.folder)

    image_labels = {}

    ## Using tqdm for progress bar
    for image in tqdm.tqdm(images):
    
        img = Image.open(os.path.join(args.folder, image))
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(img)
            output = output.view(-1)

            vector_out = np.zeros(49)

            vector_out[torch.argmax(output[0:9])] = 1
            vector_out[9 + torch.argmax(output[9:18])] = 1
            vector_out[18 + torch.argmax(output[18:26])] = 1
            vector_out[26 + torch.argmax(output[26:33])] = 1
            vector_out[33 + torch.argmax(output[33:36])] = 1
            vector_out[36:39] = F.sigmoid(output[36:39])
        
            if (torch.argmax(output[39:41]) == 1):
                vector_out[39] = 1
            
            if (torch.argmax(output[41:45]) != 3):
                vector_out[40 + torch.argmax(output[41:45])] = 1
            
            vector_out[43 + torch.argmax(output[45:48])] = 1
            vector_out[46 + torch.argmax(output[48:51])] = 1
            

            vector_out = vector_out.astype(int)

            image_labels[int(image.rsplit('.',1)[0])] = vector_out
    
    # Sorting the dictionary
    image_labels = dict(sorted(image_labels.items()))

    with open('../submission.txt', 'w') as f:
        for key, value in image_labels.items():
            value = ' '.join([str(i) for i in value])
            f.write(f"{key} {value}\n")

            

            





        
    
    
    