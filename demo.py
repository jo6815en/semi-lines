import sys
import torch
import cv2
import matplotlib.pyplot as plt
from albumentations import Normalize
from mlsd_pytorch.cfg.default import get_cfg_defaults
from mlsd_pytorch.models.build_model import build_model
from mlsd_pytorch.utils.decode import deccode_lines_TP



def do_one(model,imname, cfg):
    
    input_size = cfg.datasets.input_size
    thresh = cfg.decode.score_thresh
    topk = cfg.decode.top_k
    min_len = cfg.decode.len_thresh
    img0 = cv2.imread(imname)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    size0 = img0.shape
    img = cv2.resize(img0, (input_size, input_size))
    
    aug = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    img_norm = aug(image=img)['image']
    img_norm = img_norm.transpose((2,0,1))
    
    if torch.cuda.is_available():
        imt = torch.from_numpy(img_norm).cuda().unsqueeze(0)
    else:
        imt = torch.from_numpy(img_norm).unsqueeze(0)
    
    with torch.no_grad():
        output = model(imt)
        output = output[:, 7:, :, :]
        center_ptss, pred_lines, _, scores = \
            deccode_lines_TP(output, thresh, min_len, topk, 3)

        pred_lines = pred_lines.cpu().numpy()
        scores = scores.cpu().numpy()
        
           
        
        for l in pred_lines:
            # convert to original image
            xratio = 2 * size0[1] / input_size
            yratio = 2 * size0[0] / input_size          
            cv2.line(img0, (int(xratio * l[0]), int(yratio * l[1])), (int(xratio * l[2]), int(yratio * l[3])), (255, 165, 0), 2)
        plt.imshow(img0)
        plt.axis('off')
        plt.show()
        
      

def main(imname):
    # load model and set some parameters  
    cfg = get_cfg_defaults()
    cfg.model.model_name = 'mobilev2_mlsd'
    cfg.model.with_deconv = True
    cfg.decode.score_thresh = 0.1
    cfg.decode.len_thresh = 100
    modelchpt = 'exp/finnwoods/supervised/supervised_all/all/best_sAP.pth'
    
    if torch.cuda.is_available():
        model = build_model(cfg).cuda()
        checkpoint = torch.load(modelchpt, weights_only=False)
        model.load_state_dict(checkpoint['model'])
    else:
        model = build_model(cfg)
        checkpoint = torch.load(modelchpt,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
    
   
    # run model on an image
    model.eval()
    do_one(model,imname,cfg)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('./images/test.png')
    