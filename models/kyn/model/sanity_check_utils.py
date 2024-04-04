import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import numpy as np
import os


def save_img_sem_mask(img, sem_mask, path):
    # img: b nv c h w
    # sem: b c h w


    h, w = img.shape[-2:]
    total_img = Image.new('RGB', (w, h*2))
    img = (img * 0.5 + 0.5)[0,0].permute(1,2,0).cpu().numpy()
    pil_img = Image.fromarray((img*255.0).astype(np.uint8))
    
    # write sem maps
    vl_seg_mask = sem_mask[0,0].cpu().numpy()
    new_palette = get_new_pallete(21)
    seg_color = get_new_mask_pallete(vl_seg_mask, new_palette)
    pil_seg = Image.fromarray(seg_color.permute(1,2,0).numpy().astype(np.uint8))

    total_img.paste(pil_img, (0, 0))
    total_img.paste(pil_seg, (0, h))
    files = os.listdir(path)
    ids = len(files)
    
    total_img.save(os.path.join(path, f"res_{ids}.png"))
    print("image saved!")



def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete

def get_new_mask_pallete(npimg, new_palette):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(new_palette)
    out_img = out_img.convert("RGBA")
    return torch.from_numpy(np.array(out_img)[:,:,:3]).permute(2, 0, 1).type(torch.uint8)