import os

import sys
import cv2
import numpy as np
import torch
from PIL import Image
from models.sample_model import SampleFromPoseModel
from utils.language_utils import (generate_shape_attributes,
                                  generate_texture_attributes)
from utils.util import set_random_seed
from scipy.stats.kde import gaussian_kde
from utils.options import dict_to_nonedict, parse

def gaussian_noise_image(image):
    row, col, ch=image.shape
    mean=0
    sigma=50
    gauss=np.random.normal(mean, sigma, (row, col, ch))
    gauss=gauss.reshape(row, col, ch)
    noisy=image+gauss
    return noisy


def generate(poseFile, shapeText, textureText, step):
    opt = './configs/sample_from_pose.yml'
    opt = parse(opt, is_train=False)
    opt = dict_to_nonedict(opt)
    sample_model = SampleFromPoseModel(opt)

    # Load Pose Image
    pose_img=Image.open(poseFile)
    pose_img=np.array(pose_img.resize(size=(256, 512), resample=Image.LANCZOS))[:, :, 2:].transpose(2, 0, 1).astype(np.float32)
    pose_img=pose_img/12.-1
    pose_img=torch.from_numpy(pose_img).unsqueeze(1)
    sample_model.feed_pose_data(pose_img)

    # Generate Parsing
    file_txt=open(shapeText)
    shape_text=file_txt.read()
    shape_attributes=generate_shape_attributes(shape_text)
    shape_attributes=torch.LongTensor(shape_attributes).unsqueeze(0)
    sample_model.feed_shape_attributes(shape_attributes)
    sample_model.generate_parsing_map()
    sample_model.generate_quantized_segm()
    colored_segm=sample_model.palette_result(sample_model.segm[0].cpu())
    cv2.imwrite("parsing.png", colored_segm)

    # Generate Human
    file_txt=open(textureText)
    texture_text=file_txt.read()
    texture_attributes=generate_texture_attributes(texture_text)
    texture_attributes=torch.LongTensor(texture_attributes)
    sample_model.feed_texture_attributes(texture_attributes)
    sample_model.generate_texture_map()
    result=sample_model.sample_and_refine(200)
    result=result.permute(0, 2, 3, 1)
    result=result.detach().cpu().numpy()
    result*=255
    result=np.asarray(result[0, :, :, :], dtype=np.uint8)
    save_path='./evaluation_matrix/differentmodel/model2/'
    image_name=save_path+'model2_'+str(step)+"_4denim.png"
    cv2.imwrite(image_name, result[:, :, ::-1])

if __name__ == '__main__':
    print('Nice Try and Good Luck')
    poseFile=['./evaluation_matrix/poses/2mensit.png', './evaluation_matrix/poses/2womanopen.png', './evaluation_matrix/poses/2womenside.png', './evaluation_matrix/poses/2womenstand.png','./evaluation_matrix/poses/2womenstandstraight.png']
    shapeText=sys.argv[1]
    textureText=sys.argv[2]
    for i, posef in enumerate(poseFile):
        generate(posef, shapeText, textureText, i)
    print("Complete!")
