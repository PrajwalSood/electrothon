import os
import tensorflow as tf
from tensorflow import keras
import torch
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
import cv2
import sys
# sys.argv=['']
from PIL import Image
from PIL import ImageDraw
import torchvision.transforms as transforms
# import ACGPN.networks
from ACGPN.utils.transforms import get_affine_transform
from ACGPN.utils.transforms import transform_logits
from torch.autograd import Variable
print('loaded all successfuly')

from ACGPN.options.test_options import TestOptions
from ACGPN.models.models import create_model
import ACGPN.util as util

edge_model = keras.models.load_model('unet_resnet34.h5')
# pose_model = models.load_model('/content/drive/MyDrive/pose.h5')
seg_model = keras.models.load_model('unet_resnet34_body_parse.h5')

sys.argv = ['']

os.makedirs('sample', exist_ok=True)
opt = TestOptions().parse()

model = create_model(opt)

class general_pose_model(object):
    def __init__(self, modelpath):
        # Specify the model to be used
        #   Body25: 25 points
        #   COCO:   18 points
        #   MPI:    15 points
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.05
        self.pose_net = self.general_coco_model(modelpath)

    def general_coco_model(self, modelpath):
        self.points_name = {
            "Nose": 0, "Neck": 1, 
            "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, 
            "RHip": 8, "RKnee": 9, "RAnkle": 10, 
            "LHip": 11, "LKnee": 12, "LAnkle": 13, 
            "REye": 14, "LEye": 15, 
            "REar": 16, "LEar": 17, 
            "Background": 18}
        self.num_points = 18
        self.point_pairs = [[1, 0], [1, 2], [1, 5], 
                            [2, 3], [3, 4], [5, 6], 
                            [6, 7], [1, 8], [8, 9],
                            [9, 10], [1, 11], [11, 12], 
                            [12, 13], [0, 14], [0, 15], 
                            [14, 16], [15, 17]]
        prototxt   = os.path.join(
            modelpath, 
            'pose_deploy_linevec.prototxt')
        caffemodel = os.path.join(
            modelpath, 
            'pose_iter_440000.caffemodel')
        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return coco_model

    def predict(self, imgfile):
        img_cv2 = cv2.imread(imgfile)
        img_height, img_width, _ = img_cv2.shape
        inpBlob = cv2.dnn.blobFromImage(img_cv2, 
                                        1.0 / 255, 
                                        (self.inWidth, self.inHeight),
                                        (0, 0, 0), 
                                        swapRB=False, 
                                        crop=False)
        self.pose_net.setInput(inpBlob)
        self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        output = self.pose_net.forward()

        H = output.shape[2]
        W = output.shape[3]
        
        points = []
        for idx in range(self.num_points):
            probMap = output[0, idx, :, :] # confidence map.

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (img_width * point[0]) / W
            y = (img_height * point[1]) / H

            if prob > self.threshold:
                points.append(x)
                points.append(y)
                points.append(prob)
            else:
                points.append(0)
                points.append(0)
                points.append(0)

        return points

modelpath = 'ACGPN/pose'
pose_m = general_pose_model(modelpath)

transform_A = transforms.Compose([transforms.Resize([256, 192], Image.BICUBIC), transforms.ToTensor()])
transform_B = transforms.Compose([transforms.Resize([256, 192], Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
transform_l = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

def load_data(img_path, cloth_path):
    img = Image.open(img_path)
    img_t = torch.unsqueeze(transform_B(img.convert('RGB')),0).cuda()
    img_p = np.array(img.resize((256,256)))/255
    img_p = tf.expand_dims(img_p, 0)

    cloth = Image.open(cloth_path)
    cloth_t = torch.unsqueeze(transform_B(cloth.convert('RGB')),0).cuda()
    cloth_p = np.array(cloth.resize((256,256)))/255
    cloth_p = tf.expand_dims(cloth_p, 0)

    pred = seg_model.predict(img_p)
    label = torch.from_numpy(cv2.resize(pred[0], (192,256)))
    label = torch.argmax(label, -1, keepdims = True)
    label = torch.unsqueeze(label.permute(2,0,1), 0).float().cuda()

    pred = edge_model.predict(cloth_p)
    edge = torch.from_numpy(cv2.resize(pred[0], (192,256)))
    edge = torch.unsqueeze(edge, 0)
    edge = torch.unsqueeze(edge, 0)

    res_points = np.array(pose_m.predict(img_path)).reshape(-1,3)

    point_num = res_points.shape[0]
    pose_map = torch.zeros(point_num, 256, 192)
    r = 5
    im_pose = Image.new('L', (192, 256))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (192, 256))
        draw = ImageDraw.Draw(one_map)
        pointx = res_points[i, 0]
        pointy = res_points[i, 1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-r, pointy-r, pointx +
                            r, pointy+r), 'white', 'white')
            pose_draw.rectangle(
                (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        one_map = transform_B(one_map.convert('RGB'))
        pose_map[i] = one_map[0]
    pose = torch.unsqueeze(pose_map,0).cuda()

    mask = transform_A(img.convert('L'))
    mask_t = torch.unsqueeze(mask,0).cuda()

    data = {'label':label,
        'image': img_t,
        'path' : 'conten/ACGPN',
        'name' : img_path,
        'edge': edge.cuda(),
        'color': cloth_t,
        'mask': img_t,
        'colormask': img_t,
        'pose': pose}
    
    return data

def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch


def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], NC))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label


def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int)
    M_f = torch.FloatTensor(M_f).cuda()
    masked_img = img*(1-mask)
    M_c = (1-mask.cuda())*M_f
    M_c = M_c+torch.zeros(img.shape).cuda()  # broadcasting
    return masked_img, M_c, M_f


def compose(label, mask, color_mask, edge, color, noise):
    masked_label = label*(1-mask)
    masked_edge = mask*edge
    masked_color_strokes = mask*(1-color_mask)*color
    masked_noise = mask*noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise


def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((old_label.cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((old_label.cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((old_label.cpu().numpy() == 7).astype(np.int))
    label = label*(1-arm1)+arm1*4
    label = label*(1-arm2)+arm2*4
    label = label*(1-noise)+noise*4
    return label
# data = load_data('static/upload/1.jpg', 'static/upload/2.jpg')
# for j in data.keys():
#   if j == 'path' or j == 'name':
#     continue
#   else:
#     print(j)
#     print(torch.max(data[j]))
def tensor_to_image(img_tensor, grayscale=False):
    if grayscale:
        tensor = img_tensor.cpu().clamp(0, 255)
    else:
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

    try:
        array = tensor.numpy().astype('uint8')
    except:
        array = tensor.detach().numpy().astype('uint8')

    if array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.shape[0] == 3:
        array = array.swapaxes(0, 1).swapaxes(1, 2)

    return array

def save_image(image_numpy, image_path, grayscale=False):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_tensor_as_image(image_tensor, image_path, grayscale=False):
    image_numpy = tensor_to_image(image_tensor, grayscale)
    save_image(image_numpy, image_path, grayscale)

def load_output(ROOT):
    data = load_data(ROOT + '/static/upload/1.jpg', ROOT + '/static/upload/2.jpg')
    
    t_mask = torch.FloatTensor(
        (data['label'].cpu().numpy() == 7).astype(np.float)).cuda()*0
    data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
    mask_clothes = torch.FloatTensor(
        (data['label'].cpu().numpy() == 4).astype(np.int)).cuda()
    mask_fore = torch.FloatTensor(
        (data['label'].cpu().numpy() > 0).astype(np.int)).cuda()
    img_fore = data['image'] * mask_fore
    img_fore_wc = img_fore * mask_fore
    all_clothes_label = changearm(data['label'].cpu())

    ############## Forward Pass ######################
    fake_image, warped_cloth, refined_cloth = model(Variable(data['label'].cuda()), Variable(data['edge'].cuda()), Variable(img_fore.cuda()), Variable(
        mask_clothes.cuda()), Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()), Variable(data['pose'].cuda()), Variable(data['image'].cuda()), Variable(mask_fore.cuda()))

    save_tensor_as_image(fake_image[0], ROOT + '/static/gen/1.jpg')
    # save fake image
    # fake_image = fake_image.data.cpu()
    # fake_image = fake_image.clamp(min=0.1, max=0.9)
    # save_tensor_as_image(fake_image, 'static/gen/1.jpg')

    return 'static/gen/1.jpg'

# print(load_output())