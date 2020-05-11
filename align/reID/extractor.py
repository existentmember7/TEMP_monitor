import torch
import os
import numpy as np
from torch.nn.parallel import DataParallel
from torchvision import transforms
import cv2
from PIL import Image

from .aligned_reid.model.Model import Model
from .aligned_reid.utils.utils import set_devices
from .aligned_reid.utils.utils import load_state_dict
from .aligned_reid.dataset.PreProcessImage import PreProcessIm
from .aligned_reid.utils.ExtractFeature import ExtractFeature

from .aligned_reid_plus import models
from .aligned_reid_plus.util.FeatureExtractor import FeatureExtractor
from .aligned_reid_plus.util.utils import *

from .strong_baseline.modeling.baseline import Baseline
from .strong_baseline.config import cfg

from .utils.gradcam import GradCam
from .utils.misc_functions import save_class_activation_images

package_directory = os.path.dirname(os.path.abspath(__file__))


class StrongBaseline():
    def __init__(self, visualize=False):
        cfg.freeze()
        self.save_ids = 1
        self.visualize = visualize

        if cfg.MODEL.DEVICE == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

        self.model = Baseline(751, cfg.MODEL.LAST_STRIDE,
                              cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, 'after',
                              cfg.MODEL.NAME, 'self')
        self.load_model(
            model_weight_file=package_directory +
            '/strong_baseline/market_resnet50_model_120_rank1_945.pth')

        normalize_transform = transforms.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        self.img_transform = transforms.Compose([
            transforms.Resize(cfg.INPUT.SIZE_TEST),
            transforms.ToTensor(), normalize_transform
        ])

    def load_model(self, model_weight_file):
        assert os.path.exists(
            model_weight_file), 'The file with {0} does not exist'.format(
                model_weight_file)
        self.model.load_param(model_weight_file)
        if cfg.MODEL.DEVICE:
            if torch.cuda.device_count() > 1 and not self.visualize:
                self.model = torch.nn.DataParallel(self.model)
            self.model.to('cuda')
        print('Loaded model weights from {}'.format(model_weight_file))
        print(self.model)

    def extract_feature(self, detections, image):

        imgs = []
        for detect in detections:
            bbox = detect[0]  # [left, top, w, h]
            crop_img = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] +
                             bbox[2]]
            crop_img = Image.fromarray(
                cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            prep_img = img_to_tensor(crop_img, self.img_transform)

            if self.visualize:
                prep_img = prep_img.to(
                    cfg.MODEL.DEVICE
                ) if torch.cuda.device_count() >= 1 else prep_img
                # print(prep_img.size())
                self.visualize_saliency_maps(
                    org_img=crop_img,
                    prep_img=prep_img,
                    file_name='{:06d}'.format(self.save_ids))
                self.save_ids += 1

            imgs.append(prep_img)

        if len(imgs) == 1:
            imgs = (torch.stack(imgs)).view(1, 3, cfg.INPUT.SIZE_TEST[0],
                                            cfg.INPUT.SIZE_TEST[1])
        else:
            imgs = torch.squeeze(torch.stack(imgs))

        self.model.eval()
        with torch.no_grad():
            imgs = imgs.to(
                cfg.MODEL.DEVICE) if torch.cuda.device_count() >= 1 else imgs
            global_features = self.model(imgs)

        global_features = global_features.data.cpu()
        return global_features

    def visualize_saliency_maps(self, org_img, prep_img, file_name):
        grad_cam = GradCam(
            self.model, target_layer='layer4', model_name='strong-baseline')
        activation_map = grad_cam.generate_cam(org_img, prep_img)
        save_class_activation_images(
            org_img=org_img,
            activation_map=activation_map,
            folder='strong-baseline',
            file_name=file_name)


class AlignedReID_Plus():
    def __init__(self, visualize=False):
        self.visualize = visualize
        self.save_ids = 1
        self.use_gpu = torch.cuda.is_available()
        self.model = models.init_model(
            name='resnet50',
            num_classes=751,
            loss={'softmax', 'metric'},
            use_gpu=self.use_gpu,
            aligned=True)

        if self.use_gpu:
            self.model = self.model.cuda()

        self.load_model(model_weight_file=package_directory +
                        '/aligned_reid_plus/checkpoint_ep300.pth.tar')
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_weight_file):
        assert os.path.exists(
            model_weight_file), 'The file with {0} does not exist'.format(
                model_weight_file)
        checkpoint = torch.load(model_weight_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Loaded model weights from {}'.format(model_weight_file))
        print(self.model)

    def extract_feature(self, detections, image):
        imgs = []
        for detect in detections:
            bbox = detect[0]  # [left, top, w, h]
            crop_img = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] +
                             bbox[2]]
            crop_img = Image.fromarray(
                cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            prep_img = img_to_tensor(crop_img, self.img_transform)
            if self.use_gpu:
                prep_img = prep_img.cuda()

            if self.visualize:
                prep_img = prep_img.to(
                    cfg.MODEL.DEVICE
                ) if torch.cuda.device_count() >= 1 else prep_img
                # print(prep_img.size())
                self.visualize_saliency_maps(
                    org_img=crop_img,
                    prep_img=prep_img,
                    file_name='{:06d}'.format(self.save_ids))
                self.save_ids += 1

            imgs.append(prep_img)

        if len(imgs) == 1:
            imgs = (torch.stack(imgs)).view(1, 3, 256, 128)
        else:
            imgs = torch.squeeze(torch.stack(imgs))

        imgs = imgs.cuda()
        self.model.eval()
        global_features, _ = self.model(imgs)
        global_features = global_features.data.cpu()
        return global_features

    def visualize_saliency_maps(self, org_img, prep_img, file_name):
        grad_cam = GradCam(
            self.model, target_layer='7', model_name='aligned-reid-plus')
        activation_map = grad_cam.generate_cam(org_img, prep_img)
        save_class_activation_images(
            org_img=org_img,
            activation_map=activation_map,
            folder='aligned-reid-plus',
            file_name=file_name)


class AlignedReID():
    def __init__(self, visualize=False):
        self.visualize = visualize
        self.save_ids = 1
        TVT, TMO = set_devices((0, ))  # gpu_id
        self.model = Model(local_conv_out_channels=128, num_classes=751)
        if not visualize:
            self.model = DataParallel(self.model)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=2e-4, weight_decay=0.0005)
        # Bind them together just to save some codes in the following usage.
        modules_optims = [self.model, optimizer]
        TMO(modules_optims)

        self.load_model(model_weight_file=package_directory +
                        '/aligned_reid/model_weight.pth')
        self.extract_feat_func = ExtractFeature(self.model, TVT)

        self.pre_process_im = PreProcessIm()

    def load_model(self, model_weight_file):
        assert os.path.exists(
            model_weight_file), 'The file with {0} does not exist'.format(
                model_weight_file)
        map_location = (lambda storage, loc: storage)
        sd = torch.load(model_weight_file, map_location=map_location)
        load_state_dict(self.model, sd)
        print('Loaded model weights from {}'.format(model_weight_file))
        print(self.model)

    def extract_feature(self, detections, image):
        global_features = []
        local_feautres = []
        for detect in detections:
            bbox = detect[0]  # [left, top, w, h]
            crop_img = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] +
                             bbox[2]]
            im = np.asarray(crop_img)
            im, _ = self.pre_process_im(im)
            im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))

            crop_img = Image.fromarray(
                cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            if self.visualize:
                prep_img = img_to_tensor(crop_img, transforms.ToTensor())
                prep_img = prep_img.to(
                    cfg.MODEL.DEVICE
                ) if torch.cuda.device_count() >= 1 else prep_img
                # print(prep_img.size())
                self.visualize_saliency_maps(
                    org_img=crop_img,
                    prep_img=prep_img,
                    file_name='{:06d}'.format(self.save_ids))
                self.save_ids += 1

            global_feat, local_feat = self.extract_feat_func(im)
            global_features.append(global_feat[0])
            local_feautres.append(local_feat)
        return global_features

    def visualize_saliency_maps(self, org_img, prep_img, file_name):
        grad_cam = GradCam(
            self.model, target_layer='layer4', model_name='aligned-reid')
        activation_map = grad_cam.generate_cam(org_img, prep_img)
        save_class_activation_images(
            org_img=org_img,
            activation_map=activation_map,
            folder='aligned-reid',
            file_name=file_name)
