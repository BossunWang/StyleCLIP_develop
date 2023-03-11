
from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import clip
from contextlib import contextmanager
from tqdm.auto import tqdm
import shutil
import cv2
import onnxruntime
import pickle
import math
from scipy.stats import norm


from encoder4editing.utils.common import tensor2im
from encoder4editing.models.psp import pSp
from global_torch.manipulate import Manipulator
from global_torch.StyleCLIP import GetDt, GetBoundary
import dlib
from encoder4editing.utils.alignment import align_face
from arcface_transform_matrix import norm_crop

# pip install dlib==19.20


def run_alignment(image_path, predictor):
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    # print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def display_alongside_source_image(result_image, source_image, resize_dims):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)


def run_on_batch(inputs, net, experiment_type):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    if experiment_type == 'cars_encode':
        images = images[:, :, 32:224, :]
    return images, latents


# Mute GetBoundary()
# https://stackoverflow.com/a/25061573
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def gen_image(fs3, dt, M, dlatent_tmp, beta, alpha):
    M.alpha = [alpha]
    with suppress_stdout():
        boundary_tmp2, c = GetBoundary(fs3, dt, M,threshold=beta)
    codes = M.MSCode(dlatent_tmp, boundary_tmp2)
    out = M.GenerateImg(codes)
    pil_image = Image.fromarray(out[0,0])
    return np.array(pil_image)[:, :, ::-1].copy()


def GetBoundaryNum(fs3, dt, threshold):
    tmp = np.dot(fs3, dt)
    select = np.abs(tmp) < threshold
    num_c = np.sum(~select)
    return num_c



def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort()返回数组值从小到大的索引值
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:  # 还有数据
        # print("order:", order)
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        IOU = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        # print("IOU:", IOU)
        left_index = (np.where(IOU <= thresh))[0]

        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[left_index + 1]

    return keep


def box_iou_numpy_ver(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) -
             np.maximum(box1[:, None, :2], box2[:, :2]))
    inter = np.clip(inter, 0, np.max(inter)).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression_face_numpy_ver(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, labels=(), nc=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 15 if nc is None else nc  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 1  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, prediction.shape[2]))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 15), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 15] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        # x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        x[:, 15:16] = x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        if multi_label:
            i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 15, None], x[:, 5:15], j[:, None].float()), 1)
        else:  # best class only
            conf = np.max(x[:, 15:16], axis=1, keepdims=True)
            j = np.argmax(x[:, 15:16], axis=1)
            j = np.expand_dims(j, axis=1).astype(np.float32)
            if x.shape[1] > 16:
                x = np.concatenate([box, conf, x[:, 5:15], j, x[:, 15:]], axis=1)[conf.reshape(-1) > conf_thres]
            else:
                x = np.concatenate([box, conf, x[:, 5:15], j], axis=1)[conf.reshape(-1) > conf_thres]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        # if i.shape[0] > max_det:  # limit detections
        #    i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou_numpy_ver(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.dot(weights, x[:, :4]).float() / weights.sum(1, keepdims=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def clip_coords_numpy_ver(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def scale_coords_numpy_ver(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords_numpy_ver(coords, img0_shape)
    return coords


def scale_coords_landmarks_numpy_ver(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clip(0, img0_shape[1])  # x1
    coords[:, 1].clip(0, img0_shape[0])  # y1
    coords[:, 2].clip(0, img0_shape[1])  # x2
    coords[:, 3].clip(0, img0_shape[0])  # y2
    coords[:, 4].clip(0, img0_shape[1])  # x3
    coords[:, 5].clip(0, img0_shape[0])  # y3
    coords[:, 6].clip(0, img0_shape[1])  # x4
    coords[:, 7].clip(0, img0_shape[0])  # y4
    coords[:, 8].clip(0, img0_shape[1])  # x5
    coords[:, 9].clip(0, img0_shape[0])  # y5
    return coords


def inference_onnx(ort_session, img_raw, save_path
                   , long_side=640
                   , confidence_threshold=0.5
                   , nms_threshold=0.3):
    # Set inputs
    img = np.float32(img_raw)
    height, width, _ = img.shape
    im_size_max = np.max(img.shape[0:2])

    image_t = np.empty((im_size_max, im_size_max, 3), dtype=img.dtype)
    image_t[:, :] = (0, 0, 0)
    image_t[0:0 + height, 0:0 + width] = img
    img = cv2.resize(image_t, (long_side, long_side))

    img /= (255, 255, 255)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img.reshape(1, 3, long_side, long_side)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    outname = [output.name for output in ort_session.get_outputs()]

    print("inputs name:", ort_inputs.keys(), "|| outputs name:", outname)

    keys = list(ort_inputs.keys())

    tic = time.time()
    ort_outs = ort_session.run(outname, ort_inputs)[0]
    print('net forward time: {:.4f}'.format(time.time() - tic))

    pred = non_max_suppression_face_numpy_ver(ort_outs, confidence_threshold, nms_threshold, nc=1)[0]

    img0 = img_raw.copy()
    h, w, c = img0.shape
    h = max(h, w)
    w = max(h, w)
    img0_shape = (h, w, c)
    gn = np.array(img0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
    boxes = []
    landmarks_list = []
    uc_list = []
    face_number = 0

    if pred is not None:
        pred[:, :4] = scale_coords_numpy_ver(img.shape[2:], pred[:, :4], img0_shape).round()
        pred[:, 5:15] = scale_coords_landmarks_numpy_ver(img.shape[2:], pred[:, 5:15], img0_shape).round()
        for j in range(pred.shape[0]):
            xywh = (xyxy2xywh(pred[j, :4].reshape(1, 4)) / gn).reshape(-1)
            conf = pred[j, 4]
            landmarks = (pred[j, 5:15].reshape(1, 10)).reshape(-1)

            al_confidence_value = 1.0
            cls_uc_confidence_value = 1.0

            if conf >= confidence_threshold:
                x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
                y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
                x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
                y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
                boxes.append(np.array([x1, y1, x2, y2, conf]))
                landmarks_list.append(landmarks)
                uc_list.append([al_confidence_value, cls_uc_confidence_value])

        # show image
        for b, landmarks, uc_values in zip(boxes, landmarks_list, uc_list):
            # b = list(map(lambda x:int(round(x, 0)), b))
            b = list(map(int, b))
            landmarks = list(map(int, landmarks))
            ROI_img = norm_crop(img0, np.array(landmarks).reshape((5, 2)))
            cv2.imwrite(save_path, ROI_img)


def run(source_neutral_dir
        , target_dir
        , target_latent_dir
        , emotion_list
        , target_prompt_list
        , neutral
        , experiment_type
        , resize_dims
        , EXPERIMENT_ARGS
        , alphas
        , betas
        , fs3
        , num_frames
        , predictor
        , net
        , clip_model
        , M
        , ort_session
        , minc=100
        , maxc=500
        , save_path_direct=False):
    for root, dirs, files in os.walk(source_neutral_dir):
        for name in files:
            image_path = os.path.join(root, name)
            target_path = os.path.join(target_dir, emotion_list[-1], name)

            # @title Align image
            original_image = Image.open(image_path)
            original_image = original_image.convert("RGB")

            if experiment_type == "ffhq_encode":
                input_image = run_alignment(image_path, predictor)
            else:
                input_image = original_image

            if input_image is None:
                continue

            input_image.resize(resize_dims)
            # input_image.save(target_path)

            # @title Invert the image
            img_transforms = EXPERIMENT_ARGS['transform']
            transformed_image = img_transforms(input_image)

            with torch.no_grad():
                images, latents = run_on_batch(transformed_image.unsqueeze(0), net, experiment_type)
                result_image, latent = images[0], latents[0]
            latent_path = target_latent_dir + '/{}_latents.pt'.format(image_path.split("/")[-1].replace(".jpg", ""))
            if not os.path.exists(latent_path):
                torch.save(latents, latent_path)

            # Choose Mode (and show input image)
            img_index = 0
            latents = torch.load(latent_path)
            dlatents_loaded = M.G.synthesis.W2S(latents)

            img_indexs = [img_index]
            dlatents_loaded = M.S2List(dlatents_loaded)
            dlatent_tmp = [tmp[img_indexs] for tmp in dlatents_loaded]
            M.num_images = len(img_indexs)

            M.alpha = [0]
            M.manipulate_layers = [0]
            codes, out = M.EditOneC(0, dlatent_tmp)
            M.manipulate_layers = None

            for ti, target_prompts in enumerate(target_prompt_list):
                for tj, target_prompt in enumerate(target_prompts):
                    classnames = [target_prompt, neutral]
                    dt = GetDt(classnames, clip_model)
                    print(classnames)

                    # video
                    # Renders a video interpolating from the base image with provided beta to the target_alpha.
                    # (target_alpha can be positive or negative)
                    best_beta = 0.08
                    best_num_c = 0
                    for b in betas:
                        num_c = GetBoundaryNum(fs3, dt, b)
                        if minc < num_c < maxc:
                            best_beta = b
                            best_num_c = num_c

                    print("best beta:", best_beta)
                    print("best_num_c:", best_num_c)
                    beta = best_beta

                    print("Generating Frames:")
                    for ai, alpha in tqdm(enumerate(alphas), total=num_frames):
                        image_name = name.replace(".jpg", "_{}_ev_{}.jpg".format(tj, ai))
                        if save_path_direct:
                            save_path = os.path.join(target_dir, emotion_list[ti], image_name)
                        elif ai <= 2 and ti == 4 and tj == 0:
                            save_path = os.path.join(target_dir, emotion_list[-1], image_name)
                        elif ai >= 3:
                            save_path = os.path.join(target_dir, emotion_list[ti], image_name)
                        else:
                            continue
                        # print(save_path)
                        output_image = gen_image(fs3, dt, M, dlatent_tmp, beta, alpha)
                        inference_onnx(ort_session, output_image, save_path)


def run_neutral(source_neutral_dir
                , target_dir
                , target_latent_dir
                , emotion_list
                , target_prompt_list
                , neutral
                , experiment_type
                , resize_dims
                , EXPERIMENT_ARGS
                , alpha
                , betas
                , fs3
                , num_frames
                , predictor
                , net
                , clip_model
                , M
                , ort_session):
    for root, dirs, files in os.walk(source_neutral_dir):
        for name in files:
            image_path = os.path.join(root, name)
            target_path = os.path.join(target_dir, emotion_list[-1], name)

            # @title Align image
            original_image = Image.open(image_path)
            original_image = original_image.convert("RGB")

            if experiment_type == "ffhq_encode":
                input_image = run_alignment(image_path, predictor)
            else:
                input_image = original_image

            if input_image is None:
                continue

            input_image.resize(resize_dims)
            # input_image.save(target_path)

            # @title Invert the image
            img_transforms = EXPERIMENT_ARGS['transform']
            transformed_image = img_transforms(input_image)

            with torch.no_grad():
                images, latents = run_on_batch(transformed_image.unsqueeze(0), net, experiment_type)
                result_image, latent = images[0], latents[0]
            latent_path = target_latent_dir + '/{}_latents.pt'.format(image_path.split("/")[-1].replace(".jpg", ""))
            if not os.path.exists(latent_path):
                torch.save(latents, latent_path)

            # Choose Mode (and show input image)
            img_index = 0
            latents = torch.load(latent_path)
            dlatents_loaded = M.G.synthesis.W2S(latents)

            img_indexs = [img_index]
            dlatents_loaded = M.S2List(dlatents_loaded)
            dlatent_tmp = [tmp[img_indexs] for tmp in dlatents_loaded]
            M.num_images = len(img_indexs)

            M.alpha = [0]
            M.manipulate_layers = [0]
            codes, out = M.EditOneC(0, dlatent_tmp)
            M.manipulate_layers = None

            for ti, target_prompts in enumerate(target_prompt_list):
                for tj, target_prompt in enumerate(target_prompts):
                    classnames = [target_prompt, neutral]
                    dt = GetDt(classnames, clip_model)
                    print(classnames)

                    # video
                    # Renders a video interpolating from the base image with provided beta to the target_alpha.
                    # (target_alpha can be positive or negative)
                    best_beta = 0.08
                    best_num_c = 0
                    for b in betas:
                        num_c = GetBoundaryNum(fs3, dt, b)
                        if 0 < num_c < 500:
                            best_beta = b
                            best_num_c = num_c

                    print("best beta:", best_beta)
                    print("best_num_c:", best_num_c)
                    beta = best_beta

                    print("Generating Frames:")
                    image_name = name.replace(".png", "_{}_ev_{}.png".format(tj, 0))
                    save_path = os.path.join(target_dir, emotion_list[ti], image_name)
                    # print(save_path)
                    output_image = gen_image(fs3, dt, M, dlatent_tmp, beta, alpha)
                    output_image = cv2.resize(output_image, (256, 256))
                    cv2.imwrite(save_path, output_image)


def gen_RAFDB_data(experiment_type, resize_dims, EXPERIMENT_ARGS, fs3, net, clip_model, M, predictor, ort_session):
    emotion_list = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
    # input text description
    neutral = 'face'  # @param {type:"string"}
    target_prompt_list = [
        ['surprised face', 'shocked face'],
        ['scared face'],
        ['vomiting face'],
        ['happy face', 'happy and wrinkle face'],
        ['sad face', 'depressed face'],
        ['angry face'],
    ]

    target_alpha = 6.  # @param {type:"number"}
    num_frames = 10  # @param {type:"number"}
    alphas = np.linspace(0, target_alpha, num_frames)
    # select beta adaptively
    min_beta = 0.08
    max_beta = 0.3
    betas = np.linspace(min_beta, max_beta, 10)

    source_train_neutral_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/org_train_class/neutral"
    source_test_neutral_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/org_test_class/neutral"
    target_train_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated/train_class_aligned"
    target_test_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated/test_class_aligned"
    target_latent_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_latents"
    os.makedirs(target_latent_dir, exist_ok=True)
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_test_dir, exist_ok=True)

    for emotion_name in emotion_list:
        os.makedirs(os.path.join(target_train_dir, emotion_name), exist_ok=True)
        os.makedirs(os.path.join(target_test_dir, emotion_name), exist_ok=True)

    run(source_train_neutral_dir
        , target_train_dir
        , target_latent_dir
        , emotion_list
        , target_prompt_list
        , neutral
        , experiment_type
        , resize_dims
        , EXPERIMENT_ARGS
        , alphas
        , betas
        , fs3
        , num_frames
        , predictor
        , net
        , clip_model
        , M
        , ort_session)

    run(source_test_neutral_dir
        , target_test_dir
        , target_latent_dir
        , emotion_list
        , target_prompt_list
        , neutral
        , experiment_type
        , resize_dims
        , EXPERIMENT_ARGS
        , alphas
        , betas
        , fs3
        , num_frames
        , predictor
        , net
        , clip_model
        , M
        , ort_session)


def gen_RAFDB_data_v2(experiment_type, resize_dims, EXPERIMENT_ARGS, fs3, net, clip_model, M, predictor, ort_session):
    emotion_list = ["happy"]
    # input text description
    neutral = 'face'  # @param {type:"string"}
    target_prompt_list = [
        ['Laugh heartily face'],
    ]

    target_alpha = 6.  # @param {type:"number"}
    num_frames = 6  # @param {type:"number"}
    alphas = np.linspace(2, target_alpha, num_frames)
    # select beta adaptively
    min_beta = 0.1
    max_beta = 0.3
    betas = np.linspace(min_beta, max_beta, 10)

    source_train_neutral_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/org_train_class/neutral"
    source_test_neutral_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/org_test_class/neutral"
    target_train_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_v2/train_class_aligned"
    target_test_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_v2/test_class_aligned"
    target_latent_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_latents"
    os.makedirs(target_latent_dir, exist_ok=True)
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_test_dir, exist_ok=True)

    for emotion_name in emotion_list:
        os.makedirs(os.path.join(target_train_dir, emotion_name), exist_ok=True)
        os.makedirs(os.path.join(target_test_dir, emotion_name), exist_ok=True)

    run(source_train_neutral_dir
        , target_train_dir
        , target_latent_dir
        , emotion_list
        , target_prompt_list
        , neutral
        , experiment_type
        , resize_dims
        , EXPERIMENT_ARGS
        , alphas
        , betas
        , fs3
        , num_frames
        , predictor
        , net
        , clip_model
        , M
        , ort_session
        , minc=10
        , maxc=200
        , save_path_direct=True)

    run(source_test_neutral_dir
        , target_test_dir
        , target_latent_dir
        , emotion_list
        , target_prompt_list
        , neutral
        , experiment_type
        , resize_dims
        , EXPERIMENT_ARGS
        , alphas
        , betas
        , fs3
        , num_frames
        , predictor
        , net
        , clip_model
        , M
        , ort_session
        , minc=10
        , maxc=50
        , save_path_direct=True)


def gen_RAFDB_data_v3(experiment_type, resize_dims, EXPERIMENT_ARGS, fs3, net, clip_model, M, predictor, ort_session):
    emotion_list = ["happy"]
    # input text description
    neutral = 'face'  # @param {type:"string"}
    target_prompt_list = [
        ['Laugh heartily face'],
    ]

    target_alpha = 6.  # @param {type:"number"}
    num_frames = 10  # @param {type:"number"}
    alphas = np.linspace(1, target_alpha, num_frames)
    # select beta adaptively
    min_beta = 0.1
    max_beta = 0.3
    betas = np.linspace(min_beta, max_beta, 10)

    source_train_neutral_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/org_train_class/neutral"
    source_test_neutral_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/org_test_class/neutral"
    target_train_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_v3/train_class_aligned"
    target_test_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_v3/test_class_aligned"
    target_latent_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_latents"
    os.makedirs(target_latent_dir, exist_ok=True)
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_test_dir, exist_ok=True)

    for emotion_name in emotion_list:
        os.makedirs(os.path.join(target_train_dir, emotion_name), exist_ok=True)
        os.makedirs(os.path.join(target_test_dir, emotion_name), exist_ok=True)

    run(source_train_neutral_dir
        , target_train_dir
        , target_latent_dir
        , emotion_list
        , target_prompt_list
        , neutral
        , experiment_type
        , resize_dims
        , EXPERIMENT_ARGS
        , alphas
        , betas
        , fs3
        , num_frames
        , predictor
        , net
        , clip_model
        , M
        , ort_session
        , minc=10
        , maxc=200
        , save_path_direct=True)

    run(source_test_neutral_dir
        , target_test_dir
        , target_latent_dir
        , emotion_list
        , target_prompt_list
        , neutral
        , experiment_type
        , resize_dims
        , EXPERIMENT_ARGS
        , alphas
        , betas
        , fs3
        , num_frames
        , predictor
        , net
        , clip_model
        , M
        , ort_session
        , minc=10
        , maxc=200
        , save_path_direct=True)


def gen_FFHQ_data(experiment_type, resize_dims, EXPERIMENT_ARGS, fs3, net, clip_model, M, predictor, ort_session):
    emotion_list = ["neutral"]
    target_prompt_list = [
        ['bored face'],
    ]

    target_alpha = 6.  # @param {type:"number"}
    num_frames = 10  # @param {type:"number"}
    alphas = np.linspace(0, target_alpha, num_frames)
    # select beta adaptively
    min_beta = 0.08
    max_beta = 0.3
    betas = np.linspace(min_beta, max_beta, 10)

    source_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/FFHQ/images256x256"
    target_dir = "/media/glory/Bossun_D/dataset/face_dataset/FFHQ/generate"
    target_latent_dir = "/media/glory/Bossun_D/dataset/face_dataset/FFHQ/target_latents"
    os.makedirs(target_latent_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    for emotion_name in emotion_list:
        os.makedirs(os.path.join(target_dir, emotion_name), exist_ok=True)

    run_neutral(source_dir
                , target_dir
                , target_latent_dir
                , emotion_list
                , target_prompt_list
                , "happy face"
                , experiment_type
                , resize_dims
                , EXPERIMENT_ARGS
                , target_alpha
                , betas
                , fs3
                , num_frames
                , predictor
                , net
                , clip_model
                , M
                , ort_session)


def main():
    # input dataset name
    dataset_name = 'ffhq'  # @param ['ffhq'] {allow-input: true}

    if not os.path.isfile('global_torch/model/' + dataset_name + '.pkl'):
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/'
        name = 'stylegan2-' + dataset_name + '-config-f.pkl'
        os.system('wget ' + url + name + '  -P  global_torch/model/')
        os.system('mv global_torch/model/' + name + ' global_torch/model/' + dataset_name + '.pkl')

    # input prepare data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    network_pkl = 'global_torch/model/' + dataset_name + '.pkl'
    device = torch.device('cuda')
    M = Manipulator()
    M.device = device
    G = M.LoadModel(network_pkl, device)
    M.G = G
    M.SetGParameters()
    num_img = 100_000
    M.GenerateS(num_img=num_img)
    M.GetCodeMS()
    np.set_printoptions(suppress=True)

    file_path = 'global_torch/npy/' + dataset_name + '/'
    fs3 = np.load(file_path + 'fs3.npy')

    # @title e4e setup
    # @ e4e setup
    experiment_type = 'ffhq_encode'
    EXPERIMENT_ARGS = {"model_path": "pretrained/e4e_ffhq_encode.pt", 'transform': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    resize_dims = (256, 256)

    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    # pprint.pprint(opts)  # Display full options used
    # update the training options
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    predictor = dlib.shape_predictor("pretrained/shape_predictor_68_face_landmarks.dat")
    # load yolov5 face onnx model
    onnx_path = "../Face_Detection_Project/yolov5_face/log_yolov5n-0.5/train_uc_append_masked_data/exp/weights/best_640.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # gen_RAFDB_data(experiment_type, resize_dims, EXPERIMENT_ARGS, fs3, net, clip_model, M, predictor, ort_session)
    # gen_RAFDB_data_v2(experiment_type, resize_dims, EXPERIMENT_ARGS, fs3, net, clip_model, M, predictor, ort_session)
    gen_RAFDB_data_v3(experiment_type, resize_dims, EXPERIMENT_ARGS, fs3, net, clip_model, M, predictor, ort_session)


if __name__ == '__main__':
    main()