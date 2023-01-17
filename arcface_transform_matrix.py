"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com
"""
# based on:
# https://github.com/deepinsight/insightface/blob/master/recognition/common/face_align.py

import os
import cv2
import numpy as np

src1 = np.array([
    [51.642, 50.115],
    [57.617, 49.990],
    [35.740, 69.007],
    [51.157, 89.050],
    [57.025, 89.702]], dtype=np.float32)
# <--left
src2 = np.array([
    [45.031, 50.118],
    [65.568, 50.872],
    [39.677, 68.111],
    [45.177, 86.190],
    [64.246, 86.758]], dtype=np.float32)

# ---frontal
src3 = np.array([
    [39.730, 51.138],
    [72.270, 51.138],
    [56.000, 68.493],
    [42.463, 87.010],
    [69.537, 87.010]], dtype=np.float32)

# -->right
src4 = np.array([
    [46.845, 50.872],
    [67.382, 50.118],
    [72.737, 68.111],
    [48.167, 86.758],
    [67.236, 86.190]], dtype=np.float32)

# -->right profile
src5 = np.array([
    [54.796, 49.990],
    [60.771, 50.115],
    [76.673, 69.007],
    [55.388, 89.702],
    [61.257, 89.050]], dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]], dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


def SimilarityTransform(src, dst, estimate_scale=True):
    state = 0

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # print('src_mean', src_mean)
    # print('dst_mean', dst_mean)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # print('src_demean', src_demean)
    # print('dst_demean', dst_demean)

    # Eq. (38).
    A = (dst_demean.T @ src_demean) / num

    # print('A:', A)

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    # print('d:', d)

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # print('U:', U)
    # print('S:', S)
    # print('V:', V)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        # print('rank = 0')
        return np.nan * T, state
    elif rank == dim - 1:
        # print('rank:', rank)
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            # print('det(U) * det(V) > 0')
            T[:dim, :dim] = U @ V
            state = 1
        else:
            # print('det(U) * det(V) <= 0')
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
            state = 2
    else:
        # print('rank:', rank)
        T[:dim, :dim] = U @ np.diag(d) @ V
        state = 3

    # print('T:', T)

    if estimate_scale:
        # print('src_demean.var:', src_demean.var(axis=0))
        # print('src_demean.var sum:', src_demean.var(axis=0).sum())
        # print('S @ d:', S @ d)

        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
        # print('scale:', scale)
    else:
        scale = 1.0

    # print('full_term:', dst_mean - scale * (T[:dim, :dim] @ src_mean.T))
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T, state


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    # M, pose_index = estimate_norm(landmark, image_size, mode)
    T, _ = SimilarityTransform(landmark, arcface_src[0])
    warped = cv2.warpAffine(img, T[:2], (image_size, image_size), borderValue=0.0)
    return warped


if __name__ == '__main__':
    image_size = 112
    mode = 'arcface'

    # landmarks = np.array([
    #     [686, 789],
    #     [902, 848],
    #     [801, 924],
    #     [654, 1040],
    #     [831, 1089]], dtype=np.float32)
    #
    # T, _ = SimilarityTransform(landmarks, arcface_src[0])
    # print('Transform:', T)
    #
    # landmarks = np.array([
    #     [320, 828],
    #     [540, 758],
    #     [420, 876],
    #     [384, 1053],
    #     [571, 996]], dtype=np.float32)
    #
    # T, _ = SimilarityTransform(landmarks, arcface_src[0])
    # print('Transform:', T)
    #
    # landmarks = np.array([
    #     [447, 1032],
    #     [703, 1046],
    #     [569, 1192],
    #     [453, 1332],
    #     [658, 1345]], dtype=np.float32)
    #
    # T, _ = SimilarityTransform(landmarks, arcface_src[0])
    # print('Transform:', T)
    #
    # landmarks = np.array([
    #     [270, 207],
    #     [330, 208],
    #     [317, 249],
    #     [264, 277],
    #     [318, 280]], dtype=np.float32)
    #
    # T, _ = SimilarityTransform(landmarks, arcface_src[0])
    # print('Transform:', T)

    landmarks = np.array([
        [178, 207],
        [256, 202],
        [199, 239],
        [182, 297],
        [243, 291]], dtype=np.float32)

    T, _ = SimilarityTransform(landmarks, arcface_src[0])
    print('Transform:', T[:2])

    M, _ = estimate_norm(landmarks, image_size, mode)
    print(M)

    # test rank 1
    # landmarks = np.array([
    #     [100, 200],
    #     [200, 300],
    #     [300, 400],
    #     [400, 500],
    #     [600, 700]], dtype=np.float32)
    # T, state = SimilarityTransform(landmarks, arcface_src[0])
    # print('Transform:', T)
    #
    # # Collinear
    # landmarks = np.array([
    #     [100, 100],
    #     [300, 300],
    #     [400, 400],
    #     [500, 500],
    #     [700, 700]], dtype=np.float32)
    # T, state = SimilarityTransform(landmarks, arcface_src[0])
    # print('Transform:', T)

    # # test same points
    # landmarks = np.array([
    #     [38.2946, 51.6963],
    #     [73.5318, 51.5014],
    #     [56.0252, 71.7366],
    #     [41.5493, 92.3655],
    #     [70.7299, 92.2041]], dtype=np.float32)
    # T, state = SimilarityTransform(landmarks, arcface_src[0])
    # print('Transform:', T)
    #
    # # test zero points
    # landmarks = np.zeros((5, 2))
    # T, state = SimilarityTransform(landmarks, arcface_src[0])
    # print('Transform:', T)
