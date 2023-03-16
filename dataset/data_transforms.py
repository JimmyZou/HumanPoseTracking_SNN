import numpy as np
import random
import cv2


def get_aug_config(img_shape, input_shape):
    # if aug:
    #     rot_factor = 20
    #     rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= 0.6 else 0
    #     do_flip = random.random() <= 0.5
    #
    #     center_x_scale = random.uniform(0.7, 1.3)
    #     center_y_scale = random.uniform(0.7, 1.3)
    #     bbx_cx = img_shape[0] * 0.5 * center_x_scale
    #     bbx_cy = img_shape[0] * 0.5 * center_y_scale
    #     if do_flip:
    #         bbx_cx = img_shape[0] - bbx_cx - 1
    #
    #     bbx_scale = random.uniform(0.7, 1.3)
    #     bbx_length = max(img_shape[0], img_shape[1])
    #     bbx_with = bbx_length * bbx_scale
    #     bbx_height = bbx_length * bbx_scale
    #     bbx = [bbx_cx, bbx_cy, bbx_with, bbx_height]
    #
    #     trans = gen_trans_from_patch_cv(bbx_cx, bbx_cy, bbx_with, bbx_height,
    #                                     input_shape[0], input_shape[1], rot, False)
    # else:
    #     rot, do_flip = 0, False
    #     bbx_cx = img_shape[0] * 0.5
    #     bbx_cy = img_shape[0] * 0.5
    #     bbx_length = max(img_shape[0], img_shape[1])
    #     bbx_with = bbx_length
    #     bbx_height = bbx_length
    #     bbx = [bbx_cx, bbx_cy, bbx_with, bbx_height]
    #
    #     trans = gen_trans_from_patch_cv(bbx_cx, bbx_cy, bbx_with, bbx_height,
    #                                     input_shape[0], input_shape[1], rot, False)

    rot_factor = 20
    rot = np.clip(np.random.randn(), -1.0, 1.0) * rot_factor if random.random() <= 0.6 else 0  # random.random() -> [0,1)
    do_flip = random.random() <= 0.5
    # do_flip = True  # mirror

    center_x_scale = random.uniform(0.8, 1.3)
    center_y_scale = random.uniform(0.8, 1.3)
    # center_x_scale = 1.5
    # center_y_scale = 1.5
    bbx_cx = img_shape[0] * 0.5 * center_x_scale
    bbx_cy = img_shape[0] * 0.5 * center_y_scale
    if do_flip:
        bbx_cx = img_shape[0] - bbx_cx - 1

    bbx_scale = random.uniform(0.8, 1.3)
    # bbx_scale = 1.5
    bbx_length = max(img_shape[0], img_shape[1])
    bbx_with = bbx_length * bbx_scale
    bbx_height = bbx_length * bbx_scale
    bbx = [bbx_cx, bbx_cy, bbx_with, bbx_height]

    trans = gen_trans_from_patch_cv(bbx_cx, bbx_cy, bbx_with, bbx_height,
                                    input_shape[0], input_shape[1], rot, False)

    return rot, do_flip, bbx, bbx_scale, trans


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, rot, inv=False):
    src_w = src_width
    src_h = src_height
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def rotate_2d_matrix(rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    matrix = np.array([[cs, -sn], [sn, cs]])
    return matrix


def generate_patch_image(cvimg, do_flip, trans, input_shape):
    img = cvimg.copy()
    if do_flip:
        img = img[:, ::-1, :]

    img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_NEAREST)
    return img_patch


def trans_point2d(pts, trans):
    # pts [B, N, 2], trans [2, 3]
    assert pts.shape[-1] == 2
    b, n, _ = pts.shape
    src_pts = np.concatenate([pts, np.ones_like(pts)[..., 0:1]], axis=-1)
    dst_pts = np.dot(src_pts, trans.T)
    return dst_pts[..., 0:2]


def estimate_translation_np(S, joints_2d, joints_conf, cam_intr):
    """
    This function is borrowed from https://github.com/nkolot/SPIN/utils/geometry.py

    Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
        cam_intr: (fx, fy, cx, cy)
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = cam_intr[0:2]
    # optical center
    center = cam_intr[2:4]

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans