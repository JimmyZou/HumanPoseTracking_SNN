"""render"""
import cv2
import numpy as np
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.01,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


def simple_renderer(rn, verts, faces, case=1, color=None):

    if color is None:
        # Rendered model_detr color
        color = colors['pink']
    else:
        color = color
        # color = colors['pink']

    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))

    albedo = rn.vc

    if case == 0:
        # Construct front Light
        rn.vc = LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([0, 0, -300]),
            vc=albedo,
            light_color=np.array([1, 1, 1]))
    elif case == 1:
        # top left and right
        rn.vc = LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([-300, -300, -300]),
            vc=albedo,
            light_color=np.array([0.8, 0.8, 0.8]))

        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([300, -300, -300]),
            vc=albedo,
            light_color=np.array([0.8, 0.8, 0.8]))

    elif case == 2:
        # top bottom light
        rn.vc = LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([0, -300, -300]),
            vc=albedo,
            light_color=np.array([0.8, 0.8, 0.8]))
        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([0, 300, -300]),
            vc=albedo,
            light_color=np.array([0.8, 0.8, 0.8]))

    elif case == 3:
        # strong right light
        rn.vc = LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([300, 0, -300]),
            vc=albedo,
            light_color=np.array([1.0, 1.0, 1.0]))
        # slight left light
        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([-300, 0, -300]),
            vc=albedo,
            light_color=np.array([0.4, 0.4, 0.4]))

    elif case == 4:
        # slight right light
        rn.vc = LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([300, 0, -300]),
            vc=albedo,
            light_color=np.array([0.4, 0.4, 0.4]))
        # strong left light
        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([-300, 0, -300]),
            vc=albedo,
            light_color=np.array([1.0, 1.0, 1.0]))

    else:
        raise ValueError('error light cases {}'.format(case))

    return rn.r


def get_alpha(imtmp, bgval=1.):
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)
    b_channel, g_channel, r_channel = cv2.split(imtmp)
    im_RGBA = cv2.merge(
        (b_channel, g_channel, r_channel, alpha.astype(imtmp.dtype)))
    return im_RGBA


def render_model(verts, faces, w, h, cam_param, cam_t, cam_rt, near=0.5, far=25, img=None, color=None, case=0):
    f = cam_param[0:2]
    c = cam_param[2:4]
    rn = _create_renderer(w=w, h=h, near=near, far=far, rt=cam_rt, t=cam_t, f=f, c=c)
    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img
    imtmp = simple_renderer(rn, verts, faces, case=case, color=color)
    # # If white bg, make transparent.
    # if img is None:
    #     imtmp = get_alpha(imtmp)
    return imtmp



def render_depth_v(verts, faces, require_visi = False,
                   t = [0.,0.,0.], img_size=[448, 448], f=[400.0,400.0], c=[224.,224.]):
    from opendr.renderer import DepthRenderer
    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt = np.zeros(3),
                              t = t,
                              f = f,
                              c = c,
                              k = np.zeros(5))
    rn.frustum = {'near': .01, 'far': 10000.,
                  'width': img_size[1], 'height': img_size[0]}
    rn.v = verts
    rn.f = faces
    rn.bgcolor = np.zeros(3)
    if require_visi is True:
        return rn.r, rn.visibility_image
    else:
        return rn.r


def projection_np(xyz, intr_param, simple_mode=True):
    # xyz: [N, 3]
    # intr_param: (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    assert xyz.shape[1] == 3
    fx, fy, cx, cy = intr_param[0:4]

    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_p = xyz[:, 0] / xyz[:, 2]
        y_p = xyz[:, 1] / xyz[:, 2]
        r2 = x_p ** 2 + y_p ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        b = b + (b == 0)
        d = a / b

        x_pp = x_p * d + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p ** 2)
        y_pp = y_p * d + p1 * (r2 + 2 * y_p ** 2) + 2 * p2 * x_p * y_p

        u = fx * x_pp + cx
        v = fy * y_pp + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
    else:
        u = xyz[:, 0] / xyz[:, 2] * fx + cx
        v = xyz[:, 1] / xyz[:, 2] * fy + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)


def unprojection_np(uvd, intr_param, simple_mode=True):
    # uvd: [N, 3]
    # cam_param: (fx, fy, cx, cy)
    # dist_coeff: (k1, k2, p1, p2, k3, k4, k5, k6)
    assert uvd.shape[1] == 3
    fx, fy, cx, cy = intr_param[0:4]

    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_pp = (uvd[:, 0] - cx) / fx
        y_pp = (uvd[:, 1] - cy) / fy
        r2 = x_pp ** 2 + y_pp ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        a = a + (a == 0)
        di = b / a

        x_p = x_pp * di
        y_p = y_pp * di

        x = uvd[:, 2] * (x_p - p2 * (y_p ** 2 + 3 * x_p ** 2) - p1 * 2 * x_p * y_p)
        y = uvd[:, 2] * (y_p - p1 * (x_p ** 2 + 3 * y_p ** 2) - p2 * 2 * x_p * y_p)
        z = uvd[:, 2]

        return np.stack([x, y, z], axis=1)
    else:
        x = uvd[:, 2] * (uvd[:, 0] - cx) / fx
        y = uvd[:, 2] * (uvd[:, 1] - cy) / fy
        z = uvd[:, 2]
        return np.stack([x, y, z], axis=1)


