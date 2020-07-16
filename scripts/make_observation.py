#!/usr/bin/env python
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = (500, 500)
NUM_ITER = 25
PERCENT_COVERAGE = 0.1

RECT = {"mean": {"width": 27, "height": 8},
        "var": {"width": 5, "height": 2}}
CIRCLE = {"mean": 10, "var": 2}
LINKSPACE = 27


def normalize_angle(angle):
    result = angle % 2.0 * np.pi
    if result < 0:
        return result + 2.0 * np.pi
    return result


class Rectangle(object):

    def __init__(self, x, y, theta, w, h):
        self.x = x
        self.y = y
        self.theta = theta
        self.w = w
        self.h = h
        self.points = []

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) >= (B[1] - A[1]) * (C[0] - A[0])

    def intersect(self, A, B, C, D):
        # Algorithm to check intersection of two line segments.
        #   (https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/)
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def pointInside(self, pt_x, pt_y, padding=False):
        # Algorithm to check if a point is inside a polygon.
        #   (https://en.wikipedia.org/wiki/Point_in_polygon)
        if pt_x < 0 or pt_x >= IMG_SIZE[0] or pt_y < 0 or pt_y >= IMG_SIZE[1]:
            return False

        num_intersect = 0

        origin = [0, 0]
        pt = [pt_x, pt_y]

        corners = self.points
        if padding:
            corners = self.padded_points

        for i in range(0, 4):
            if self.intersect(origin, pt, corners[i], corners[(i + 1) % 4]):
                num_intersect += 1

        return num_intersect % 2 != 0

    def draw(self, img):
        for i in range(int(np.floor(self.x - self.w)), int(np.ceil(self.x + self.w))):
            for j in range(int(np.floor(self.y - self.w)), int(np.ceil(self.y + self.w))):
                if self.pointInside(i, j):
                    img[i, j] = 1

    def free(self, img):
        # Add space so elements aren't touching.

        for i in range(int(np.floor(self.x - self.w)), int(np.ceil(self.x + self.w))):
            for j in range(int(np.floor(self.y - self.w)), int(np.ceil(self.y + self.w))):
                if self.pointInside(i, j, True):
                    if img[i, j] == 1:
                        return False

        return True

    def calc_points(self):
        tw = make_trans(self.x, self.y)
        rot = make_rot(self.theta)
        pt = np.array([[0], [0], [1]])

        rect_tf = tw.dot(rot)

        top_left_tf = make_trans(-self.w / 2, self.h / 2)
        bottom_left_tf = make_trans(-self.w / 2, -self.h / 2)
        top_right_tf = make_trans(self.w / 2, self.h / 2)
        bottom_right_tf = make_trans(self.w / 2, -self.h / 2)

        top_left = rect_tf.dot(top_left_tf).dot(pt).flatten().tolist()[:-1]
        bottom_left = rect_tf.dot(bottom_left_tf).dot(pt).flatten().tolist()[:-1]
        top_right = rect_tf.dot(top_right_tf).dot(pt).flatten().tolist()[:-1]
        bottom_right = rect_tf.dot(bottom_right_tf).dot(pt).flatten().tolist()[:-1]

        self.points = [top_left, top_right, bottom_right, bottom_left]

        # Padded points.
        pad = (2 * RECT["var"]["width"], 2 * RECT["var"]["height"])
        top_left_tf = make_trans(-self.w / 2 - pad[0], self.h / 2 + pad[1])
        bottom_left_tf = make_trans(-self.w / 2 - pad[0], -self.h / 2 - pad[1])
        top_right_tf = make_trans(self.w / 2 + pad[0], self.h / 2 + pad[1])
        bottom_right_tf = make_trans(self.w / 2 + pad[0], -self.h / 2 - pad[1])

        top_left = rect_tf.dot(top_left_tf).dot(pt).flatten().tolist()[:-1]
        bottom_left = rect_tf.dot(bottom_left_tf).dot(pt).flatten().tolist()[:-1]
        top_right = rect_tf.dot(top_right_tf).dot(pt).flatten().tolist()[:-1]
        bottom_right = rect_tf.dot(bottom_right_tf).dot(pt).flatten().tolist()[:-1]

        self.padded_points = [top_left, top_right, bottom_right, bottom_left]


def make_rot(theta):
    rot = np.eye(3)
    c, s = np.cos(theta), np.sin(theta)
    rot[0:2, 0:2] = np.array(((c, -s), (s, c)))
    return rot


def make_trans(x, y):
    t = np.eye(3)
    t[0, 2] = x
    t[1, 2] = y
    return t


def joints_to_rect(root, joints):
    w, h = RECT["mean"]["width"], RECT["mean"]["height"]
    ll = w
    ls = LINKSPACE
    tw = make_trans(root[0], root[1])

    top_left_tf = make_trans(ls, h / 2)
    bottom_left_tf = make_trans(ls, -h / 2)
    top_right_tf = make_trans(ls + ll, h / 2)
    bottom_right_tf = make_trans(ls + ll, -h / 2)

    rects = []

    num_joints = 8

    for i in range(num_joints):

        rect_center_tf = make_trans(ll / 2 + ls, 0)
        theta = 0

        if i < num_joints / 2:
            # first layer
            theta = joints[i]
            t1 = make_trans(ll / 2 + ls, 0)
            rot1 = make_rot(theta)
            rect_tf = tw.dot(rot1)
        else:
            parent_joint = joints[i - num_joints / 2]
            theta = normalize_angle(joints[i] + parent_joint)
            t1 = make_trans(ll + ls, 0)
            rot1 = make_rot(parent_joint)
            rot2 = make_rot(joints[i])
            rect_tf = tw.dot(rot1).dot(t1).dot(rot2)

        pt = np.array([[0], [0], [1]])
        new_pt = rect_tf.dot(rect_center_tf).dot(pt).flatten()
        r = Rectangle(new_pt[0], new_pt[1], theta, w, h)

        top_left = rect_tf.dot(top_left_tf).dot(pt).flatten().tolist()[:-1]
        bottom_left = rect_tf.dot(bottom_left_tf).dot(pt).flatten().tolist()[:-1]
        top_right = rect_tf.dot(top_right_tf).dot(pt).flatten().tolist()[:-1]
        bottom_right = rect_tf.dot(bottom_right_tf).dot(pt).flatten().tolist()[:-1]

        r.points = [top_left, top_right, bottom_right, bottom_left]

        rects.append(r)

    return rects


def draw_circle(img, circle):
    # Circle is (x, y, radius).
    x, y, r = circle
    ym, xm = np.ogrid[-x:img.shape[0] - x, -y:img.shape[1] - y]
    mask = xm * xm + ym * ym <= r * r

    img[mask] = 1


def circle_free(img, circle):
    x, y, r = circle
    r += 2 * CIRCLE["var"]
    ym, xm = np.ogrid[-x:img.shape[0] - x, -y:img.shape[1] - y]
    mask = xm * xm + ym * ym <= r * r

    return np.sum(img[mask]) <= 0


def make_observation():
    obs = np.zeros(IMG_SIZE, dtype=np.int)

    joint_var = 0.3

    # First draw the target.
    center = np.random.uniform(100, 400, 2).tolist()
    circle = center + [CIRCLE["mean"]]

    joints_center = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2]) + np.random.normal(0, joint_var, 4)
    joints_out = np.random.normal(0, joint_var, 4)
    joints = joints_center.tolist() + joints_out.tolist()

    rects = joints_to_rect(center, joints)

    gt = [(r.x, r.y, r.theta, r.w, r.h) for r in rects]
    gt = [circle] + gt

    draw_circle(obs, circle)
    for r in rects:
        r.draw(obs)

    obs_rects = []
    obs_circ = []

    step = 40
    start = IMG_SIZE[0] % (step * int(IMG_SIZE[0] / step)) / 2
    num_grids = IMG_SIZE[0] / step + 1
    order = np.arange(0, num_grids * num_grids)
    np.random.shuffle(order)

    area = float(IMG_SIZE[0] * IMG_SIZE[1])
    min_r, min_w, min_h = CIRCLE["mean"], RECT["mean"]["width"], RECT["mean"]["height"]
    max_r, max_w, max_h = CIRCLE["mean"], RECT["mean"]["width"], RECT["mean"]["height"]
    min_area, max_area = [RECT["mean"]["width"] * RECT["mean"]["height"]] * 2
    idx = 0
    while np.sum(obs) / area < PERCENT_COVERAGE:
        col = order[idx] % num_grids
        row = order[idx] / num_grids
        x = np.random.normal(col * step + start, 10)
        y = np.random.normal(row * step + start, 10)
        if np.random.random() < 8. / 9:
            # Rectangle!
            theta = np.random.uniform(0, np.pi)
            w = np.random.normal(RECT["mean"]["width"], RECT["var"]["width"])
            h = np.random.normal(RECT["mean"]["height"], RECT["var"]["height"])
            r = Rectangle(x, y, theta, w, h)
            r.calc_points()
            if r.free(obs):
                r.draw(obs)
                obs_rects.append((x, y, theta, w, h))
                min_w = min(min_w, w)
                max_w = max(max_w, w)
                min_h = min(min_h, h)
                max_h = max(max_h, h)
                min_area = min(min_area, h * w)
                max_area = max(max_area, h * w)
        else:
            # Circle.
            r = np.random.normal(CIRCLE["mean"], CIRCLE["var"])
            if circle_free(obs, [x, y, r]):
                draw_circle(obs, [x, y, r])
                obs_circ.append((x, y, r))
                min_r = min(min_r, r)
                max_r = max(max_r, r)

        idx += 1
        idx = idx % (num_grids * num_grids)

    print("Radius limits: {} to {}".format(min_r, max_r))
    print("Width limits: {} to {}".format(min_w, max_w))
    print("Height limits: {} to {}".format(min_h, max_h))
    print("Rectangle area limits: {} to {}".format(min_area, max_area))

    return obs, gt, obs_rects, obs_circ


if __name__ == '__main__':
    obs, gt, rects, circles = make_observation()

    print("Saving files.")

    with open("data/obs.pbm", "w") as f:
        f.write("P1\n")
        f.write(" ".join(str(x) for x in IMG_SIZE) + "\n")
        for row in range(obs.shape[0]):
            f.write(" ".join(str(x) for x in obs[row]) + "\n")

    with open("data/obs_data.txt", "w") as f:
        f.write("GT\n")
        for ele in gt:
            f.write(" ".join(str(x) for x in ele) + "\n")

        f.write("CIRCLES\n")
        f.write(" ".join(str(x) for x in gt[0]) + "\n")
        for c in circles:
            f.write(" ".join(str(x) for x in c) + "\n")

        f.write("RECTS\n")
        for i in range(1, len(gt)):
            f.write(" ".join(str(x) for x in gt[i]) + "\n")
        for r in rects:
            f.write(" ".join(str(x) for x in r) + "\n")

    print("Observation and data saved. Use the following command for the visualization image:")
    print("\n\tconvert obs.pbm -negate obs.png\n")

    plt.figure()
    plt.imshow(obs)
    plt.show()
