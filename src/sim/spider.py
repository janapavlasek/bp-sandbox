import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw


def make_rot_matrix(x, y, theta=0):
    tf = np.eye(3)
    tf[0:2, 0:2] = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
    tf[0, 2] = x
    tf[1, 2] = y

    return tf


def affine_from_matrix(mat):
    return mat.reshape((9,)).tolist()[:6]


def normalize_angle(angle):
    """Normalize angle to stay between -PI and PI"""
    result = np.fmod(angle + np.pi, 2.0 * np.pi)
    if result <= 0:
        return result + np.pi
    return result - np.pi


class Rectangle(object):
    def __init__(self, w, h, img_size=(640, 480), tag=1):
        self.w = w
        self.h = h
        self.tag = tag

        self.img_size = img_size

        self.x = self.img_size[0] / 2
        self.y = self.img_size[1] / 2
        self.theta = 0

        self._mask = Image.new('L', self.img_size)
        center_x, center_y = self.img_size[0] / 2, self.img_size[1] / 2
        rect_coords = [center_x - self.w / 2, center_y - self.h / 2,
                       center_x + self.w / 2, center_y + self.h / 2]
        draw = ImageDraw.Draw(self._mask)
        draw.rectangle(rect_coords, fill=255)

    def set_state(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = normalize_angle(-theta)

    def calc_tf(self):
        center_x, center_y = self.img_size[0] / 2, self.img_size[1] / 2
        rot = make_rot_matrix(center_x, center_y, self.theta)
        translate = make_rot_matrix(-self.x, -self.y, 0)
        tf = rot.dot(translate)
        return tf

    def mask(self):
        tf = affine_from_matrix(self.calc_tf())
        mask = self._mask.transform(self.img_size, Image.AFFINE, data=tf, fillcolor=0)

        return np.array(mask) == 255


class Circle(object):
    def __init__(self, radius, img_size=(640, 480), tag=1):
        self.radius = radius
        self.img_size = img_size
        self.tag = tag

        self.x = self.img_size[0] / 2
        self.y = self.img_size[1] / 2

    def set_state(self, x, y):
        self.x = x
        self.y = y

    def calc_bbox(self):
        return [self.x - self.radius, self.y - self.radius,
                self.x + self.radius, self.y + self.radius]

    def mask(self):
        mask = Image.new('L', self.img_size)

        draw = ImageDraw.Draw(mask)
        draw.ellipse(self.calc_bbox(), fill=255)

        return np.array(mask) == 255


class Spider(object):
    def __init__(self, x=0, y=0, w=40, h=10, r=10, img_size=(640, 480)):
        self.n_links = 8
        self.w = w
        self.h = h
        self.r = r
        self.img_size = img_size

        self.x = x
        self.y = y

        self.root = Circle(r)
        self.links = [Rectangle(w, h, tag=i + 2) for i in range(self.n_links)]
        self.qs = self.random_init()

        self.update_links()

    def random_init(self, std=np.pi / 8):
        quarters = [i * (np.pi / 2) + np.pi / 4 for i in range(4)]
        init = quarters + [0] * 4
        init = [np.random.normal(q, std) for q in init]
        return init

    def set_state(self, x, y, qs=[]):
        self.x = x
        self.y = y

        if len(qs) <= len(self.links):
            for i in range(len(qs)):
                self.qs[i] = qs[i]

        self.update_links()

    def update_links(self):
        self.root.set_state(self.x, self.y)
        tw = make_rot_matrix(self.x, self.y)
        rect_center_tf = make_rot_matrix(self.w / 2, 0)

        for i in range(self.n_links):
            theta = self.qs[i]
            rect_tf = None
            if (i < self.n_links / 2):
                # This is the first layer of joints, connected to the root.
                t1 = make_rot_matrix(self.r + self.w / 2, 0)
                rot1 = make_rot_matrix(0, 0, theta)
                rect_tf = rot1.dot(t1)
            else:
                # This is the second layer of joints, connected to the first layer.
                parent_joint = self.qs[i - self.n_links // 2]
                theta = normalize_angle(theta + parent_joint)

                t2 = make_rot_matrix(self.w / 2, 0)
                rot2 = make_rot_matrix(0, 0, self.qs[i])
                t1 = make_rot_matrix(self.r + self.w + self.w / 2, 0)
                rot1 = make_rot_matrix(0, 0, parent_joint)

                rect_tf = rot1.dot(t1).dot(rot2).dot(t2)

            pt = np.array([0, 0, 1]).reshape((3, 1))
            new_pt = tw.dot(rect_tf).dot(rect_center_tf).dot(pt)

            self.links[i].set_state(new_pt[0, 0], new_pt[1, 0], theta)

    def observation(self):
        obs = np.zeros(self.img_size).T
        # Stamp circle.
        obs = np.where(self.root.mask(), np.full(self.img_size, self.root.tag).T, obs)
        # Stamp rectangles.
        for r in self.links:
            mask = r.mask()
            obs = np.where(mask, np.full(self.img_size, r.tag).T, obs)

        return obs


class Scene(object):
    def __init__(self, n_rect, n_circ, img_size=(640, 480)):
        self.n_rect = n_rect
        self.n_circ = n_circ
        self.img_size = img_size

        self.spider = Spider(np.random.normal(img_size[0] / 2, 60),
                             np.random.normal(img_size[1] / 2, 60))

        num_colours = 2 + self.spider.n_links
        self.rects = [Rectangle(40, 10, tag=np.random.randint(2, num_colours)) for _ in range(self.n_rect)]
        self.circles = [Circle(10) for _ in range(self.n_circ)]
        for r in self.rects:
            x = np.random.randint(0, self.img_size[0])
            y = np.random.randint(0, self.img_size[1])
            theta = np.random.uniform(0, np.pi)
            r.set_state(x, y, theta)
        for c in self.circles:
            x = np.random.randint(0, self.img_size[0])
            y = np.random.randint(0, self.img_size[1])
            c.set_state(x, y)

    def observation(self):
        obs = np.zeros(self.img_size).T
        for r in self.rects:
            mask = r.mask()
            obs = np.where(mask, np.full(self.img_size, r.tag).T, obs)
        for c in self.circles:
            mask = c.mask()
            obs = np.where(mask, np.full(self.img_size, c.tag).T, obs)

        spi_obs = self.spider.observation()
        obs = np.where(spi_obs > 0, spi_obs, obs)

        return obs

    def image(self):
        obs = self.observation()
        mask = Image.fromarray(np.uint8((obs > 0) * 255))
        obs = Image.fromarray(np.uint8(cm.hsv(obs / obs.max()) * 255))
        img = Image.new('RGB', self.img_size, color="#ffffff")
        img.paste(obs, mask=mask)

        return img


if __name__ == '__main__':
    r = Rectangle(40, 10)
    r.set_state(40, 60, 0.3)

    mask = r.mask()

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(mask)

    s = Spider(320, 240)
    plt.subplot(1, 2, 2)
    plt.imshow(s.observation(), cmap=plt.get_cmap("jet"))

    scene = Scene(20, 5)
    obs = scene.observation()
    img = scene.image()

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(obs, cmap=plt.get_cmap("hsv"))
    plt.subplot(1, 2, 2)
    plt.imshow(img)

    plt.show()
