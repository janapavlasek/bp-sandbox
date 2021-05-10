import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


def make_rot_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])


def make_tf_matrix(x, y, theta=0):
    tf = np.eye(3)
    tf[0:2, 0:2] = make_rot_matrix(theta)
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


def keep_in_range(rows, cols, size):
    rows, cols = np.round(rows).astype(int), np.round(cols).astype(int)
    in_range = np.bitwise_and(rows >= 0, rows < size[1])
    in_range = np.bitwise_and(cols >= 0, in_range)
    in_range = np.bitwise_and(cols < size[0], in_range).nonzero()
    return rows[in_range], cols[in_range]


class Shape(object):
    def __init__(self):
        self.tag = 0

    def set_state(self, *args):
        raise NotImplementedError()

    def mask(self, *args):
        raise NotImplementedError()

    def area(self, *args):
        raise NotImplementedError()


class Rectangle(Shape):
    def __init__(self, w, h, x=0, y=0, theta=0, img_size=(640, 480), tag=1):
        super(Rectangle, self).__init__()

        self.w = w
        self.h = h

        self.tag = tag

        self.img_size = img_size

        self.x = x
        self.y = y
        self.theta = normalize_angle(theta)

        rows, cols = np.linspace(-h / 2, h / 2, num=h * 2), np.linspace(-w / 2, w / 2, num=w * 2)
        rows, cols = np.repeat(rows, cols.shape[0], axis=0), np.tile(cols, (rows.shape[0],))
        self.rows, self.cols = rows, cols

    def set_state(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = normalize_angle(theta)

    def area(self):
        return self.w * self.h

    def calc_tf(self):
        center_x, center_y = self.img_size[0] / 2, self.img_size[1] / 2
        rot = make_tf_matrix(center_x, center_y, -self.theta)
        translate = make_tf_matrix(-self.x, self.y - self.img_size[1], 0)
        tf = rot.dot(translate)
        return tf

    def mask(self):
        rot = make_rot_matrix(self.theta)
        rotated = rot.dot(np.stack([self.cols, self.rows], axis=0))
        rows, cols = keep_in_range(rotated[1, :] + (self.img_size[1] - self.y),
                                   rotated[0, :] + self.x, self.img_size)

        mask = np.zeros(self.img_size, dtype=np.bool).T
        mask[rows, cols] = True

        return mask


class Circle(Shape):
    def __init__(self, radius, x=0, y=0, img_size=(640, 480), tag=1):
        super(Circle, self).__init__()

        self.radius = int(round(radius))
        self.img_size = img_size
        self.tag = tag

        self.x = x
        self.y = y

        r = np.arange(-radius, radius)
        rows, cols = np.repeat(r, 2 * radius, axis=0), np.tile(r, (2 * radius,))
        keep = (rows * rows + cols * cols) < radius**2
        self.indices = (rows[keep], cols[keep])

    def set_state(self, x, y):
        self.x = x
        self.y = y

    def area(self):
        return np.pi * self.radius * self.radius

    def mask(self):
        rows, cols = keep_in_range(self.indices[0] + (self.img_size[1] - self.y),
                                   self.indices[1] + self.x, self.img_size)

        mask = np.zeros(self.img_size, dtype=np.bool).T
        mask[rows, cols] = True

        return mask


class Spider(object):
    def __init__(self, x=0, y=0, qs=[], w=40, h=10, r=10, img_size=(640, 480)):
        self.n_links = 8
        self.w = w
        self.h = h
        self.r = r
        self.img_size = img_size

        self.x = x
        self.y = y

        self.root = Circle(r)
        self.links = [Rectangle(w, h, tag=i + 2) for i in range(self.n_links)]
        self.qs = self.random_init() if len(qs) != self.n_links else qs

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

    def area(self):
        return self.n_links * self.links[0].area() + self.root.area()

    def update_links(self):
        self.root.set_state(self.x, self.y)
        tw = make_tf_matrix(self.x, self.img_size[1] - self.y)
        rect_center_tf = make_tf_matrix(self.w / 2, 0)

        for i in range(self.n_links):
            theta = self.qs[i]
            rect_tf = None
            if (i < self.n_links / 2):
                # This is the first layer of joints, connected to the root.
                t1 = make_tf_matrix(self.r + self.w / 2, 0)
                rot1 = make_tf_matrix(0, 0, theta)
                rect_tf = rot1.dot(t1)
            else:
                # This is the second layer of joints, connected to the first layer.
                parent_joint = self.qs[i - self.n_links // 2]
                theta = normalize_angle(theta + parent_joint)

                t2 = make_tf_matrix(self.w / 2, 0)
                rot2 = make_tf_matrix(0, 0, self.qs[i])
                t1 = make_tf_matrix(self.r + self.w + self.w / 2, 0)
                rot1 = make_tf_matrix(0, 0, parent_joint)

                rect_tf = rot1.dot(t1).dot(rot2).dot(t2)

            pt = np.array([0, 0, 1]).reshape((3, 1))
            new_pt = tw.dot(rect_tf).dot(rect_center_tf).dot(pt)

            self.links[i].set_state(new_pt[0, 0], self.img_size[1] - new_pt[1, 0], theta)

    def observation(self):
        obs = np.zeros(self.img_size).T
        # Stamp circle.
        obs[self.root.mask()] = self.root.tag
        # Stamp rectangles.
        for r in self.links:
            obs[r.mask()] = r.tag

        return obs


class SpiderScene(object):
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
            obs[r.mask()] = r.tag
        for c in self.circles:
            obs[c.mask()] = c.tag

        spi_obs = self.spider.observation()
        obs = np.where(spi_obs > 0, spi_obs, obs)

        return obs

    def image(self, obs=None):
        if obs is None:
            obs = self.observation()
        mask = Image.fromarray(np.uint8((obs > 0) * 255))
        obs = Image.fromarray(np.uint8(cm.hsv(obs / obs.max()) * 255))
        img = Image.new('RGB', self.img_size, color="#ffffff")
        img.paste(obs, mask=mask)

        return img

    def display_belief(self, marginals, img=None):
        if img is None:
            img = self.image()

        # Make the image black and white.
        img = img.convert("L").convert("RGB")

        mask = np.zeros(self.img_size).T
        bel_obs = np.zeros(self.img_size).T
        for bel in marginals:
            for x in bel:
                m = x.mask()
                mask[m] += 1
                bel_obs[m] = x.tag

        mask = Image.fromarray(np.uint8(mask / mask.max() * 255))
        bel_obs = Image.fromarray(np.uint8(cm.hsv(bel_obs / bel_obs.max()) * 255))

        img.paste(bel_obs, mask=mask)

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

    scene = SpiderScene(20, 5)
    obs = scene.observation()
    img = scene.image()

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(obs, cmap=plt.get_cmap("hsv"))
    plt.subplot(1, 2, 2)
    plt.imshow(img)

    plt.show()
