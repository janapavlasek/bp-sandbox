import numpy as np
from . import spider

MIN = 1e-4
BETA = 5e-3
GAMMA = 3


def batch_ave_score(obs, idx, tag):
    """Calculates batch average score over an observation at the indices
    specified.

    Args:
        obs: Observation (H, W).
        idx: Indices of the elements over which to average the score (N, D, 2).
             Each index is in form (col, row).
        tag: The label to look for in the observation (int).
    """
    H, W = obs.shape
    N, D, _ = idx.shape
    # Find which pixel coordinates are in the range of the image.
    in_range = (idx >= np.array([0, 0])) * (idx < np.array((W, H)))
    in_range = in_range.prod(axis=-1)   # (N, D), zero where out of range.

    idx = idx.reshape(N * D, -1)        # (NxD, 2)
    # All the coordinates that are out of range should be set to 0 to avoid
    # out of bounds errors.
    idx[(1 - in_range.reshape(-1)).nonzero()] = np.array([0, 0])

    cols, rows = idx[:, 0], idx[:, 1]
    # Grab the values at the rows and cols.
    scores = obs[rows, cols].reshape((N, D))
    # Make sure that the out of bounds elements don't show up as a match.
    scores = (scores == tag).astype(int) * in_range
    scores = scores.sum(axis=1)

    # Total number of visible pixels.
    total = in_range.sum(axis=1)
    # To avoid divide by zero, set pixels that are not visible to minimum score.
    zero_indices = (total < 1).nonzero()
    total[zero_indices] = 1
    scores[zero_indices] = MIN

    return np.maximum(scores / total, MIN)


def shape_ave_score(obs, shape):
    # The score is the average sum of pixels of the appropriate tag within the
    # shape mask.
    m = shape.mask()
    score = np.count_nonzero(np.logical_and(m, obs == shape.tag))
    total = np.count_nonzero(m)
    if total == 0:
        return MIN
    return max(score / total, MIN)


def batch_circle_ave_score(obs, states, radius, tag):
    # The score is the average sum of pixels of the appropriate tag within the
    # shape mask.
    H, W = obs.shape
    c = spider.Circle(radius)

    idx = np.stack((c.indices[1], c.indices[0]))
    pixel_coords = np.array([1, -1]) * states + np.array([0, H])
    pixel_coords = np.round(np.expand_dims(pixel_coords, -1) + idx).astype(int)
    N, _, D = pixel_coords.shape
    pixel_coords = pixel_coords.transpose((0, 2, 1))  # (N, D, 2)

    return batch_ave_score(obs, pixel_coords, tag)


def batch_rect_ave_score(obs, states, w, h, tag):
    # The score is the average sum of pixels of the appropriate tag within the
    # shape mask.
    r = spider.Rectangle(w, h)

    sin_t, cos_t = np.sin(states[:, 2]), np.cos(states[:, 2])
    rot = np.stack((cos_t, sin_t, -sin_t, cos_t), axis=1)
    rot = np.expand_dims(rot, axis=-1).reshape((-1, 2, 2))  # (N, 2, 2)

    idx = np.stack((r.cols, r.rows), axis=0)  # (2, D)
    idx = np.matmul(rot, idx)  # (N, 2, D)

    pixel_coords = np.array([1, -1]) * states[:, :2] + np.array([0, r.img_size[1]])
    pixel_coords = np.round(np.expand_dims(pixel_coords, -1) + idx).astype(int)
    pixel_coords = pixel_coords.transpose((0, 2, 1))  # (N, D, 2)

    return batch_ave_score(obs, pixel_coords, tag)


def spider_ave_score(obs, spi):
    # The score is the average sum of pixels of the appropriate tag within the
    # shape mask.
    x = spi.observation()
    m = x > 0
    score = np.count_nonzero(np.logical_and(m, obs == x))
    total = np.count_nonzero(m)
    if total == 0:
        return MIN
    return max(score / total, MIN)


def root_arm_pairwise(x_s, x_t):
    # Node s is the root, node t is the arm.
    # Two vectors pointing from the circle to the center of the rectangle and
    # in the direction parallel to the rectangle.
    root_to_arm = [x_t.x - x_s.x, x_t.y - x_s.y]
    from_arm = [np.cos(x_t.theta), np.sin(x_t.theta)]

    # Distance from the center of the circle to the center of the rectangle.
    dist = np.sqrt(root_to_arm[0]**2 + root_to_arm[1]**2)

    # The distance from the circle to the rectangle should be radius + width.
    error_in_dist = np.abs(dist - (x_t.w + x_s.radius))
    # The dot product between the two vectors should be zero.
    dot = np.abs(from_arm[0] * root_to_arm[0] / dist + from_arm[1] * root_to_arm[1] / dist)

    # Pairwise penalizes either of these not being the case.
    return np.exp(-BETA * error_in_dist - GAMMA * (1 - dot))


def arm_arm_pairwise(x_s, x_t):
    # Node s is the inner arm, node t is the outer arm.
    # Point at the end of inner arm which is the joint connecting outer arm.
    joint_x = x_s.x + x_s.w * np.cos(x_s.theta) / 2
    joint_y = x_s.y + x_s.w * np.sin(x_s.theta) / 2

    # Two vectors pointing from the joint to the center of the outer arm and
    # in the direction parallel to the outer arm.
    joint_to_outer = [x_t.x - joint_x, x_t.y - joint_y]
    from_outer = [np.cos(x_t.theta), np.sin(x_t.theta)]

    # Distance from the joint to the center of the outer arm.
    dist = np.sqrt(joint_to_outer[0]**2 + joint_to_outer[1]**2)

    # The distance from the joint to the outer arm should be width.
    error_in_dist = np.abs(dist - x_t.w)
    # The dot product between the two vectors should be zero.
    dot = np.abs(from_outer[0] * joint_to_outer[0] / dist + from_outer[1] * joint_to_outer[1] / dist)

    # Pairwise penalizes either of these not being the case.
    return np.exp(-BETA * error_in_dist - GAMMA * (1 - dot))


def batch_root_arm_pairwise(x_s, x_t, w, radius):
    # Node s is the root, node t is the arm.
    # Two vectors pointing from the circle to the center of the rectangle and
    # in the direction parallel to the rectangle.
    root_to_arm = x_t[:, :2] - x_s
    from_arm = np.stack([np.cos(x_t[:, 2]), np.sin(x_t[:, 2])], axis=-1)

    # Distance from the center of the circle to the center of the rectangle.
    dist = np.sqrt(np.sum(root_to_arm * root_to_arm, axis=1))

    # The distance from the circle to the rectangle should be radius + width.
    error_in_dist = np.abs(dist - (w + radius))
    # The dot product between the two vectors should be zero.
    root_to_arm = root_to_arm / dist.reshape(-1, 1)
    dot = np.abs(np.sum(from_arm * root_to_arm, axis=1))

    # Pairwise penalizes either of these not being the case.
    return np.exp(-BETA * error_in_dist - GAMMA * (1 - dot))


def batch_arm_arm_pairwise(x_s, x_t, w):
    # Node s is the inner arm, node t is the outer arm.
    # Point at the end of inner arm which is the joint connecting outer arm.
    joint_x = x_s[:, 0] + w * np.cos(x_s[:, 2]) / 2
    joint_y = x_s[:, 1] + w * np.sin(x_s[:, 2]) / 2
    joint = np.stack([joint_x, joint_y], axis=-1)

    # Two vectors pointing from the joint to the center of the outer arm and
    # in the direction parallel to the outer arm.
    joint_to_outer = x_t[:, :2] - joint
    from_outer = np.stack([np.cos(x_t[:, 2]), np.sin(x_t[:, 2])], axis=-1)

    # Distance from the joint to the center of the outer arm.
    dist = np.sqrt(np.sum(joint_to_outer * joint_to_outer, axis=1))

    # The distance from the joint to the outer arm should be width.
    error_in_dist = np.abs(dist - w)
    # The dot product between the two vectors should be zero.
    joint_to_outer = joint_to_outer / dist.reshape(-1, 1)
    dot = np.abs(np.sum(from_outer * joint_to_outer, axis=1))

    # Pairwise penalizes either of these not being the case.
    return np.exp(-BETA * error_in_dist - GAMMA * (1 - dot))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    scene = spider.SpiderScene(20, 5)
    obs = scene.observation()
    img = scene.image()

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(obs, cmap=plt.get_cmap("hsv"))
    plt.subplot(1, 2, 2)
    plt.imshow(img)

    root = scene.spider.root
    root_to_inner = [root_arm_pairwise(root, arm) for arm in scene.spider.links[:4]]

    inner, outer = scene.spider.links[:4], scene.spider.links[4:]
    inner_to_outer = [arm_arm_pairwise(i, o) for i, o in zip(inner, outer)]
    print("Root to inner arm likelihoods:", root_to_inner)
    print("Inner to outer arm likelihoods:", inner_to_outer)

    batch_root = np.array([[root.x, root.y],
                           [root.x, root.y],
                           [root.x, root.y]])
    batch_inner = np.array([[inner[0].x, inner[0].y, inner[0].theta],
                            [inner[0].x, inner[0].y, inner[0].theta],
                            [inner[0].x, inner[0].y, inner[0].theta]])
    batch_outer = np.array([[outer[0].x, outer[0].y, outer[0].theta],
                            [outer[0].x, outer[0].y, outer[0].theta],
                            [outer[0].x, outer[0].y, outer[0].theta]])

    batch_root_to_inner = batch_root_arm_pairwise(batch_root, batch_inner, inner[0].w, root.radius)
    batch_inner_to_outer = batch_arm_arm_pairwise(batch_inner, batch_outer, inner[0].w)

    print("Batch root to inner arm likelihoods:", batch_root_to_inner)
    print("Batch inner to outer arm likelihoods:", batch_inner_to_outer)

    # Test wrong pairwise.
    OFF = [10, 25, 50]
    off_states = []
    off_rects = []
    for o in OFF:
        x = inner[0].x + o * np.cos(inner[0].theta)
        y = inner[0].y + o * np.sin(inner[0].theta)
        offset_inner = spider.Rectangle(inner[0].w, inner[0].h, x=x, y=y, theta=inner[0].theta)
        print("For offset", o, root_arm_pairwise(root, offset_inner))

        off_states.append([x, y, inner[0].theta])
        off_rects.append(offset_inner)

    batch_root = np.array([[root.x, root.y] for _ in OFF])
    batch_offset = batch_root_arm_pairwise(batch_root, np.array(off_states), inner[0].w, root.radius)
    print(batch_offset)

    plt.figure(2)
    plt.subplot(1, len(OFF) + 1, 1)
    plt.title("Perfect: {:.4f}".format(root_to_inner[0]))
    plt.imshow(np.bitwise_or(root.mask(), inner[0].mask()))

    for i in range(len(OFF)):
        plt.subplot(1, len(OFF) + 1, i + 2)
        plt.title("Offset by {}: {:.4f}".format(OFF[i], batch_offset[i]))
        plt.imshow(np.bitwise_or(root.mask(), off_rects[i].mask()))

    ROT = [np.pi / 10, np.pi / 6, np.pi / 4, np.pi / 2]
    rot_states = []
    rot_rects = []
    for o in ROT:
        theta = inner[0].theta + o
        offset_inner = spider.Rectangle(inner[0].w, inner[0].h, x=inner[0].x, y=inner[0].y, theta=theta)
        print("For rotation", o, root_arm_pairwise(root, offset_inner))

        rot_states.append([inner[0].x, inner[0].y, theta])
        rot_rects.append(offset_inner)

    batch_root = np.array([[root.x, root.y] for _ in ROT])
    batch_rot = batch_root_arm_pairwise(batch_root, np.array(rot_states), inner[0].w, root.radius)
    print(batch_rot)

    plt.figure(3)
    plt.subplot(1, len(ROT) + 1, 1)
    plt.title("Perfect: {:.4f}".format(root_to_inner[0]))
    plt.imshow(np.bitwise_or(root.mask(), inner[0].mask()))

    for i in range(len(ROT)):
        plt.subplot(1, len(ROT) + 1, i + 2)
        plt.title("Offset by {:.4f}: {:.4f}".format(ROT[i], batch_rot[i]))
        plt.imshow(np.bitwise_or(root.mask(), rot_rects[i].mask()))

    plt.show()
