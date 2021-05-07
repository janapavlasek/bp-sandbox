import numpy as np

MIN = 1e-4


def shape_ave_score(obs, shape):
    # The score is the average sum of pixels of the appropriate tag within the
    # shape mask.
    m = shape.mask()
    score = np.count_nonzero(np.logical_and(m, obs == shape.tag))
    total = np.count_nonzero(m)
    if total == 0:
        return MIN
    return max(score / total, MIN)


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
    beta = 2
    gamma = 10
    return np.exp(-beta * error_in_dist - gamma * dot)


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
    beta = 2
    gamma = 10
    return np.exp(-beta * error_in_dist - gamma * dot)


def batch_root_arm_pairwise(x_s, x_t, w, radius):
    # Node s is the root, node t is the arm.
    # Two vectors pointing from the circle to the center of the rectangle and
    # in the direction parallel to the rectangle.
    root_to_arm = x_t[:, :2] - x_s[:, :2]
    from_arm = np.stack([np.cos(x_t[:, 2]), np.sin(x_t[:, 2])], axis=-1)

    # Distance from the center of the circle to the center of the rectangle.
    dist = np.sqrt(np.sum(root_to_arm * root_to_arm, axis=1))

    # The distance from the circle to the rectangle should be radius + width.
    error_in_dist = np.abs(dist - (w + radius))
    # The dot product between the two vectors should be zero.
    root_to_arm = root_to_arm / dist
    dot = np.abs(np.sum(from_arm * root_to_arm, axis=1))

    # Pairwise penalizes either of these not being the case.
    beta = 2
    gamma = 10
    return np.exp(-beta * error_in_dist - gamma * dot)
