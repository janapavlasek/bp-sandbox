import numpy as np
from . import spider
from . import sampling
from . import likelihoods


class NodeType:
    CIRCLE = 100
    RECTANGLE = 101


class EdgeType:
    ROOT_TO_ARM = 90
    ARM_TO_ROOT = 91
    INNER_TO_OUTER = 92
    OUTER_TO_INNER = 93


class SpiderNode(object):
    def __init__(self, id, shape, particles=[]):
        self.id = id
        self.type = shape

        self.particles = particles
        self.N = len(self.particles)
        self.weights = [1.0 / self.N for _ in range(self.N)]

    def set_particles(self, particles):
        self.N = len(self.particles)
        self.particles = particles

    def unary(self, x_s, obs):
        return likelihoods.shape_ave_score(obs, x_s)

    def resample(self):
        print("resample", self.weights)
        self.particles = sampling.importance_sample(self.particles, self.weights)

    def jitter(self, jitter_vars):
        for p in self.particles:
            p.jitter(jitter_vars)


class SpiderGraph(object):

    def __init__(self, N, img_size=(640, 480)):
        self.img_size = img_size
        self.N = N
        self.jitter_vars = [10, 10, 0.1]

        self.nodes = [SpiderNode(0, NodeType.CIRCLE)]
        self.nodes += [SpiderNode(i + 1, NodeType.RECTANGLE) for i in range(8)]

        self.edges = [[1, 2, 3, 4],  # Root is connected to four 1st layer arms.
                      # 1st layer arms are connected to root and a 2nd layer arm.
                      [0, 5],
                      [0, 6],
                      [0, 7],
                      [0, 8],
                      # 2nd layer arms are connected to a 1st layer arm.
                      [1],
                      [2],
                      [3],
                      [4]]

        # Initialize messages.
        self.messages = [np.full((len(n_edges), N), 1.0 / N) for n_edges in self.edges]
        self.proposal = self.marginals()

    def random_rect(self, tag):
        return spider.Rectangle(40, 10, tag=tag,
                                x=np.random.uniform(0, self.img_size[0]),
                                y=np.random.uniform(0, self.img_size[1]),
                                theta=np.random.uniform(0, np.pi))

    def random_circle(self, tag):
        return spider.Circle(10, tag=tag,
                             x=np.random.uniform(0, self.img_size[0]),
                             y=np.random.uniform(0, self.img_size[1]))

    def init_random_rects(self, tag):
        return [self.random_rect(tag) for _ in range(self.N)]

    def init_random_circles(self, tag):
        return [self.random_circle(tag) for _ in range(self.N)]

    def init_random(self):
        self.nodes[0].particles = self.init_random_circles(0)
        for i in range(1, 9):
            self.nodes[i].particles = self.init_random_rects(i + 1)
        self.proposal = self.marginals()

    def update_messages(self, obs):
        new_msgs = [np.zeros((len(n_edges), self.N)) for n_edges in self.edges]
        for s, n_edges in enumerate(self.edges):
            # Update the messages to node s from its neighbours, m_{t->s}
            for nbr_idx, t in enumerate(n_edges):
                edge_type = self.get_edge_type(t, s)
                for i, x_s in enumerate(self.nodes[s].particles):
                    m_ts = 0
                    for j, x_t in enumerate(self.proposal[t]):
                        m_st = self.messages[t][self.get_nbr_idx(s, t), j]
                        # Note: bel / bel_bar = p(z|xt).
                        m_ts += self.pairwise(x_s, x_t, edge_type) * self.nodes[t].unary(x_t, obs) / m_st

                    new_msgs[s][nbr_idx, i] = m_ts * 1.0 / self.N

        print("range", [(m.min(), m.max()) for m in new_msgs])
        print("NaNs?", [np.any(np.isnan(m)) for m in new_msgs])
        self.messages = new_msgs
        self.proposal = self.marginals()  # Register the current belief as the proposal.

    def update_belief(self, obs):
        for s, n_edges in enumerate(self.edges):
            weights = np.zeros(self.N)
            for i, x_s in enumerate(self.nodes[s].particles):
                bel_s = np.log(self.nodes[s].unary(x_s, obs))
                for nbr_idx, t in enumerate(n_edges):
                    bel_s += np.log(self.messages[s][nbr_idx, i])

                weights[i] = bel_s

            self.nodes[s].weights = sampling.normalize_weights(weights, log=True)

    def resample(self):
        for n in self.nodes:
            n.resample()

    def marginals(self):
        return [n.particles for n in self.nodes]

    def jitter(self):
        for n in self.nodes:
            n.jitter(self.jitter_vars)

    def get_nbr_idx(self, n_from, n_to):
        for idx, n in enumerate(self.edges[n_to]):
            if n == n_from:
                return idx

    def get_edge_type(self, n_from, n_to):
        if self.nodes[n_to].type == NodeType.CIRCLE and self.nodes[n_from].type == NodeType.RECTANGLE:
            return EdgeType.ARM_TO_ROOT
        elif self.nodes[n_to].type == NodeType.RECTANGLE and self.nodes[n_from].type == NodeType.CIRCLE:
            return EdgeType.ROOT_TO_ARM
        elif n_to <= 4 and n_from > 4:
            return EdgeType.OUTER_TO_INNER
        elif n_to > 4 and n_from <= 4:
            return EdgeType.INNER_TO_OUTER

    def pairwise(self, x_s, x_t, edge_type):
        # Edge type is from t->s.
        if edge_type == EdgeType.ARM_TO_ROOT:
            return likelihoods.root_arm_pairwise(x_s, x_t)
        elif edge_type == EdgeType.ROOT_TO_ARM:
            return likelihoods.root_arm_pairwise(x_t, x_s)
        elif edge_type == EdgeType.OUTER_TO_INNER:
            return likelihoods.arm_arm_pairwise(x_s, x_t)
        elif edge_type == EdgeType.INNER_TO_OUTER:
            return likelihoods.arm_arm_pairwise(x_t, x_s)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    scene = spider.Scene(20, 5)
    obs = scene.observation()
    img = scene.image(obs)

    print("Initializing")
    g = SpiderGraph(50)
    g.init_random()
    print("Displaying belief")
    initial_bel = scene.display_belief(g.marginals(), img)

    # Run one iteration.
    print("Update messages")
    g.update_messages(obs)
    print("Update belief")
    g.update_belief(obs)
    print("Resample")
    g.resample()

    bel2 = scene.display_belief(g.marginals(), img)

    plt.subplot(1, 3, 1)
    plt.title("Observation")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("Initial belief")
    plt.imshow(initial_bel)

    plt.subplot(1, 3, 3)
    plt.title("Belief after one iteration")
    plt.imshow(bel2)

    plt.show()
