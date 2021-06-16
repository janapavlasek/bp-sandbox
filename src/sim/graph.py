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


class AlgoType:
    SUM_PRODUCT = 999
    MAX_PRODUCT = 998


class SpiderNode(object):
    def __init__(self, tag, shape, particles=[], state=None):
        self.tag = tag
        self.type = shape

        self.particles = particles
        self.state = state  # Matrix version of state for batch computations.
        self.weights = np.full(len(particles), 1.0 / len(particles))
        self.unaries = np.zeros(len(particles))

        if state is None and len(particles) > 0:
            D = 2 if shape == NodeType.CIRCLE else 3
            self.state = np.zeros((len(self.particles), D))

        assert self.state.shape[0] == len(self.particles)

    def update_particles(self):
        for i, p in enumerate(self.particles):
            p.set_state(*self.state[i].tolist())

    def update_unaries(self, obs):
        if self.type == NodeType.CIRCLE:
            r = self.particles[0].radius
            self.unaries = likelihoods.batch_circle_ave_score(obs, self.state, r, self.tag)
        elif self.type == NodeType.RECTANGLE:
            w, h = self.particles[0].w, self.particles[0].h
            self.unaries = likelihoods.batch_rect_ave_score(obs, self.state, w, h, self.tag)

    def unary(self, x_s, obs):
        return likelihoods.shape_ave_score(obs, x_s)

    def resample(self):
        choice = sampling.importance_sample(self.weights, keep_best=True)
        self.state = self.state[choice]

    def jitter(self, jitter_vars, keep_best=True):
        N, D = self.state.shape
        noise = np.stack([np.random.normal(0, jitter_vars[dim], size=N) for dim in range(D)], axis=-1)
        if keep_best:
            noise[0] = np.zeros(D)
        self.state = self.state + noise

    def estimate(self):
        return self.particles[self.weights.argmax()]


class SpiderGraph(object):

    def __init__(self, N, algo_type=AlgoType.SUM_PRODUCT, img_size=(640, 480)):
        self.img_size = img_size
        self.N = N
        self.jitter_vars = [10, 10, 0.1]
        self.w = 40
        self.h = 10
        self.radius = 10
        self.algo_type = algo_type

        self.nodes = [SpiderNode(1, NodeType.CIRCLE, [spider.Circle(self.radius, tag=1) for _ in range(N)])]
        self.nodes += [SpiderNode(i + 2, NodeType.RECTANGLE,
                                  [spider.Rectangle(self.w, self.h, tag=i + 2) for _ in range(N)]) for i in range(8)]

        self.particles_updated = True

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

        self.proposal_states = []
        self.proposal_unaries = []
        self.register_current()

    def register_current(self):
        self.proposal_states = [n.state for n in self.nodes]
        self.proposal_unaries = [n.unaries for n in self.nodes]

    def init_random_rects(self):
        states = [np.random.uniform(0, s, size=self.N) for s in self.img_size]
        states += [np.random.uniform(0, np.pi, size=self.N)]
        return np.stack(states, axis=-1)

    def init_random_circles(self):
        states = [np.random.uniform(0, s, size=self.N) for s in self.img_size]
        return np.stack(states, axis=-1)

    def init_random(self, obs):
        self.nodes[0].state = self.init_random_circles()
        self.nodes[0].update_particles()
        for i in range(1, len(self.nodes)):
            self.nodes[i].state = self.init_random_rects()
            self.nodes[i].update_particles()

        self.particles_updated = True
        self.update_unaries(obs)
        self.register_current()

    def obs_idx_select(self, obs, tag, noise=0):
        h, _ = obs.shape
        idx = (obs == tag).nonzero()
        idx = np.stack([idx[1], h - idx[0]], axis=-1)  # (row, col) = [x, y]
        choice = np.random.randint(0, idx.shape[0], size=self.N)
        idx = idx[choice]
        return idx + np.random.normal(0, noise, size=idx.shape)

    def init_obs(self, obs):
        circ_idx = self.obs_idx_select(obs, self.nodes[0].tag, noise=10)
        self.nodes[0].state = circ_idx
        self.nodes[0].update_particles()
        for i in range(1, len(self.nodes)):
            rect_idx = self.obs_idx_select(obs, self.nodes[i].tag, noise=10)
            rect_idx = np.concatenate([rect_idx, np.random.uniform(0, np.pi, size=(self.N, 1))], axis=-1)
            self.nodes[i].state = rect_idx
            self.nodes[i].update_particles()

        self.particles_updated = True
        self.update_unaries(obs)
        self.register_current()

    def update_unaries(self, obs):
        for n in self.nodes:
            n.update_unaries(obs)

    def update_messages(self, obs):
        self.update_unaries(obs)
        new_msgs = [np.zeros((len(n_edges), self.N)) for n_edges in self.edges]
        for s, n_edges in enumerate(self.edges):
            # Update the messages to node s from its neighbours, m_{t->s}
            for nbr_idx, t in enumerate(n_edges):
                edge_type = self.get_edge_type(t, s)
                s_to_t = self.get_nbr_idx(s, t)

                # Batched x_s is [(x_s,1 x N) ... (x_s,D x N)].T
                batch_xs = np.repeat(self.nodes[s].state, self.N, axis=0)        # (N * N, Ds)
                # Batched x_t is [(x_s,1 ... x_s,D) x N].T
                batch_xt = np.tile(self.proposal_states[t], (self.N, 1))         # (N * N, Dt)
                # Get the batch pairwise scores, a flattened version of the matrix
                # where P[t[i], s[j]] is the pairwise score between particles i
                # and j from nodes t and s respectively.
                batch_pair = self.batch_pairwise(batch_xs, batch_xt, edge_type)  # (N * N,)

                m_st = np.tile(self.messages[t][s_to_t, :], (self.N,))       # (N * N,)
                prop_unaries = np.tile(self.proposal_unaries[t], (self.N,))  # (N * N,)

                # Note: bel / bel_bar = p(z|xt).
                m_ts = batch_pair * prop_unaries / m_st                   # (N * N,)
                m_ts = m_ts.reshape(self.N, self.N)                       # (N, N)

                if self.algo_type == AlgoType.SUM_PRODUCT:
                    m_ts = m_ts.sum(axis=1) / self.N  # (N,)
                elif self.algo_type == AlgoType.MAX_PRODUCT:
                    m_ts = m_ts.max(axis=1)  # (N,)

                new_msgs[s][nbr_idx] = m_ts

        self.messages = new_msgs
        self.register_current()  # Register the current belief as the proposal.

    def update_belief(self, obs):
        for s, n_edges in enumerate(self.edges):
            bel_s = np.log(self.nodes[s].unaries) + np.log(self.messages[s]).sum(axis=0)
            self.nodes[s].weights = sampling.normalize_weights(bel_s, log=True)

    def resample(self):
        for n in self.nodes:
            n.resample()
        self.particles_updated = False

    def update_particles(self):
        for n in self.nodes:
            n.update_particles()
        self.particles_updated = True

    def marginals(self):
        if not self.particles_updated:
            self.update_particles()

        return [n.particles for n in self.nodes]

    def estimate(self):
        if not self.particles_updated:
            self.update_particles()

        root = self.nodes[0].estimate()
        links = [n.estimate() for n in self.nodes[1:]]
        return spider.Spider(root=root, links=links)

    def jitter(self):
        for n in self.nodes:
            n.jitter(self.jitter_vars)
        self.particles_updated = False

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

    def batch_pairwise(self, x_s, x_t, edge_type):
        # Edge type is from t->s.
        if edge_type == EdgeType.ARM_TO_ROOT:
            return likelihoods.batch_root_arm_pairwise(x_s, x_t, self.w, self.radius)
        elif edge_type == EdgeType.ROOT_TO_ARM:
            return likelihoods.batch_root_arm_pairwise(x_t, x_s, self.w, self.radius)
        elif edge_type == EdgeType.OUTER_TO_INNER:
            return likelihoods.batch_arm_arm_pairwise(x_s, x_t, self.w)
        elif edge_type == EdgeType.INNER_TO_OUTER:
            return likelihoods.batch_arm_arm_pairwise(x_t, x_s, self.w)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    scene = spider.SpiderScene(20, 5)
    obs = scene.observation()
    img = scene.image(obs)

    print("Initializing")
    g = SpiderGraph(100)
    # g.init_random()
    g.init_obs(obs)
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
