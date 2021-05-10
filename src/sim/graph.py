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
    def __init__(self, tag, shape, particles=[], state=None):
        self.tag = tag
        self.type = shape

        self.particles = particles
        self.state = state  # Matrix version of state for batch computations.
        self.weights = [1.0 / len(particles) for _ in particles]
        self.unaries = np.zeros(len(particles))

        if state is None and len(particles) > 0:
            D = 2 if shape == NodeType.CIRCLE else 3
            self.state = np.zeros((len(self.particles), D))

        assert self.state.shape[0] == len(self.particles)

    def set_state(self, state):
        self.state = state
        for i, p in enumerate(self.particles):
            p.set_state(*self.state[i].tolist())

    def update_unaries(self, obs):
        unaries = []
        for p in self.particles:
            unaries.append(self.unary(p, obs))

        self.unaries = np.array(unaries)

    def unary(self, x_s, obs):
        return likelihoods.shape_ave_score(obs, x_s)

    def resample(self):
        choice = sampling.importance_sample(self.weights)
        self.state = self.state[choice]
        self.particles = [self.particles[idx] for idx in choice]

    def jitter(self, jitter_vars):
        N, D = self.state.shape
        state = self.state + np.stack([np.random.normal(0, jitter_vars[dim], size=N) for dim in range(D)], axis=-1)
        self.set_state(state)


class SpiderGraph(object):

    def __init__(self, N, img_size=(640, 480)):
        self.img_size = img_size
        self.N = N
        self.jitter_vars = [10, 10, 0.1]
        self.w = 40
        self.h = 10
        self.radius = 10

        self.nodes = [SpiderNode(1, NodeType.CIRCLE, [spider.Circle(self.radius, tag=1) for _ in range(N)])]
        self.nodes += [SpiderNode(i + 2, NodeType.RECTANGLE,
                                  [spider.Rectangle(self.w, self.h, tag=i + 2) for _ in range(N)]) for i in range(8)]

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

        self.register_current()

    def register_current(self):
        self.proposal_states = [n.state for n in self.nodes]
        self.proposal_unaries = [n.unaries for n in self.nodes]
        self.proposal_particles = [n.particles for n in self.nodes]

    def init_random_rects(self):
        states = [np.random.uniform(0, s, size=self.N) for s in self.img_size]
        states += [np.random.uniform(0, np.pi, size=self.N)]
        return np.stack(states, axis=-1)

    def init_random_circles(self):
        states = [np.random.uniform(0, s, size=self.N) for s in self.img_size]
        return np.stack(states, axis=-1)

    def init_random(self, obs):
        self.nodes[0].set_state(self.init_random_circles())
        for i in range(1, len(self.nodes)):
            self.nodes[i].state = self.init_random_rects()
            self.nodes[i].set_state(self.init_random_rects())

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
        self.nodes[0].set_state(circ_idx)
        for i in range(1, len(self.nodes)):
            rect_idx = self.obs_idx_select(obs, self.nodes[i].tag, noise=10)
            rect_idx = np.concatenate([rect_idx, np.random.uniform(0, np.pi, size=(self.N, 1))], axis=-1)
            self.nodes[i].set_state(rect_idx)

        self.update_unaries(obs)
        self.register_current()

    def update_unaries(self, obs):
        for n in self.nodes:
            n.update_unaries(obs)

    def update_messages_slow(self, obs):
        self.update_unaries(obs)
        new_msgs = [np.zeros((len(n_edges), self.N)) for n_edges in self.edges]
        for s, n_edges in enumerate(self.edges):
            # Update the messages to node s from its neighbours, m_{t->s}
            for nbr_idx, t in enumerate(n_edges):
                edge_type = self.get_edge_type(t, s)
                for i, x_s in enumerate(self.nodes[s].particles):
                    m_ts = 0
                    for j, x_t in enumerate(self.proposal_particles[t]):
                        m_st = self.messages[t][self.get_nbr_idx(s, t), j]
                        # Note: bel / bel_bar = p(z|xt).
                        m_ts += self.pairwise(x_s, x_t, edge_type) * self.proposal_unaries[t][j] / m_st

                    new_msgs[s][nbr_idx, i] = m_ts * 1.0 / self.N

        # print("range", [(m.min(), m.max()) for m in new_msgs])
        # print("NaNs?", [np.any(np.isnan(m)) for m in new_msgs])

        self.messages = new_msgs
        self.register_current()  # Register the current belief as the proposal.

    def update_messages(self, obs):
        self.update_unaries(obs)
        new_msgs = [np.zeros((len(n_edges), self.N)) for n_edges in self.edges]
        for s, n_edges in enumerate(self.edges):
            # Update the messages to node s from its neighbours, m_{t->s}
            for nbr_idx, t in enumerate(n_edges):
                edge_type = self.get_edge_type(t, s)
                s_to_t = self.get_nbr_idx(s, t)

                batch_xs = np.repeat(self.nodes[s].state, self.N, axis=0)
                batch_xt = np.tile(self.proposal_states[t], (self.N, 1))
                batch_pair = self.batch_pairwise(batch_xs, batch_xt, edge_type)

                m_st = np.tile(self.messages[t][s_to_t, :], (self.N,))
                prop_unaries = np.tile(self.proposal_unaries[t], (self.N,))

                m_ts = batch_pair * prop_unaries / m_st
                m_ts = m_ts.reshape(self.N, self.N).sum(axis=1) / self.N

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
    g = SpiderGraph(50)
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
