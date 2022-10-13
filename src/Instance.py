'''
Created by: Quyen Linh TA
Username: qlinhta
Date: 14/12/2021
PyCharm 2021.3.1 (Professional Edition)
Licensed to Quyen Linh TA
'''

from graph import Graph
import queue


class Instance:
    def __init__(self, n, edges):
        self.graph = Graph(n, edges)
        self.be_cluster = [0] * self.graph.n()
        self.size_of_cluster = [0] * self.graph.n()
        self.candidate_clusters_adj = [0] * self.graph.n()
        self.n = self.graph.n()

        self.size_of_cluster[0] = self.n
        self.cluster_size_null = []
        for i in range(self.n):
            self.cluster_size_null.append(i)

        self.cost_calculate = (self.n * (self.n - 1)) / 2 - self.graph.m()

#I start with a random vertex alpha, a solution S, the equation returns a solution S'
#where vertex alpha is moved across a cluster C (is an empty cluster or nearest cluster)
#to reduce the cost of S0. It can therefore be implemented to run in O(d(v)time), where d(v)
#is the degree of v.

    def reinit_all_zero(self):
        self.size_of_cluster = [0] * len(self.size_of_cluster)
        self.be_cluster = [0] * len(self.be_cluster)
        self.size_of_cluster[0] = self.n

        self.cluster_size_null.clear()
        for i in range(self.n):
            self.cluster_size_null.append(i)
        self.cost_calculate = (self.n * (self.n - 1)) / 2 - self.graph.m()

    def reinit_state(self, v, cost):
        for i in range(self.n):
            self.size_of_cluster[i] = 0

        for i in range(self.n):
            self.be_cluster[i] = v[i]
            self.size_of_cluster[v[i]] += 1

        self.cost_calculate = cost
        self.cluster_size_null.clear()
        for i in range(self.n):
            if self.size_of_cluster[i] == 0:
                self.cluster_size_null.append(i)

    def get_zero_size(self):
        result = self.cluster_size_null[-1]

        while self.size_of_cluster[result] != 0 and len(self.cluster_size_null) != 0:
            self.cluster_size_null.pop()
            result = self.cluster_size_null[-1]

        return result

    def move(self, v, c):
        cv = self.be_cluster[v]
        if cv == c: return
        self.be_cluster[v] = c
        self.size_of_cluster[cv] -= 1
        self.size_of_cluster[c] += 1

        if self.size_of_cluster[cv] == 0:
            self.cluster_size_null.append(cv)

    def greedy_move(self, v):
        cv = self.be_cluster[v]

        for u in self.graph.neighbors(v):
            cu = self.be_cluster[u]
            self.candidate_clusters_adj[cu] += 1

        self_edges = self.candidate_clusters_adj[cv]
        best_cost = 0
        best_cluster = -1
        self_cost = -(self.size_of_cluster[cv] - 1 - 2 * self_edges)

        for u in self.graph.neighbors(v):
            cu = self.be_cluster[u]
            if cu == cv:
                continue

            cost = (self.size_of_cluster[cu] - 2 * self.candidate_clusters_adj[cu]) + self_cost

            if cost < best_cost:
                best_cost = cost
                best_cluster = cu

        for u in self.graph.neighbors(v):
            cu = self.be_cluster[u]
            self.candidate_clusters_adj[cu] = 0

        if self_cost < best_cost and len(self.cluster_size_null) != 0:
            best_cost = self_cost
            best_cluster = self.get_zero_size()

        if best_cluster == -1:
            return False

        self.move(v, best_cluster)
        self.cost_calculate += best_cost

        return True

    def residual_cost(self, v, c):
        cv = self.be_cluster[v]
        if cv == c:
            return 0

        self_edges = 0
        to_edges = 0

        for u in self.graph.neighbors(v):
            if self.be_cluster[u] == cv:
                self_edges += 1

            if self.be_cluster[u] == c:
                to_edges += 1

        return (self.size_of_cluster[c] - 2 * to_edges) - (self.size_of_cluster[cv] - 1 - 2 * self_edges)

    def move_residual_cost(self, v, c, residual):
        self.move(v, c)
        self.cost_calculate += residual

    def merge_to_null_cluster(self, vs):
        for v in vs:
            if self.size_of_cluster[self.be_cluster[v]] == 1:
                continue
            c = self.get_zero_size()
            self.move_residual_cost(v, c, self.residual_cost(v, c))

    def merge_to_same_cluster_null(self, vs):
        first_alone = self.size_of_cluster[self.be_cluster[vs[0]]] == 1
        c = first_alone if self.be_cluster[vs[0]] else self.get_zero_size()

        for i in range(first_alone, len(vs)):
            v = vs[i]
            self.move_residual_cost(v, c, self.residual_cost(v, c))

    def revert_cluster_of_with_cost(self, vs, old_cluster_of_vs, old_cost):
        for i in range(len(vs)):
            v = vs[i]
            c = old_cluster_of_vs[i]
            self.move(v, c)

        self.cost_calculate = old_cost

    def destroy_greedy_repair(self, vs, same_zero_size):
        if same_zero_size:
            self.merge_to_same_cluster_null(vs)
        else:
            self.merge_to_null_cluster(vs)

        for v in vs:
            self.greedy_move(v)

    def bfs_fill_vs(self, v, nv, vs, cluster_of_vs, seen):
        q = [v]
        while len(q) != 0 and len(vs) < nv:
            v = q.pop(0)
            if seen[v]:
                continue
            seen[v] = True
            vs.append(v)
            cluster_of_vs.append(self.be_cluster[v])
            q.extend(self.graph.neighbors(v))

    def m(self):
        return self.graph.m()

    def sol(self):
        return self.be_cluster

    def cost(self):
        return self.cost_calculate
