import random


def bfs_destroy_repair_ls_one_it_order_norand(g, n, ndestroy, order, seen, best_sol, best_cost, same_zero_size):
    vs = []
    old_cluster_of_vs = []

    seen = [False] * len(seen)
    random.shuffle(order)

    res = False

    for i in range(n):
        if seen[order[i]]:
            continue

        vs.clear()
        old_cluster_of_vs.clear()
        g.bfs_fill_vs(order[i], ndestroy, vs, old_cluster_of_vs, seen)

        if len(vs) == 0:
            continue

        random.shuffle(vs)
        for j in range(len(vs)):
            old_cluster_of_vs[j] = g.sol()[vs[j]]

        old_cost = g.cost()
        g.destroy_greedy_repair(vs, same_zero_size)

        if g.cost() < best_cost:
            res = True
            best_cost = g.cost()
            best_sol = g.sol()

        if g.cost() > old_cost:
            g.revert_cluster_of_with_cost(vs, old_cluster_of_vs, old_cost)

    return res
