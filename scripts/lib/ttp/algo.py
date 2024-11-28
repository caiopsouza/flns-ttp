import numpy as np

from lib.ttp import Problem, Solution

import pandas as pd
from scipy.sparse.csgraph import dijkstra


def investigate_node(solution, node_investigating_label):
    node_investigating = solution.node_index_by_label(node_investigating_label)

    labels = [node.label for node in solution.problem.nodes]

    node_investigating_obj = solution.route[node_investigating]

    dijkstra_dist = dijkstra(csgraph=solution.problem.get_matrix(), directed=False, indices=node_investigating_obj.i)
    plane_dist = [dijkstra_dist[node.i] for node in solution.problem.nodes]

    route_dist = [solution.distance_nodes_labels(node_investigating_label, label) for label in labels]

    dists = pd.DataFrame({'label': labels, 'dist plane': plane_dist, 'dist route': route_dist})

    dists['dist diff'] = dists['dist route'] - dists['dist plane']

    # dists['relative index'] = dists.index - node_investigating

    dists['dist ratio'] = dists['dist diff'] / dists['dist plane']
    dists['ratio x weight'] = node_investigating_obj.weight() * dists['dist ratio']

    dists = dists.sort_values(by='dist ratio', ascending=False)

    return dists


def calculate_ratios(solution):
    ratios = []
    for item in solution.problem.items:
        # if sol.items[item.i]:
        inv = investigate_node(solution, item.node.label)
        inv = inv.iloc[0]
        ratios.append((item.weight * inv['dist ratio'], item.node.label, int(inv['label'])))

    ratios.sort(reverse=True)
    return ratios


def improve_solution(solutions):
    for ratio in calculate_ratios(solutions[-1]):
        # print('trying', ratio)

        solution = solutions[-1].clone()
        node = solution.find_node_by_label(ratio[1])

        node_candidate = solution.find_node_by_label(ratio[2])
        node_candidate_index = solution.route.index(node_candidate)

        if node_candidate_index == 0:
            pass
        elif node_candidate_index == len(solution.route):
            node_candidate_index -= 1
        else:
            node_before = solution.route[node_candidate_index - 1]
            node_after = solution.route[node_candidate_index + 1]

            node_before_dist = node.distance(node_before)
            node_after_dist = node.distance(node_after)

            if node_before_dist < node_after_dist:
                node_candidate_index -= 1

        solution.change_route(solution.route[node_candidate_index].label, node.label)
        solution.local_search()

        if solution.fitness() > solutions[-1].fitness():
            solutions.append(solution)
            return True

    return False


def improve_solution2(solutions, chunks):
    solution = solutions[-1].clone()

    for ratio in chunks:
        # print('trying', ratio)

        node = solution.find_node_by_label(ratio[1])

        node_candidate = solution.find_node_by_label(ratio[2])
        node_candidate_index = solution.route.index(node_candidate)

        if node_candidate_index == 0:
            pass
        elif node_candidate_index == len(solution.route):
            node_candidate_index -= 1
        else:
            node_before = solution.route[node_candidate_index - 1]
            node_after = solution.route[node_candidate_index + 1]

            node_before_dist = node.distance(node_before)
            node_after_dist = node.distance(node_after)

            if node_before_dist < node_after_dist:
                node_candidate_index -= 1

        solution.change_route(solution.route[node_candidate_index].label, node.label)

    solution.local_search()

    if solution.fitness() > solutions[-1].fitness():
        solutions.append(solution)
        return True

    return False


def improve_solution3(solutions, chunk_size):
    chunks = calculate_ratios(solutions[-1])
    chunks = np.array_split(chunks, chunk_size)

    for ratios in chunks:
        solution = solutions[-1].clone()

        for ratio in ratios:
            node = solution.find_node_by_label(ratio[1])

            node_candidate = solution.find_node_by_label(ratio[2])
            node_candidate_index = solution.route.index(node_candidate)

            if node_candidate_index == 0:
                pass
            elif node_candidate_index == len(solution.route):
                node_candidate_index -= 1
            else:
                node_before = solution.route[node_candidate_index - 1]
                node_after = solution.route[node_candidate_index + 1]

                node_before_dist = node.distance(node_before)
                node_after_dist = node.distance(node_after)

                if node_before_dist < node_after_dist:
                    node_candidate_index -= 1

            solution.change_route(solution.route[node_candidate_index].label, node.label)

        solution.local_search()

        if solution.fitness() > solutions[-1].fitness():
            solutions.append(solution)
            return True

    return False


def improve_solution4(solutions, chunk_size):
    chunks = calculate_ratios(solutions[-1])
    chunks = np.array_split(chunks, len(chunks) // chunk_size)

    solution = solutions[-1].clone()
    for chunk in chunks:
        solution.clone_from(solutions[-1])
        for ratio in chunk:
            node = solution.find_node_by_label(ratio[1])

            node_candidate = solution.find_node_by_label(ratio[2])
            node_candidate_index = solution.route.index(node_candidate)

            if node_candidate_index == 0:
                pass
            elif node_candidate_index == len(solution.route):
                node_candidate_index -= 1
            else:
                node_before = solution.route[node_candidate_index - 1]
                node_after = solution.route[node_candidate_index + 1]

                node_before_dist = node.distance(node_before)
                node_after_dist = node.distance(node_after)

                if node_before_dist < node_after_dist:
                    node_candidate_index -= 1

            solution.change_route(solution.route[node_candidate_index].label, node.label)

        solution.local_search()

        if solution.fitness() > solutions[-1].fitness():
            solutions.append(solution)
            return True

    return False


def test_improve_solution(solutions, chunk_size):
    ratios = calculate_ratios(solutions[-1])
    ratios = np.array_split(ratios, chunk_size)
    improved = False

    for chunks in ratios:
        if improve_solution2(solutions, chunks):
            improved = True

    return improved


def solve(name, category, number) -> [Solution]:
    problem = Problem(name, category, number)

    solution = Solution(problem, problem.solve_tsp())
    solution.pick_from_dist_in_route()

    solution.local_search()
    solutions = [solution]

    improved = True
    chunks = [1, 2, 3, 5, 8, 13, 21, 34]

    # while improved:
    #     improved = False
    #
    #     for chunk in chunks:
    #         print('chunk', chunk)
    #         while test_improve_solution(solutions, chunk):
    #             improved = True
    #             print('improved:', solutions[-1].fitness())

    print('start:', solutions[-1].fitness())
    while improved:
        improved = False

        for chunk in chunks:
            print('chunk', chunk)
            while improve_solution4(solutions, chunk):
                print('improved:', solutions[-1].fitness())
                improved = True

    # while improved:
    #     improved = False
    #
    #     for chunk in chunks:
    #         print('chunk', chunk)
    #         while improve_solution3(solutions, chunk):
    #             print('improved:', solutions[-1].fitness())
    #             improved = True
    #
    return solutions
