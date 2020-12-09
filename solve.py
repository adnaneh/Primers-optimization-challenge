# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 11:59:57 2019

@author: adnane
"""
import random
import numpy as np
import numba
# =============================================================================
# Reading input
# =============================================================================


def make_matrices(size, f):
    distances = []
    for i in range(size):
        newline = f.readline()
        data = newline.split(",")
        line = []
        for case in data:
            line.append(int(case))
        distances.append(line)
    return distances


def parsing():
    f = open("input_10.txt", "r")
    if f.mode != "r":
        raise FileNotFoundError
    size = int(f.readline())
    distances = make_matrices(size, f)
    junctions = make_matrices(size, f)
    return size, distances, junctions


if __name__ == "__main__":
    try:
        size, distances, junctions = parsing()

    except FileNotFoundError:
        print("file not found")
junctions = np.array(junctions)
distances = np.array(distances)

# =============================================================================
# Initialization
# =============================================================================


def init_graph():
    '''graphe des avions sauf les arrêtes de poids nulles'''
    # on intialise également le graphe
    # les sommets: les avions
    # les côtés, les transits
    # coût des côté: nombre de passagers à transiter
    graph = dict.fromkeys(range(size))
    for key in graph:
        graph[key] = {}
    for i in range(size):
        for j in range(size):
            if junctions[i][j] != 0:
                graph[i][j] = junctions[i][j]
    return(graph)

    # fonction de calcul de coût


def conditional_init_graph(sup, inf):
    '''initialisation du graph des avions, gardant seulement les arrêtes de poids tq inf<poids<sup
    pour garder toute les arrêtes sauf celles de poids nul on peut faire conditional_init_graph(100,0)'''
    # on intialise également le graphe
    # les sommets: les avions
    # les côtés, les transits
    # coût des côté: nombre de passagers à transiter
    graph = dict.fromkeys(range(size))
    for key in graph:
        graph[key] = {}
    for i in range(size):
        for j in range(size):
            if sup > junctions[i][j] > inf:
                graph[i][j] = junctions[i][j]
    return(graph)

    # fonction de calcul de coût


def init_map():
    '''on associe aux avions des portes de manière alétoire.'''
    map_planes_to_gate = list(range(size))

    random.shuffle(map_planes_to_gate)
    return(np.array(map_planes_to_gate))

# =============================================================================
# Utils
# =============================================================================


def calculate_cost(graph, map_planes_to_gate):
    '''calcul du coût d'une solution'''
    cost = 0
    for avion in graph:
        porte1 = map_planes_to_gate[avion]
        for voisin in graph[avion]:
            porte2 = map_planes_to_gate[voisin]
            distance = distances[porte1][porte2]
            cost += graph[avion][voisin] * distance
    return(cost)


def random_couple():
    '''retourne un couple aléatoire d'avions'''
    avion1 = random.randint(0, size - 1)
    avion2 = random.randint(0, size - 1)
    while avion2 == avion1:
        avion2 = random.randint(0, size - 1)
    return(avion1, avion2)


# =============================================================================
# Core functions
# =============================================================================

@numba.jit(nopython=True)
def cost_update_after_permutation(graph, map_planes_to_gate, avion1, avion2, distances):
    '''retourne la différence de coût si on fait la permutation des avions 1 et 2'''

    '''situation actuelle'''
    current_cost = 0

    '''cout de la situation actuelle pour le premier avion'''
    for voisin in range(size):
        gate = map_planes_to_gate[avion1]
        distance = distances[gate][map_planes_to_gate[voisin]]
        cout_temp = distance * (graph[avion1][voisin] + graph[voisin][avion1])
        current_cost += cout_temp
    '''cout de la situation actuelle pour le 2eme avion'''
    for voisin in range(size):
        if voisin != avion1:  # edge case
            gate = map_planes_to_gate[avion2]
            distance = distances[gate][map_planes_to_gate[voisin]]
            cout_temp = distance * \
                (graph[avion2][voisin] + graph[voisin][avion2])
            current_cost += cout_temp
    '''on fait l'échange temporaire des ports d'embarquement'''
    map_planes_to_gate[avion1], map_planes_to_gate[avion2] = map_planes_to_gate[avion2], map_planes_to_gate[avion1]

    '''situation après échange'''
    updated_cost = 0
    '''cout de la situation après échange pour le premier avion'''
    for voisin in range(size):
        gate = map_planes_to_gate[avion1]
        distance = distances[gate][map_planes_to_gate[voisin]]
        cout_temp = distance * (graph[avion1][voisin] + graph[voisin][avion1])
        updated_cost += cout_temp
    '''cout de la situation après échange pour le deuxieme avion'''
    for voisin in range(size):
        if voisin != avion1:  # edge case
            gate = map_planes_to_gate[avion2]
            distance = distances[gate][map_planes_to_gate[voisin]]
            cout_temp = distance * \
                (graph[avion2][voisin] + graph[voisin][avion2])
            updated_cost += cout_temp
    difference = updated_cost - current_cost
    '''on annule l'échange'''
    map_planes_to_gate[avion1], map_planes_to_gate[avion2] = map_planes_to_gate[avion2], map_planes_to_gate[avion1]

    return(difference)


def minimum_cost_permutation(graph, map_planes_to_gate, distances):
    '''on prend la permutation qui minimise le plus le coût ie hill climbing'''

    best_perm = None
    best_diff = None
    for avion1 in range(size):
        for avion2 in range(avion1, size):
            difference = cost_update_after_permutation(
                graph, map_planes_to_gate, avion1, avion2, distances)
            if best_diff == None:
                best_perm = avion1, avion2
                best_diff = difference
            elif difference < best_diff:
                best_perm = avion1, avion2
                best_diff = difference
    return(best_perm, best_diff)


# =============================================================================
# Step 1: hill climbing and restart until having a quite good solution, of cost<50000
# =============================================================================
'''ATTENTION: patience parfois mais c'est testé: normalement moins de 10 restart'''

best_cost = 10**6

# graph initialization
graph = init_graph()
while best_cost >= 44000:
    map_planes_to_gate = init_map()
    cost = calculate_cost(graph, map_planes_to_gate)
    print('initial', cost)

    for i in range(10**6):
        '''On tente une permutation au hasard, on la garde si elle améliore le coût'''
        avion1, avion2 = random_couple()
        difference = cost_update_after_permutation(
            junctions, map_planes_to_gate, avion1, avion2, distances)
    #    print(difference)
        perturbation = random.randint(1, 10000)
        if difference <= 0 or perturbation == 1:
            map_planes_to_gate[avion1], map_planes_to_gate[avion2] = map_planes_to_gate[avion2], map_planes_to_gate[avion1]
            cost = cost + difference
    #        if cost<100000:
            if cost < best_cost:
                best_cost = cost
                best_sol = map_planes_to_gate.copy()
                print('new opt', best_cost)
print('best etape 1', best_cost)
# =============================================================================
# Step 2: a variation of annealing
# =============================================================================


for i in range(10):
    counter = 0
    map_planes_to_gate = best_sol.copy()
    cost = calculate_cost(graph, map_planes_to_gate)
    for i in range(10**6):
        '''On tente une permutation au hasard, on la garde si elle améliore le coût'''
        avion1, avion2 = random_couple()
        difference = cost_update_after_permutation(
            junctions, map_planes_to_gate, avion1, avion2, distances)
    #    print(difference)
        perturbation = random.randint(1, 10000)
        if perturbation == 1:
            counter = 1
        if difference <= 0 or counter > 0:
            #        if counter >0:
            #            print(counter)
            counter -= 1
            map_planes_to_gate[avion1], map_planes_to_gate[avion2] = map_planes_to_gate[avion2], map_planes_to_gate[avion1]
            cost = cost + difference
    #        if cost<100000:
            if cost < best_cost:
                best_cost = cost
                best_sol = map_planes_to_gate.copy()
                print('new opt', best_cost)
    print('best etape 2', best_cost)
