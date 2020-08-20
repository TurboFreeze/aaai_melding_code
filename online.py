import numpy as np

def online_matching(graph, n):
    graph = graph.detach().numpy().copy()
    rho = 1 / (1 - (1 / np.exp(1)))
    matches = np.zeros((n, n))
    alphas = np.zeros(n)
    betas = np.zeros(n)
    rs = [np.random.rand() for _ in range(n)]
    deltas = [rho * np.exp(r - 1) for r in rs]
    for v in range(n):
        # take each vertex as they arrive online
        col = graph[:, v]
        if np.sum(col) == 0:
        	# no available neighbors
        	continue
        match = np.argmax((rho - deltas) * col)
        # print(match)
        alphas[match] = deltas[match]
        betas[v] = rho - deltas[match]
        matches[match, v] = 1
        graph[match] = np.zeros(n)
        
    return matches.flatten()#, alphas, betas
