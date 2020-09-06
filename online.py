import numpy as np

def online_matching(graph, n):
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

        alphas[match] = deltas[match]
        betas[v] = rho - deltas[match]
        matches[match, v] = 1
        graph[match] = np.zeros(n)
        
    return matches.flatten(), alphas, betas


def prop_alloc(graph, n, R, lam, eps=0.2):
    # http://proceedings.mlr.press/v80/agrawal18b/agrawal18b.pdf
    betas = [(1 + eps) ** -R for _ in range(n)]
    matches = np.zeros((n, n))
    # iterate through rounds
    for r in range(R):
        # i is impression, j is advertiser
        for i in range(n):
            # sum across advertisers for this impression
            imp_sum = np.sum([betas[j] * np.exp(graph[i, j] / lam - 1) for j in range(n)])
            for j in range(n):
                D = np.exp(graph[i, j] / lam - 1)
                if imp_sum > 1:
                    matches[i, j] = betas[j] * D / imp_sum
                else:
                    matches[i, j] = betas[j] * D

        for j in range(n):
            # calculate alloc
            alloc = np.sum(matches[:, j])
            if alloc <= 1 / (1 + eps):
                betas[j] *= 1 + eps
            if alloc >= 1 + eps:
                betas[j] /= 1 + eps


    # for i in range(n):
    #     if np.sum(matches[i, :]) > 1:
    #         print('badrowmeh', i)

    for j in range(n):
        for i in range(n):
            if np.sum(matches[:, j]) <= 1:
                break
            if matches[i, j] >= 1 / n:
                # reduce
                matches[i, j] -= min(matches[i, j], np.sum(matches[:, j]) - 1)

    assert(np.sum(matches[:, j]) <= 1)
    assert(round(np.sum(matches[:, j]), 5) <= 1)
    gammas = lam * np.log(1 / np.array(betas))  # Corollary 1 of PropAlloc Paper
    # Lemma 2 of PropAlloc Paper
    zsum = np.array([np.sum([np.exp(-gammas[a] / lam) * np.exp(graph[i, a] / lam - 1) 
        for a in range(n)]) 
        for i in range(n)])
    zs = np.array([lam * np.log(z) if z >= 1 else 0 for z in zsum])
    return matches.flatten(), np.concatenate([gammas, zs])

