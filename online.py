import numpy as np
import torch
from torch.autograd import Function

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



def prop_alloc_combined(graph, n, R, lam, eps=0.2):
    torch.set_printoptions(precision=8)
    # http://proceedings.mlr.press/v80/agrawal18b/agrawal18b.pdf
    betas = [(1 + eps) ** -R for _ in range(n)]
    matches = np.zeros((n, n))
    graphnp = graph.detach().numpy().copy()
    betas_torch = [torch.tensor([(1 + eps) ** -R for _ in range(n)]) for _ in range(R + 1)]
    matches_torch = [torch.zeros((n, n)) for _ in range(R)]
    # iterate through rounds
    for r in range(R):
        if np.round(np.sum(np.abs(betas - betas_torch[r].detach().numpy())), 4) != 0:
            print(np.round(np.sum(np.abs(betas - betas_torch[r].detach().numpy())), 4))
            print(betas)
            print(betas_torch)
            assert(False)
        # i is impression, j is advertiser
        for i in range(n):
            # sum across advertisers for this impression
            imp_sum = np.sum([betas[j] * np.exp(graphnp[i, j] / lam - 1) for j in range(n)])
            for j in range(n):
                D = np.exp(graphnp[i, j] / lam - 1)
                if imp_sum > 1:
                    matches[i, j] = betas[j] * D / imp_sum
                else:
                    matches[i, j] = betas[j] * D


            imp_sum_torch = torch.dot(betas_torch[r], torch.exp(graph[i, :] / lam - 1))
            for j in range(n):
                D_torch = torch.exp(graph[i, j] / lam - 1)
                if imp_sum_torch > 1:
                    matches_torch[r][i, j] = betas_torch[r][j] * D_torch / imp_sum_torch
                else:
                    matches_torch[r][i, j] = betas_torch[r][j] * D_torch

            assert(np.round(imp_sum - imp_sum_torch.detach().numpy(), 3) == 0)

        for j in range(n):
            # calculate alloc
            alloc = np.sum(matches[:, j])
            if alloc <= 1 / (1 + eps):
                betas[j] *= 1 + eps
            if alloc >= 1 + eps:
                betas[j] /= 1 + eps


            alloc_torch = torch.sum(matches_torch[r][:, j])
            if alloc_torch <= 1 / (1 + eps):
                betas_torch[r + 1][j] = betas_torch[r][j] * (1 + eps)
            elif alloc_torch >= 1 + eps:
                betas_torch[r + 1][j] = betas_torch[r][j] / (1 + eps)
            else:
                betas_torch[r + 1][j] = betas_torch[r][j]

            if np.round(alloc - alloc_torch.detach().numpy(), 4) != 0:
                print(j, alloc, alloc_torch, 'BADDD')
                assert(False)

            if betas[j] - betas_torch[r + 1][j].detach().numpy() > 0.0001:
                print(r, j, alloc, alloc_torch, betas[j], betas_torch[r][j], betas_torch[r+1][j])
                assert(False)


            assert(np.round(np.sum(matches[:, j]) - np.sum(matches_torch[r][:, j].detach().numpy()), 4) == 0)
            assert(np.round(np.sum(matches[j, :]) - np.sum(matches_torch[r][j, :].detach().numpy()), 4) == 0)

    for j in range(n):
        for i in range(n):
            if np.sum(matches[:, j]) <= 1:
                break
            if matches[i, j] >= 1 / n:
                # reduce
                matches[i, j] -= min(matches[i, j], np.sum(matches[:, j]) - 1)

        assert(np.round(np.sum(matches[:, j]), 5) <= 1)
        assert(np.round(np.sum(matches[j, :]), 5) <= 1)

    gammas = lam * np.log(1 / np.array(betas))  # Corollary 1 of PropAlloc Paper
    # Lemma 2 of PropAlloc Paper
    zsum = np.array([np.sum([np.exp(-gammas[a] / lam) * np.exp(graph[i, a] / lam - 1) 
        for a in range(n)]) 
        for i in range(n)])
    zs = np.array([lam * np.log(z) if z >= 1 else 0 for z in zsum])
    return matches.flatten(), np.concatenate([gammas, zs])




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

    for j in range(n):
        for i in range(n):
            if np.sum(matches[:, j]) <= 1:
                break
            if matches[i, j] >= 1 / n:
                # reduce
                matches[i, j] -= min(matches[i, j], np.sum(matches[:, j]) - 1)

        assert(np.round(np.sum(matches[:, j]), 5) <= 1)
        assert(np.round(np.sum(matches[j, :]), 5) <= 1)

    gammas = lam * np.log(1 / np.array(betas))  # Corollary 1 of PropAlloc Paper
    # Lemma 2 of PropAlloc Paper
    zsum = np.array([np.sum([np.exp(-gammas[a] / lam) * np.exp(graph[i, a] / lam - 1) 
        for a in range(n)]) 
        for i in range(n)])
    zs = np.array([lam * np.log(z) if z >= 1 else 0 for z in zsum])
    return matches.flatten(), np.concatenate([gammas, zs])


def prop_alloc_auto(graph, n, R, lam, eps=0.2):
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=8)
    # http://proceedings.mlr.press/v80/agrawal18b/agrawal18b.pdf
    betas = [torch.tensor([(1 + eps) ** -R for _ in range(n)]) for _ in range(R + 1)]
    matches = [torch.zeros((n, n)) for _ in range(R)]
    # iterate through rounds
    for r in range(R):
        # i is impression, j is advertiser
        for i in range(n):
            # sum across advertisers for this impression
            imp_sum = torch.dot(betas[r], torch.exp(graph[i, :] / lam - 1))
            for j in range(n):
                D = torch.exp(graph[i, j] / lam - 1)
                if imp_sum > 1:
                    matches[r][i, j] = betas[r][j] * D / imp_sum
                else:
                    matches[r][i, j] = betas[r][j] * D

        for j in range(n):
            # calculate alloc
            alloc = torch.sum(matches[r][:, j])
            if alloc <= 1 / (1 + eps):
                betas[r + 1][j] = betas[r][j] * (1 + eps)
            elif alloc >= 1 + eps:
                betas[r + 1][j] = betas[r][j] / (1 + eps)
            else:
                betas[r + 1][j] = betas[r][j]

    for j in range(n):
        for i in range(n):
            if torch.sum(matches[-1][:, j]) <= 1:
                break
            if matches[-1][i, j] >= 1 / n:
                # reduce
                matches[-1][i, j] -= min(matches[-1][i, j], torch.sum(matches[-1][:, j]) - 1)

        assert(np.round(torch.sum(matches[-1][:, j]).detach().numpy(), 5) <= 1)
        assert(np.round(torch.sum(matches[-1][j, :]).detach().numpy(), 5) <= 1)

    return matches[-1].flatten()


def prop_alloc_single(graph, n, R, lam, eps=0.2):
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=8)
    # http://proceedings.mlr.press/v80/agrawal18b/agrawal18b.pdf
    betas = torch.tensor([(1 + eps) ** -R for _ in range(n)])
    matches = torch.zeros((n, n))
    # i is impression, j is advertiser
    for i in range(n):
        # sum across advertisers for this impression
        imp_sum = torch.dot(betas, torch.exp(graph[i, :] / lam - 1))
        for j in range(n):
            D = torch.exp(graph[i, j] / lam - 1)
            if imp_sum > 1:
                matches[i, j] = betas[j] * D / imp_sum
            else:
                matches[i, j] = betas[j] * D

    for j in range(n):
        for i in range(n):
            if torch.sum(matches[:, j]) <= 1:
                break
            if matches[i, j] >= 1 / n:
                # reduce
                matches[i, j] -= min(matches[i, j], torch.sum(matches[:, j]) - 1)

    return matches.flatten()



class PAFunction(Function):

    def __init__():
        pass

    def forward(self, graph):
        n = 50
        zhats = prop_alloc_auto(p.reshape(n, n), n, 10, 0.1, 0.2)
        self.save_for_backward(zhats, graph)
        return zhats

    def backward(self, dl_dzhats):
        zhats.backward(dl_dzhats)
        return 
