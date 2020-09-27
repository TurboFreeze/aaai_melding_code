import torch
from torch.autograd import Function

from .util import bger, expandParam, extract_nBatch
from . import solvers
from .solvers.pdipm import batch as pdipm_b
from .solvers.pdipm import spbatch as pdipm_spb
# from .solvers.pdipm import single as pdipm_s

from enum import Enum


def make_gurobi_model(G, h, A, b, Q):
    import gurobipy as gp
    import numpy as np
    '''
    Convert to Gurobi model. Copied from 
    https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/gurobi_.py
    '''
    n = A.shape[1] if A is not None else G.shape[1]
    model = gp.Model()
    model.params.OutputFlag = 0
    x = [model.addVar(
            vtype=gp.GRB.CONTINUOUS,
            name='x_%d' % i,
            lb=-gp.GRB.INFINITY,
            ub=+gp.GRB.INFINITY)
        for i in range(n)
    ]
    model.update()   # integrate new variables

    # subject to
    #     G * x <= h
    inequality_constraints = []
    if G is not None:
        for i in range(G.shape[0]):
            row = np.where(G[i] != 0)[0]
            inequality_constraints.append(model.addConstr(gp.quicksum(G[i, j] * x[j] for j in row) <= h[i]))

    # subject to
    #     A * x == b
    equality_constraints = []
    if A is not None:
        for i in range(A.shape[0]):
            row = np.where(A[i] != 0)[0]
            equality_constraints.append(model.addConstr(gp.quicksum(A[i, j] * x[j] for j in row) == b[i]))

    obj = gp.QuadExpr()
    if Q is not None:
        rows, cols = Q.nonzero()
        for i, j in zip(rows, cols):
            obj += x[i] * Q[i, j] * x[j]

    return model, x, inequality_constraints, equality_constraints, obj

def forward_gurobi_prebuilt(Q, p, model, x, inequality_constraints, equality_constraints, G, h, quadobj):
    import gurobipy as gp
    import numpy as np
    obj = gp.QuadExpr()
    obj += quadobj
    for i in range(len(p)):
        obj += p[i] * x[i]
    model.setObjective(obj, gp.GRB.MINIMIZE)
    model.optimize()
    x_opt = np.array([x[i].x for i in range(len(x))])
    if G is not None:
        slacks = -(G@x_opt - h)
    else:
        slacks = np.array([])
    lam = np.array([inequality_constraints[i].pi for i in range(len(inequality_constraints))])
    nu = np.array([equality_constraints[i].pi for i in range(len(equality_constraints))])

    return model.ObjVal, x_opt, nu, lam, slacks

def forward_online_matching(Q, p, model, x, inequality_constraints, equality_constraints, quadobj):
    from online import online_matching
    import numpy as np
    n = 50
    x_opt, alphas, betas = online_matching(p.reshape(n, n), n)
    lam = np.concatenate([alphas, betas])
    return x_opt, lam

def forward_prop_alloc(p):
    from online import prop_alloc, prop_alloc_auto, prop_alloc_combined
    import numpy as np
    n = 50
    x_opt, gammas = prop_alloc_combined(p.view(n, n), n, 10, 0.1, 0.2)
    return x_opt, gammas


class QPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2
    GUROBI = 3
    CUSTOM = 4
    ONLINE = 5
    PROP = 6


class QPFunction(Function):
    def __init__(self, eps=1e-12, verbose=0, notImprovedLim=3,
                 maxIter=20, solver=QPSolvers.PDIPM_BATCHED, model_params = None, custom_solver=None):
        self.eps = eps
        self.verbose = verbose
        self.notImprovedLim = notImprovedLim
        self.maxIter = maxIter
        self.solver = solver
        self.custom_solver = custom_solver
#        self.constant_constraints = constant_constraints
        
        if model_params is not None:
#        if constant_constraints:
#            self.A = A
#            self.b = b
#            self.G = G
#            self.h = h
#            A_arg = A.detach().numpy() if A is not None else None
#            b_arg = b.detach().numpy() if b is not None else None
#            G_arg = G.detach().numpy() if G is not None else None
#            h_arg = h.detach().numpy() if h is not None else None
#            Q_arg = Q.detach().numpy() if Q is not None else None
#            model, x, inequality_constraints, equality_constraints, obj = make_gurobi_model(G_arg,
#                                                        h_arg, A_arg, b_arg, Q_arg)
            model, x, inequality_constraints, equality_constraints, obj = model_params
            self.model = model
            self.x = x
            self.inequality_constraints = inequality_constraints
            self.equality_constraints = equality_constraints
            self.quadobj = obj
            # print("QPFunction CREATION")
        else:
            self.model = None

    def forward(self, Q_, p_, G_, h_, A_, b_):
        """Solve a batch of QPs.

        This function solves a batch of QPs, each optimizing over
        `nz` variables and having `nineq` inequality constraints
        and `neq` equality constraints.
        The optimization problem for each instance in the batch
        (dropping indexing from the notation) is of the form

            \hat z =   argmin_z 1/2 z^T Q z + p^T z
                     subject to Gz <= h
                                Az  = b

        where Q \in S^{nz,nz},
              S^{nz,nz} is the set of all positive semi-definite matrices,
              p \in R^{nz}
              G \in R^{nineq,nz}
              h \in R^{nineq}
              A \in R^{neq,nz}
              b \in R^{neq}

        These parameters should all be passed to this function as
        Variable- or Parameter-wrapped Tensors.
        (See torch.autograd.Variable and torch.nn.parameter.Parameter)

        If you want to solve a batch of QPs where `nz`, `nineq` and `neq`
        are the same, but some of the contents differ across the
        minibatch, you can pass in tensors in the standard way
        where the first dimension indicates the batch example.
        This can be done with some or all of the coefficients.

        You do not need to add an extra dimension to coefficients
        that will not change across all of the minibatch examples.
        This function is able to infer such cases.

        If you don't want to use any equality or inequality constraints,
        you can set the appropriate values to:

            e = Variable(torch.Tensor())

        Parameters:
          Q:  A (nBatch, nz, nz) or (nz, nz) Tensor.
          p:  A (nBatch, nz) or (nz) Tensor.
          G:  A (nBatch, nineq, nz) or (nineq, nz) Tensor.
          h:  A (nBatch, nineq) or (nineq) Tensor.
          A:  A (nBatch, neq, nz) or (neq, nz) Tensor.
          b:  A (nBatch, neq) or (neq) Tensor.

        Returns: \hat z: a (nBatch, nz) Tensor.
        """
        # print("QPFunction FORWARD")
        nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
        Q, _ = expandParam(Q_, nBatch, 3)
        p, _ = expandParam(p_, nBatch, 2)
        G, _ = expandParam(G_, nBatch, 3)
        h, _ = expandParam(h_, nBatch, 2)
        A, _ = expandParam(A_, nBatch, 3)
        b, _ = expandParam(b_, nBatch, 2)

        _, nineq, nz = G.size()
        neq = A.size(1) if A.nelement() > 0 else 0
        assert(neq > 0 or nineq > 0)
        self.neq, self.nineq, self.nz = neq, nineq, nz

        if self.solver == QPSolvers.CVXPY:
            vals = torch.Tensor(nBatch).type_as(Q)
            zhats = torch.Tensor(nBatch, self.nz).type_as(Q)
            lams = torch.Tensor(nBatch, self.nineq).type_as(Q)
            nus = torch.Tensor(nBatch, self.neq).type_as(Q)
            slacks = torch.Tensor(nBatch, self.nineq).type_as(Q)
            for i in range(nBatch):
                Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                vals[i], zhati, nui, lami, si = solvers.cvxpy.forward_single_np(
                    *[x.cpu().detach().numpy() if x is not None else None
                      for x in (Q[i], p[i], G[i], h[i], Ai, bi)])
                # if zhati[0] is None:
                #     import IPython, sys; IPython.embed(); sys.exit(-1)
                zhats[i] = torch.Tensor(zhati)
                lams[i] = torch.Tensor(lami)
                slacks[i] = torch.Tensor(si)
                if neq > 0:
                    nus[i] = torch.Tensor(nui)

            self.vals = vals
            self.lams = lams
            self.nus = nus
            self.slacks = slacks
        elif self.solver == QPSolvers.GUROBI:
            vals = torch.Tensor(nBatch).type_as(Q)
            zhats = torch.Tensor(nBatch, self.nz).type_as(Q)
            lams = torch.Tensor(nBatch, self.nineq).type_as(Q)
            if self.neq > 0:
                nus = torch.Tensor(nBatch, self.neq).type_as(Q)
            else:
                nus = torch.Tensor().type_as(Q)
            slacks = torch.Tensor(nBatch, self.nineq).type_as(Q)
            for i in range(nBatch):
                Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                if self.model is None:
                    vals[i], zhati, nui, lami, si = forward_single_np_gurobi(
                        *[x.cpu().detach().numpy() if x is not None else None
                          for x in (Q[i], p[i], G[i], h[i], Ai, bi)])
                else:
                    Gi = G[i].detach().numpy() if G is not None else None
                    hi = h[i].detach().numpy() if h is not None else None
                    vals[i], zhati, nui, lami, si = forward_gurobi_prebuilt(
                            Q[i].detach().numpy(), p[i].detach().numpy(), self.model, self.x, self.inequality_constraints, 
                            self.equality_constraints, Gi, hi, self.quadobj)
                # if zhati[0] is None:
                #     import IPython, sys; IPython.embed(); sys.exit(-1)
                zhats[i] = torch.Tensor(zhati)
                lams[i] = torch.Tensor(lami)
                slacks[i] = torch.Tensor(si)
                if neq > 0:
                    nus[i] = torch.Tensor(nui)

            self.vals = vals
            self.lams = lams
            self.nus = nus
            self.slacks = slacks
        elif self.solver == QPSolvers.ONLINE:
            p = -p
            vals = torch.Tensor(nBatch).type_as(Q)  # not really necessary
            zhats = torch.Tensor(nBatch, self.nz).type_as(Q)
            lams = torch.Tensor(nBatch, self.nineq).type_as(Q)
            if self.neq > 0:
                nus = torch.Tensor(nBatch, self.neq).type_as(Q)
            else:
                nus = torch.Tensor().type_as(Q)
            slacks = torch.Tensor(nBatch, self.nineq).type_as(Q)
            # for i in range(nBatch):
            #     Gi = G[i].detach().numpy() if G is not None else None
            #     hi = h[i].detach().numpy() if h is not None else None

            #     vals[i], zhati, lami = forward_online_matching(
            #             Q[i].detach().numpy(), p[i].detach().numpy(), self.model, self.x, self.inequality_constraints, 
            #             self.equality_constraints, self.quadobj)

            #     zhats[i] = torch.Tensor(zhati)
            #     slacks[i] = torch.Tensor(-(Gi@zhati - hi))
            for i in range(nBatch):
                Gi = G[i].detach().numpy() if G is not None else None
                hi = h[i].detach().numpy() if h is not None else None
                zhati, lami = forward_online_matching(
                        Q[i].detach().numpy(), p[i].detach().numpy(), self.model,
                        self.x, self.inequality_constraints, 
                        self.equality_constraints, self.quadobj)
                zhats[i] = torch.Tensor(zhati)
                lams[i] = torch.cat((torch.Tensor(lami), torch.zeros(self.nineq - len(lami))))
                slacks[i] = torch.Tensor(-(Gi@zhati - hi))

            self.vals = vals
            self.lams = lams
            self.nus = nus
            self.slacks = slacks
        elif self.solver == QPSolvers.PROP:
            p = -p 
            zhats = torch.Tensor(nBatch, self.nz).type_as(Q)
            lams = torch.Tensor(nBatch, self.nineq).type_as(Q)
            if self.neq > 0:
                nus = torch.Tensor(nBatch, self.neq).type_as(Q)
            else:
                nus = torch.Tensor().type_as(Q)
            slacks = torch.Tensor(nBatch, self.nineq).type_as(Q)
            for i in range(nBatch):
                Gi = G[i].detach().numpy() if G is not None else None
                hi = h[i].detach().numpy() if h is not None else None
                zhati, lami = forward_prop_alloc(p[i])
                zhats[i] = torch.Tensor(zhati)
                lams[i] = torch.cat((torch.Tensor(lami), torch.zeros(self.nineq - len(lami))))
                slacks[i] = torch.Tensor(-(Gi@zhati - hi))
                # Q[i] = torch.diag(torch.Tensor(-0.1 / zhati))

            self.vals = torch.Tensor(nBatch).type_as(Q)
            self.lams = lams
            self.nus = nus
            self.slacks = slacks
        else:
            assert False

        self.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
        return zhats

    def backward(self, dl_dzhat):
        zhats, Q, p, G, h, A, b = self.saved_tensors
        nBatch = extract_nBatch(Q, p, G, h, A, b)
        Q, Q_e = expandParam(Q, nBatch, 3)
        p, p_e = expandParam(p, nBatch, 2)
        G, G_e = expandParam(G, nBatch, 3)
        h, h_e = expandParam(h, nBatch, 2)
        A, A_e = expandParam(A, nBatch, 3)
        b, b_e = expandParam(b, nBatch, 2)

        # neq, nineq, nz = self.neq, self.nineq, self.nz
        neq, nineq = self.neq, self.nineq


        if self.solver != QPSolvers.PDIPM_BATCHED:
            self.Q_LU, self.S_LU, self.R = pdipm_b.pre_factor_kkt(Q, G, A)

        # Clamp here to avoid issues coming up when the slacks are too small.
        # TODO: A better fix would be to get lams and slacks from the
        # solver that don't have this issue.
        d = torch.clamp(self.lams, min=1e-8) / torch.clamp(self.slacks, min=1e-8)

        pdipm_b.factor_kkt(self.S_LU, self.R, d)
        dx, _, dlam, dnu = pdipm_b.solve_kkt(
            self.Q_LU, d, G, A, self.S_LU,
            dl_dzhat, torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor())

        dps = dx
        dGs = bger(dlam, zhats) + bger(self.lams, dx)
        if G_e:
            dGs = dGs.mean(0).squeeze(0)
        dhs = -dlam
        if h_e:
            dhs = dhs.mean(0).squeeze(0)
        if neq > 0:
            dAs = bger(dnu, zhats) + bger(self.nus, dx)
            dbs = -dnu
            if A_e:
                dAs = dAs.mean(0).squeeze(0)
            if b_e:
                dbs = dbs.mean(0).squeeze(0)
        else:
            dAs, dbs = None, None
        dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
        if Q_e:
            dQs = dQs.mean(0).squeeze(0)
        if p_e:
            dps = dps.mean(0).squeeze(0)

        grads = (dQs, dps, dGs, dhs, dAs, dbs)

        return grads

