from debiased_jadouille.mitigation.inprocessing.schreuder_repo.lp_builder import build_A, build_c, build_b, build_l, build_u
from scipy.optimize import linprog

"""post-processing
    References:
        Schreuder, N., & Chzhen, E. (2021, December). Classification with abstention but without disparities. In Uncertainty in Artificial Intelligence (pp. 1227-1236). PMLR.
        https://github.com/evgchz/dpabst
"""

def solve_lp(ps, ns, alphas, pred_prob):
    A = build_A(ps, ns, alphas)
    c = build_c(ns, alphas)
    b = build_b(ps, alphas, ns, pred_prob)
    l = build_l(ns)
    u = build_u(ns)
    res = linprog(c, A_ub=A, b_ub=b,
                  bounds=list(zip(l, u)),
                  method='highs')
    return(res)