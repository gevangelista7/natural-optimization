import numpy.random as rd
from utils import generate_point_cloud_with_optimum
from instrumented_DA_torch import instrumented_DA_torch
import time
import torch


def statistical_verification_torch(parameters, NC, P, n_rodadas, algo_res_database, stat_verify_database):
    eps, delta, Tmin, max_iterations, T0, alpha = parameters
    test_results = []
    stat_vrf_start = time.time()

    for i in range(n_rodadas):
        print("Teste {} de {}".format(i+1, n_rodadas))
        instant_seed = int(time.time() * 1e10 % 15461)  # randon seed
        rd.seed(instant_seed)
        data_vectors, tgtJ, tgtD, _ = generate_point_cloud_with_optimum(n_clusters=NC, core_points=P, cores_dispersion=NC, dimension=2)
        data_vectors = torch.tensor(data_vectors, device='cuda', dtype=torch.float64)

        result = instrumented_DA_torch(data_vectors=data_vectors, NC=NC, T0=T0, Tmin=Tmin,
                                       max_iterations=max_iterations, alpha=alpha, eps=eps, delta=delta,
                                       target_J=tgtJ, seed=instant_seed)

        test_results.append(result)

    elapsed_time = time.time() - stat_vrf_start

    satisfactory_results = [r['satisfactory_result'] for r in test_results]
    success_rate = satisfactory_results.count(True)/len(satisfactory_results)

    mean_J = torch.mean(torch.tensor([r['final_J'] for r in test_results]))
    mean_dist2tgt = torch.mean(torch.tensor([r['target_J'] - r['final_J'] for r in test_results]))

    stat_res = {
        'NC': NC,
        'T0': T0,
        'max_iterations': max_iterations,
        'alpha': alpha,
        'eps': eps,
        'delta': delta,
        'success_rate': success_rate,
        'elapsed_time': elapsed_time,
        'mean_J': mean_J.item(),
        'mean_dist2tgt': mean_dist2tgt.item()
    }

    algo_res_database.data_entry(level_results=test_results, fieldnames=result.keys())
    stat_verify_database.data_entry(level_results=[stat_res], fieldnames=stat_res.keys())

    return success_rate



