import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from vicinal_classifier import adv_ce, adv_ce_tim, adv_svm

def test(cls_cfg, classifier):
    # prepare data
    dataset = 'miniImagenet'
    n_shot = cls_cfg['n_shot']
    n_ways = cls_cfg['n_ways']
    n_queries = 15
    n_runs = 10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    if cls_cfg['l2n']:
        ndatas = F.normalize(ndatas, dim=2)


    # ---- classification for each task

    test_accuracies = []

    print('Start classification for %d tasks...' % (n_runs))
    for i in tqdm(range(n_runs)):

        batch_idx = i

        support_data = np.copy(ndatas[batch_idx][:n_lsamples].numpy())
        support_label = np.copy(labels[batch_idx][:n_lsamples].numpy())
        query_data = np.copy(ndatas[batch_idx][n_lsamples:].numpy())
        query_label = np.copy(labels[batch_idx][n_lsamples:].numpy())

        if cls_cfg['tukey']:
            beta = 0.5
            support_data = np.power(np.absolute(support_data), beta) * np.sign(support_data)
            query_data = np.power(np.absolute(query_data), beta) * np.sign(query_data)

        if cls_cfg['rw_step'] < 1:
            raise ValueError('The rw_step must be >= 1')

        if classifier == 'ADV-CE':    ### CE + GD + vrm
            acc = adv_ce(support_data, support_label, query_data, query_label, cls_cfg)
        elif classifier == 'ADV-CE-TIM':  ### CE + GD + vrm + ti
            acc = adv_ce_tim(support_data, support_label, query_data, query_label, cls_cfg, cls_cfg['tim_para'])
        elif classifier == 'ADV-SVM':   ### KSVM + VRM
            acc = adv_svm(support_data, support_label, query_data, query_label, cls_cfg, gamma=cls_cfg['gamma'], maxIter=15)
        else:
            raise ValueError("No supported classifier. Choose 'ADV-CE' or 'ADV-SVM'.")

        acc = acc * 100

        test_accuracies.append(acc)

        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)

        if (i +1) % 50 == 0:
            print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)' \
                  .format(i+1, n_runs, avg, ci95, acc))

    return avg, ci95