import pandas as pd
import numpy as np
from pyldl.param2 import param_setting_alpha,param_setting_alpha_my
from pyldl.algorithms import IncomLDL, WInLDL, HidLDL, SA_IIS, PT_Bayes, LDL_DPA,LDL_LRR
from pyldl.utils import load_dataset, random_missing_real

X, y = load_dataset('emotion6')


metrics = ["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]


def run_experiment(X, y, missing_rate):


    results = []
    for _ in range(5):  # 每个缺失率下循环3次
        y_missing, mask = random_missing_real(y, missing_rate=missing_rate)
        incomldl = IncomLDL()
        best_alpha, best_score = param_setting_alpha(incomldl, X, y, y_missing, mask, np.geomspace(2**-10, 2**10, num=21), 100)
        results.append(("IncomLDL", missing_rate, best_alpha, pd.Series(best_score, index=metrics)))

        winldl = WInLDL()
        winldl.fit(X, y_missing, mask=mask)
        results.append(("WInLDL", missing_rate, None, pd.Series(winldl.score(X, y), index=metrics)))

        hidLDL = HidLDL()
        best_alpha, best_score, all_scores = param_setting_alpha_my(hidLDL, X, y, y_missing, mask, np.geomspace(2**-10, 2**10, num=21), 100)
        results.append(("hidLDL_best", missing_rate, best_alpha, pd.Series(best_score, index=metrics)))
        for key, value in all_scores.items():
            results.append(("hidLDL", missing_rate, key, pd.Series(value, index=metrics)))

        ldl_dpa = LDL_DPA()
        ldl_dpa.fit(X, y_missing)
        results.append(("LDL_DPA", missing_rate, None, pd.Series(ldl_dpa.score(X, y), index=metrics)))

        ldl_lrr = LDL_LRR()
        ldl_lrr.fit(X, y_missing,beta=1e-2)
        results.append(("LDL_LRR", missing_rate, None, pd.Series(ldl_lrr.score(X, y), index=metrics)))

        sa_IIS = SA_IIS()
        sa_IIS.fit(X, y_missing)
        results.append(("SA_IIS", missing_rate, None, pd.Series(sa_IIS.score(X, y), index=metrics)))

        pt_bayes = PT_Bayes()
        pt_bayes.fit(X, y_missing)
        results.append(("PT_Bayes", missing_rate, None, pd.Series(pt_bayes.score(X, y), index=metrics)))

    return results


def main(X, y):
    missing_rates = [0.4, 0.5, 0.6, 0.7, 0.8]
    all_results = []

    for rate in missing_rates:
        experiment_results = run_experiment(X, y, rate)
        all_results.extend(experiment_results)

    # 将结果转换为DataFrame并保存到Excel
    results_df = pd.DataFrame(all_results, columns=['Method', 'Missing Rate', 'Alpha', 'Scores'])
    results_df.to_excel('emotion6.xlsx', index=False)


main(X, y)
# 假设X和y是你的数据集
# main(X, y)
