import pandas as pd
import numpy as np
from pyldl.param_new_real import param_setting_alpha, param_setting_alpha_my
from pyldl.algorithms import IncomLDL, WInLDL, HidLDL, LDL_DPA, PT_Bayes, SA_IIS, LDL_LRR
from pyldl.utils import load_dataset, random_missing_real
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf

tf.experimental.numpy.experimental_enable_numpy_behavior()
metrics = ["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]
exponents = np.arange(-10, 11)

def run_single_experiment(X, y, missing_rate):
    y_missing, mask = random_missing_real(y, missing_rate=missing_rate)

    X_train, X_val, y_train, y_val, y_train_missing, y_val_missing, mask_train, mask_val = train_test_split(
        X, y, y_missing, mask, test_size=0.2, random_state=42)

    results = []

    incomldl = IncomLDL()
    best_alpha, best_score,best_result= param_setting_alpha(incomldl, X_train, y_train_missing, mask_train, y_train,X_val, y_val,
                                                2.0 ** exponents, 100)
    results.append(("IncomLDL", missing_rate, best_alpha, pd.Series(best_score, index=metrics)))
    results.append(("IncomLDL_result", missing_rate, best_alpha, pd.Series(best_result, index=metrics)))

    winldl = WInLDL()
    winldl.fit(X_train, y_train_missing, mask=mask_train)
    results.append(("WInLDL", missing_rate, None, pd.Series(winldl.score1(X_val, y_val), index=metrics)))

    hidLDL = HidLDL()
    best_alpha, best_score,best_result = param_setting_alpha_my(hidLDL, X_train, y_train_missing, mask_train,y_train, X_val, y_val, 2.0 ** exponents, 100)
    results.append(("HidLDL_best", missing_rate, best_alpha, pd.Series(best_score, index=metrics)))
    results.append(("HidLDL_best_result", missing_rate, best_alpha, pd.Series(best_result, index=metrics)))
    
    sa_IIS = SA_IIS()
    sa_IIS .fit(X_train, y_train_missing)
    results.append(("SA_IIS", missing_rate, None, pd.Series(sa_IIS.score1(X_val, y_val), index=metrics)))

    ldl_dpa = LDL_DPA()
    ldl_dpa.fit(X_train, y_train_missing)
    results.append(("LDL_DPA", missing_rate, None, pd.Series(ldl_dpa.score1(X_val, y_val), index=metrics)))

    pt_bayes = PT_Bayes()
    pt_bayes.fit(X_train, y_train_missing)
    results.append(("PT_Bayes", missing_rate, None, pd.Series(pt_bayes.score1(X_val, y_val), index=metrics)))

    ldl_LRR = LDL_LRR()
    ldl_LRR.fit(X_train, y_train_missing, beta=1e-2)
    results.append(("LDL_LRR", missing_rate, None, pd.Series(ldl_LRR.score1(X_val, y_val), index=metrics)))

    return results

def run_experiment(X, y, missing_rate, num_experiments=5):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_experiment, X, y, missing_rate) for _ in range(num_experiments)]
        results = [future.result() for future in futures]
    return [item for sublist in results for item in sublist]

def main(X, y):
    missing_rates = [0.4,0.5,0.6,0.7,0.8]
    all_results = []
    for rate in missing_rates:
        experiment_results = run_experiment(X, y, rate)
        all_results.extend(experiment_results)
    # 将结果转换为DataFrame并保存到Excel
    results_df = pd.DataFrame(all_results, columns=['Method', 'Missing Rate', 'Alpha', 'Scores'])
    results_df.to_excel('emotion6.xlsx', index=False)

if __name__ == '__main__':
    X, y = load_dataset('emotion6')  # 假设你有一个加载数据集的函数
    main(X, y)