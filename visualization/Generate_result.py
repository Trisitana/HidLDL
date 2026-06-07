import pandas as pd
import os


def process_files(file_list, output_dir, missing_rate=0.5, metric='chebyshev'):
    # 定义方法
    methods = ['HidLDL_best_result', 'IncomLDL_result', 'WInLDL', 'SA_IIS', 'LDL_LRR', 'PT_Bayes', 'LDL_DPA']

    # 创建一个空的DataFrame来存储汇总结果
    summary_df = pd.DataFrame(columns=['Dataset'] + methods)

    # 遍历每个文件
    for file_name in file_list:
        # 读取原始数据
        df = pd.read_excel(file_name)

        # 提取数据集名称
        dataset_name = os.path.splitext(os.path.basename(file_name))[0]

        # 筛选出对应的数据
        subset = df[(df['Missing Rate'] == missing_rate)]

        # 创建一行结果
        result_row = {'Dataset': dataset_name}

        # 遍历每种方法
        for method in methods:
            method_subset = subset[subset['Method'] == method]
            if not method_subset.empty:
                # 解析Scores列
                scores_list = method_subset['Scores'].apply(lambda x: dict([y.split() for y in x.split('\n')]))
                scores_df = pd.DataFrame(scores_list.tolist())

                # 计算每个指标的平均值和标准差
                if metric in scores_df.columns:
                    values = scores_df[metric].astype(float)
                    mean = values.mean()
                    std = values.std()
                    mean_str = f'{mean:.4f}'
                    std_str = f'{std:.4f}'.lstrip('0')
                    result_row[method] = f'{mean_str}±{std_str}'
                else:
                    result_row[method] = 'NaN'
            else:
                result_row[method] = 'NaN'

        # 将结果添加到汇总DataFrame中
        summary_df = summary_df._append(result_row, ignore_index=True)

    # 保存汇总结果到CSV文件
    summary_file_name = f'summary_{metric}_{int(missing_rate * 100)}_missing.csv'
    summary_file_path = os.path.join(output_dir, summary_file_name)

    summary_df.to_csv(summary_file_path, index=False)
    print(f'Summary for {metric} at {missing_rate * 100}% missing rate saved to {summary_file_path}')


# 示例用法
directory = '../results/m_predict'
output_dir = '../results/m_predict/results_all/predict_50'
file_list = [os.path.join(directory, file) for file in
             ['Yeast_alpha.xlsx', 'Yeast_cdc.xlsx', 'Yeast_cold.xlsx', 'Yeast_dtt.xlsx', 'Yeast_elu.xlsx',
              'Yeast_spo.xlsx', 'SJAFFE.xlsx', 'scene.xlsx', 'Movie.xlsx', 'SBU_3DFE.xlsx', 'emotion6.xlsx',
              'RAF_ML.xlsx']]

process_files(file_list, output_dir, missing_rate=0.8, metric='intersection')