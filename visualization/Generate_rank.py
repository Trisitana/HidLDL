import pandas as pd
import os


def process_files(file_list, output_dir, missing_rate=0.5, metric='chebyshev'):
    # 定义方法
    methods = ['HidLDL_best', 'IncomLDL', 'WInLDL', 'SA_IIS', 'LDL_LRR', 'PT_Bayes', 'LDL_DPA']

    # 创建一个空的DataFrame来存储排名结果，每一行代表一个数据集，每一列代表一种方法
    rank_df = pd.DataFrame(columns=methods)

    # 遍历每个文件
    for file_name in file_list:
        # 读取原始数据
        df = pd.read_excel(file_name)

        # 提取数据集名称
        dataset_name = os.path.splitext(os.path.basename(file_name))[0]

        # 筛选出对应的数据
        subset = df[(df['Missing Rate'] == missing_rate)]

        # 创建一个临时字典来存储当前数据集下各方法的平均值
        method_means = {}

        # 遍历每种方法
        for method in methods:
            method_subset = subset[subset['Method'] == method]
            if not method_subset.empty:
                # 解析Scores列
                scores_list = method_subset['Scores'].apply(lambda x: dict([y.split() for y in x.split('\n')]))
                scores_df = pd.DataFrame(scores_list.tolist())

                # 计算每个指标的平均值
                if metric in scores_df.columns:
                    values = scores_df[metric].astype(float)
                    mean = values.mean()
                    method_means[method] = mean
                else:
                    method_means[method] = float('inf')  # 不存在该指标时，平均值设为无穷大
            else:
                method_means[method] = float('inf')  # 不存在该方法时，平均值设为无穷大

        # 对当前数据集下各方法的平均值进行排名，值越小排名越高
        sorted_methods = sorted(method_means, key=method_means.get)
        rank = {method: sorted_methods.index(method) + 1 for method in methods}

        # 将排名结果添加到rank_df中
        rank_df = rank_df._append(rank, ignore_index=True)

    # 添加数据集名称列
    dataset_names = [os.path.splitext(os.path.basename(file))[0] for file in file_list]
    rank_df.insert(0, 'Dataset', dataset_names)

    # 转置rank_df，使每一行代表一种方法，每一列代表一个数据集
    rank_df = rank_df.set_index('Dataset').T.reset_index()
    rank_df.rename(columns={'index': 'Method'}, inplace=True)

    # 保存排名结果到CSV文件
    rank_file_name = f'rank_{metric}_{int(missing_rate * 100)}_missing.csv'
    rank_file_path = os.path.join(output_dir, rank_file_name)
    rank_df.to_csv(rank_file_path, index=False)
    print(f'Rank for {metric} at {missing_rate * 100}% missing rate saved to {rank_file_path}')


# 示例用法
directory = '../results/m'
output_dir = '../results/m/results_rank'
file_list = [os.path.join(directory, file) for file in
             ['Yeast_alpha.xlsx', 'Yeast_cdc.xlsx', 'Yeast_cold.xlsx', 'Yeast_dtt.xlsx', 'Yeast_elu.xlsx',
              'Yeast_spo.xlsx', 'SJAFFE.xlsx', 'scene.xlsx', 'Movie.xlsx', 'SBU_3DFE.xlsx', 'emotion6.xlsx',
              'RAF_ML.xlsx',]]

process_files(file_list, output_dir, missing_rate=0.8, metric='canberra')