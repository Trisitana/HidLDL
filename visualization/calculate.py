import pandas as pd
import os
def process_files(file_list):
    # 定义评价指标
    metrics = ['chebyshev', 'clark', 'canberra', 'cosine', 'intersection']

    # 遍历每个文件
    for file_name in file_list:
        # 读取原始数据
        df = pd.read_excel(file_name)

        # 创建一个空的DataFrame来存储结果
        result_df = pd.DataFrame(columns=['Missing Rate', 'Method'] + metrics)

        # 遍历每种缺失率和方法
        for missing_rate in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for method in ['HidLDL_best', 'IncomLDL', 'WInLDL', 'SA_IIS', 'LDL_LRR','PT_Bayes','LDL_DPA']:
                # 筛选出对应的数据
                subset = df[(df['Missing Rate'] == missing_rate) & (df['Method'] == method)]
                # 解析Scores列
                scores_list = subset['Scores'].apply(lambda x: dict([y.split() for y in x.split('\n')]))
                scores_df = pd.DataFrame(scores_list.tolist())

                # 计算每个指标的平均值和标准差
                metric_stats = {}
                for metric in metrics:
                    if metric in scores_df.columns:
                        values = scores_df[metric].astype(float)
                        mean = values.mean()
                        std = values.std()
                        mean_str = f'{mean:.4f}'.lstrip('0')
                        std_str = f'{std:.4f}'.lstrip('0')
                        metric_stats[metric] = f'{mean_str}±{std_str}'
                    else:
                        metric_stats[metric] = 'NaN'
                print(metric_stats)

                # 将结果添加到结果DataFrame中
                result_row = pd.DataFrame({
                    'Missing Rate': [missing_rate],
                    'Method': [method],
                    **{metric: [metric_stats[metric]] for metric in metrics}
                })
                result_df = pd.concat([result_df, result_row], ignore_index=True)

        # 保存结果到CSV文件
        base_name = os.path.basename(file_name)
        result_file_name = f'{os.path.splitext(base_name)[0]}_result.xlsx'
        result_file_path = os.path.join(output_dir, result_file_name)
        result_df.to_excel(result_file_path, index=False)
        print(f'Results for {file_name} saved to {result_file_path}')

directory = '../results/m'
output_dir = '../results/m/results'
# file = ['Yeast_alpha.xlsx','Yeast_cdc.xlsx','Yeast_cold.xlsx','Yeast_dtt.xlsx','Yeast_elu.xlsx',
#         'Yeast_spo.xlsx','SJAFFE.xlsx','Scene.xlsx','Movie.xlsx','SBU_3DFE.xlsx',
#         'emotion6.xlsx','RAF_ML.xlsx','SCUT_FBP.xlsx']
# file_list = [os.path.join(directory, file) for file in file]
file_list = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.xlsx')]
# 示例用法

process_files(file_list)