import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import FuncFormatter


def plot__vs_alpha(file_path, output_dir):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 找到缺失率为50%的情况
    df_50 = df[df['Missing Rate'] == 0.5]

    # 找到HidLDL和IncomLDL这两种方法的数据
    df_HidLDL = df_50[df_50['Method'] == 'HidLDL']
    df_IncomLDL = df_50[df_50['Method'] == 'IncomLDL']

    # 计算HidLDL在不同Alpha下的cosine指标平均值
    alphas = [2 ** i for i in range(-6, 7)]  # Alpha从2的-6次方到2的6次方
    cosine_means_HidLDL = []

    for alpha in alphas:
        subset = df_HidLDL[df_HidLDL['Alpha'] == alpha]
        cosine_values = subset['Scores'].apply(
            lambda x: float([y.split()[1] for y in x.split('\n') if y.startswith('cosine')][0]))
        cosine_mean = cosine_values.mean()
        cosine_means_HidLDL.append(cosine_mean)

    # 计算IncomLDL的cosine指标平均值
    cosine_values_IncomLDL = df_IncomLDL['Scores'].apply(
        lambda x: float([y.split()[1] for y in x.split('\n') if 'cosine' in y][0]))
    cosine_mean_IncomLDL = cosine_values_IncomLDL.mean()

    # 绘制折线图
    plt.figure(figsize=(5.8, 3.4), dpi=300)
    plt.plot(alphas, cosine_means_HidLDL, marker='o', linestyle='--', color='darkblue', fillstyle='none', label='Ours')
    plt.axhline(y=cosine_mean_IncomLDL, color='purple', linestyle='-', label='InLDL-a')

    # 设置图表标题和标签
    plt.xscale('log', base=2)  # 将x轴设置为对数刻度，底数为2

    plt.xticks(alphas, [f'$2^{{{int(np.log2(a))}}}$' for a in alphas], rotation=45)
    plt.gca().tick_params(axis='x', labelsize=16)
    # 自定义x轴标签格式


    plt.legend(fontsize=18)

    # 优化图表样式
    plt.grid(True, linestyle='-', alpha=0.1)
    plt.tight_layout()

    # 保存图表到指定目录
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}_cosine_vs_alpha.png')
    plt.savefig(output_file)
    plt.close()  # 关闭当前图表以避免重叠


# 获取目录下的所有 Excel 文件
directory = '../results/m/'
output_dir = '../results/m/hyperSetting2/'
os.makedirs(output_dir, exist_ok=True)

file_list = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.xlsx')]
# file_list = [os.path.join(directory, 'Yeast_elu.xlsx')]

# 遍历每个文件并生成折线图
for file_path in file_list:
    plot__vs_alpha(file_path, output_dir)