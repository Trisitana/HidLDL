import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_chebyshev_vs_missing_rate(file_path, output_dir):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 提取缺失率和方法
    missing_rates = df['Missing Rate'].unique()
    methods = df['Method'].unique()

    # 创建一个字典来存储每个方法的 Chebyshev 指标平均值
    chebyshev_means = {method: [] for method in methods}

    # 计算每个方法在不同缺失率下的 Chebyshev 指标平均值
    for method in methods:
        for rate in missing_rates:
            subset = df[(df['Missing Rate'] == rate) & (df['Method'] == method)]
            chebyshev_mean = subset['chebyshev'].str.split('±').str[0].astype(float).mean()
            chebyshev_means[method].append(chebyshev_mean)

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for method, means in chebyshev_means.items():
        plt.plot(missing_rates, means, marker='o', label=method)

    # 设置图表标题和标签
    # plt.title('Chebyshev Metric vs. Missing Rate')
    # plt.xlabel('Missing Rate')
    # plt.ylabel('Chebyshev')
    # plt.xticks(missing_rates)
    # plt.legend(title='Method')
    plt.title('Chebyshev Metric vs. Missing Rate', fontsize=16, fontweight='bold')
    plt.xlabel('Missing Rate', fontsize=14, fontweight='bold')
    plt.ylabel('Chebyshev', fontsize=14, fontweight='bold')  # 加粗并调整字体大小
    plt.xticks(missing_rates, fontsize=12)  # 设置 x 轴刻度字体大小
    plt.yticks(fontsize=12)  # 设置 y 轴刻度字体大小
    plt.legend(title='Method', fontsize=12, title_fontsize=12)

    # 优化图表样式
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图表到指定目录
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}_chebyshev_vs_missing_rate.png')
    plt.savefig(output_file)
    plt.close()  # 关闭当前图表以避免重叠

# 获取目录下的所有 Excel 文件
directory = '../results/m/results/'
output_dir = '../results/m/pic2/'
os.makedirs(output_dir, exist_ok=True)

file_list = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.xlsx')]
print(file_list)

# 遍历每个文件并生成折线图
for file_path in file_list:
    plot_chebyshev_vs_missing_rate(file_path, output_dir)