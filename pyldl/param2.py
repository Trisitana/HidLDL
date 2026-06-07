import numpy as np



def param_setting_alpha(model, X, y, y_missing, mask, alpha_values, max_iterations):
    best_score = [np.inf]
    best_alpha = None
    best_beta = None

    for alpha in alpha_values:
            # 训练模型
            model.fit(X, y_missing, mask=mask, alpha=alpha, max_iterations=max_iterations)
            # 评估模型
            score = model.score(X, y)
            # 检查 score 第一项是否是最小的
            if score[0] < best_score[0]:
                best_score = score
                best_alpha = alpha

    return best_alpha,  best_score


def param_setting_alpha_my(model, X, y, y_missing, mask, alpha_values, max_iterations):
    results = {}  # 创建一个字典来存储每次迭代的结果
    best_score = [np.inf]
    best_alpha = None

    for alpha in alpha_values:
        # 训练模型
        model.fit(X, y_missing, mask=mask, alpha=alpha, max_iterations=max_iterations)
        # 评估模型
        score = model.score(X, y)
        # 存储每次迭代的分数和对应的alpha值
        results[alpha] = score
        # 检查 score 第一项是否是最小的
        if score[0] < best_score[0]:
            best_score = score
            best_alpha = alpha

    # 保存所有结果
    all_scores = results
    # 返回最佳参数和分数，以及所有分数
    return best_alpha, best_score, all_scores
