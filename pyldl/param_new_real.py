import numpy as np

def param_setting_alpha(model, X_train, y_missing_train, mask_train, y_train, X_val, y_val,alpha_values, max_iterations):
    best_score = [np.inf]
    best_alpha = None
    best_result = None
    for alpha in alpha_values:
            # 训练模型
        try:
            model.fit(X_train, y_missing_train, mask=mask_train, alpha=alpha, max_iterations=max_iterations)
            # 评估模型
            score = model.score(X_train, y_train)
            # 检查 score 第一项是否是最小的
            if score[0] < best_score[0]:
                best_score = score
                best_result = model.score1(X_val, y_val)
                best_alpha = alpha
        except Exception as e:
            continue

    return best_alpha,  best_score,best_result


def param_setting_alpha_my(model, X_train, y_missing_train, mask_train, y_train, X_val, y_val,alpha_values, max_iterations):
    results = {}  # 创建一个字典来存储每次迭代的结果
    best_score = [np.inf]
    best_alpha = None
    best_result = None

    for alpha in alpha_values:
        # 训练模型
     try:
         model.fit(X_train, y_missing_train, mask=mask_train, alpha=alpha, max_iterations=max_iterations)
        # 评估模型
         score = model.score(X_train, y_train)
        # 存储每次迭代的分数和对应的alpha值
        #  results[alpha] = score
        # 检查 score 第一项是否是最小的
         if score[0] < best_score[0]:
            best_score = score
            best_result = model.score1(X_val, y_val)
            best_alpha = alpha
     except Exception as e:
         print(e)
         continue
    # 保存所有结果
    all_scores = results
    # 返回最佳参数和分数，以及所有分数
    return best_alpha, best_score,best_result
