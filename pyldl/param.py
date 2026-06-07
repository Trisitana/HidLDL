import numpy as np

def param_setting(model, X, y, y_missing, mask, alpha_values, beta_values, max_iterations):
    best_score = [np.inf]
    best_alpha = None
    best_beta = None
    model.fit(X, y_missing, mask=mask, alpha=0.1,beta=0.01, max_iterations=100)

    for alpha in alpha_values:
        for beta in beta_values:
            # 训练模型
            model.fit(X, y_missing, mask=mask, alpha=alpha, beta=beta, max_iterations=max_iterations)
            # 评估模型
            score = model.score(X, y)
            # 检查 score 第一项是否是最小的
            if score[0] < best_score[0]:
                best_score = score
                best_alpha = alpha
                best_beta = beta

    return best_alpha, best_beta, best_score

def param_setting_alpha(model, X, y, y_missing, mask, alpha_values, max_iterations):
    best_score = [np.inf]
    best_alpha = None


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

def param_setting_test(model, X_train, y_missing_train, mask_train, X_val, y_val,alpha_values, max_iterations):
    best_score = [np.inf]
    best_alpha = None

    for alpha in alpha_values:
            # 训练模型
            model.fit(X_train, y_missing_train, mask=mask_train, alpha=alpha, max_iterations=max_iterations)
            # 评估模型
            score = model.score(X_val, y_val)
            # 检查 score 第一项是否是最小的
            if score[0] < best_score[0]:
                best_score = score
                best_alpha = alpha

    return best_alpha,  best_score

def param_setting_test_new(model, X_train, y_missing_train, mask_train, y_train, X_val, y_val,alpha_values, max_iterations):
    best_score = [np.inf]
    best_alpha = None
    for alpha in alpha_values:
            # 训练模型
            model.fit(X_train, y_missing_train, mask=mask_train, alpha=alpha, max_iterations=max_iterations)
            # 评估模型
            score = model.score(X_val, y_val)
            # 检查 score 第一项是否是最小的
            if score[0] < best_score[0]:
                best_score = score
                best_alpha = alpha

    return best_alpha,  best_score


def param_setting_test_Beta(model, X_train, y_missing_train, mask_train, X_val, y_val, alpha_values,beta_values,max_iterations):
    best_score = [np.inf]
    best_alpha = None
    best_beta = None
    for alpha in alpha_values:
        for beta in beta_values:
           model.fit(X_train, y_missing_train, mask=mask_train, alpha=alpha, beta = beta,max_iterations=max_iterations)
        # 评估模型
           score = model.score(X_val, y_val)
        # 检查 score 第一项是否是最小的
           if score[0] < best_score[0]:
             best_score = score
             best_alpha = alpha
             best_beta = beta

    return best_alpha, best_beta, best_score
