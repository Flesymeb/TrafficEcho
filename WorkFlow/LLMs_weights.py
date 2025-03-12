import pandas as pd
import numpy as np

xls = pd.ExcelFile('weights_planner.xlsx')
sheets = xls.sheet_names
matrices = []
for sheet in sheets:
    data = pd.read_excel(xls, sheet_name=sheet, header=None, skiprows=1, usecols=list(range(1, 10))).values.astype(float)
    matrices.append(data)

num_models = len(matrices)
n = matrices[0].shape[0]
M = np.zeros((n, n))
#print(num_models)
#print(n)

# 合成判断矩阵
for i in range(n):
    for j in range(n):
        values = [m[i][j] for m in matrices]

        geom = np.prod(values) ** (1/num_models)
        harm = np.sum(values) / num_models
        M[i][j] = (geom + harm) / 2

        #M[i][j] = np.sum(values) / num_models
        #M[i][j] = np.prod(values) ** (1/num_models)

# 最大特征值和特征向量
norm_M = M / M.sum(axis=1, keepdims=True)
lambdas, vectors = np.linalg.eig(M)
max_idx = np.argmax(lambdas)
max_lambda = lambdas[max_idx]
print(max_lambda)
weights = vectors[:, max_idx].real
weights = weights / weights.sum()

# 一致性检验
CI = (max_lambda - n) / (n - 1)
print(CI)
RI = 1.41 # n=9
CR = CI / RI

# 输出结果
#print("判断矩阵:\n", M)
print("指标权重:")
for i, weight in enumerate(weights):
    print(f"{['安全性', '运行效率', '可靠性', '舒适性', '便捷性', '信息化与智能化', '经济性', '环境友好性', '社会可持续性'][i]}: {weight:.4f}")
print("\n一致性比例 CR:", CR)
if 0 < CR < 0.1:
    print("一致性检验通过！")
else:
    print("一致性检验未通过，请检查判断矩阵！")

