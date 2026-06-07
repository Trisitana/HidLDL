import pandas as pd
import numpy as np
from pyldl.param import param_setting_alpha
from pyldl.algorithms import IncomLDL, WInLDL, HidLDL, LDL_DPA,LDL_LRR,SA_IIS, PT_Bayes
from pyldl.utils import load_dataset, random_missing_real


X, y = load_dataset('SJAFFE')

y_missing, mask = random_missing_real(y, missing_rate=.5)

metrics=["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]


incomldl = IncomLDL()
best_alpha, best_score = param_setting_alpha(incomldl,X,y,y_missing,mask,np.geomspace(2**-10, 2**10, num=21),100)
print(best_alpha)
print(pd.Series(best_score, index=metrics))

winldl = WInLDL()
winldl.fit(X, y_missing, mask=mask)
print(pd.Series(winldl.score(X, y), index=metrics))


HidLDL = HidLDL()
best_alpha, best_score = param_setting_alpha(HidLDL,X,y,y_missing,mask,np.geomspace(2**-10, 2**10, num=21),100)
print(best_alpha)
print(pd.Series(best_score, index=metrics))

ldl_dpa = LDL_DPA()
ldl_dpa.fit(X, y_missing)
print(pd.Series(ldl_dpa.score(X, y), index=metrics))

ldl_LRR = LDL_LRR()
ldl_LRR.fit(X, y_missing,beta=1e-2)
print(pd.Series(ldl_LRR.score(X, y), index=metrics))


sa_IIS = SA_IIS()
sa_IIS.fit(X, y_missing)
print(pd.Series(sa_IIS.score(X, y), index=metrics))


pt_bayes = PT_Bayes()
pt_bayes.fit(X, y_missing)
print(pd.Series(pt_bayes.score(X, y), index=metrics))