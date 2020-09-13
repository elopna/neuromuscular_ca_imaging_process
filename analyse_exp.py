import get_transient_v20 as ta
import matplotlib.pyplot as plt

path = 'experiments/failed exps/190520T2pro/'
# path = 'experiments/190520T1pro/'
mask = ta.transient_analysis(path, fps=500)

# print(mask.shape)
# print(type(mask))
# plt.imshow(mask)
# plt.show()
# print(mask.shape)
# print(type(mask))



!!! check by 2 pics translation of mask