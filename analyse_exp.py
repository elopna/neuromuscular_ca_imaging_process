import get_transient_v20 as ta
import matplotlib.pyplot as plt

path_control = 'experiments/190520T1Con/'
path_d1 = 'experiments/190520T1pro/'
# path = 'experiments/190520T1pro/'
mask = ta.transient_analysis(path_control, fps=500)
ta.transient_analysis(path_d1, fps=500, mask=mask, control_img='experiments/190520T1Con/190520T100000000.jpg')

# print(mask.shape)
# print(type(mask))
# plt.imshow(mask)
# plt.show()
# print(mask.shape)
# print(type(mask))



# !!! check by 2 pics translation of mask