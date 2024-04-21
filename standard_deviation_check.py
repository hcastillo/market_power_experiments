#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt


male1 = [525.8,605.7,843.3,1195.5,1945.6,2135.6,2308.7,2950.0]
female1=[727.7, 1086.5, 1091.0, 1361.3, 1490.5, 1956.1]

male = pd.DataFrame(male1)
female = pd.DataFrame(female1)

print(male.std(),female.std())

plt.clf()
plt.errorbar([1,2], [male[0].mean(),female[0].mean()],
             [male[0].std()/2,female[0].std()/2], linestyle='None', marker='^')
#plt.xticks([1,2], ['male','female'])
#plt.savefig('standard_deviation_check.png')


#plt.clf()
#plt.errorbar([1,2], [4,5],
#             [1,3], linestyle='None', marker='^')
plt.show()