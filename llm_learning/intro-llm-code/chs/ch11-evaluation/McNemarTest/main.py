from mlxtend.evaluate import mcnemar
import numpy as np

tb_a = np.array([[9959, 11],
                 [1, 29]])

chi2, p = mcnemar(ary=tb_a, exact=True)

print('chi-squared-a:', chi2)
print('p-value-a:', p)


tb_b = np.array([[9945, 25],
                 [15, 15]])

chi2, p = mcnemar(ary=tb_b, exact=True)

print('chi-squared-b:', chi2)
print('p-value-b:', p)
