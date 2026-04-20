import numpy as np
# Normalized cumulants for the first design point

arr = np.load("design_0000/trento_events.npy")

# centrality
mult = arr[:,3]
rank = np.argsort(np.argsort(-mult))
cent = rank / len(mult) * 100

mask = (cent < 1)

eps3 = arr[mask, 7]

# is better to use np.average here?!
m2 = np.mean(eps3**2)
m4 = np.mean(eps3**4)

nc_2_4 = (m4 - 2*m2**2)/(m2**2)

# generalize this defintion when nc_2_4 is zero or negative!
ratio_4_2 = -(nc_2_4)**(1/4)


print(nc_2_4, ratio_4_2)
