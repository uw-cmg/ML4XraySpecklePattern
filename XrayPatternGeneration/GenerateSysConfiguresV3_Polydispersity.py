"""

Generate ALL Needed Configures

creator: Mingren Shen
created: 2020-08-17
modifiy: 2020-12-16

ChangeLog: 
(1)Auto create Folders
(2)Adding function to genreate polydispersity beads sizes

Output: 
"""
import random
import time
import os
import math
import numpy as np

Lx=2000 #system Lx
Ly=2000 #system Ly

RadiusBeads=90 #Beads Radius

NumBeads=20 #Total number of Beads
# create Beads Number folder
# create results file when needed
if not os.path.exists("Beads_"+str(NumBeads)):
    os.makedirs("Beads_"+str(NumBeads))

NumConfigures=1000 #Total Number of configures of Beads
CutOffDistance=2 #Cut-off distance

# Gaussian Distribution of polydispersity
mu, sigma = 3, 1 # mean and standard deviation
s = np.random.normal(mu, sigma, 3 * NumBeads)
coefficients = [0] * NumBeads
radius = [0] * NumBeads
radius_square = [0] * NumBeads
radius_quadra = [0] * NumBeads
# generate radius of each bead first
# (1) get total scatter areas of two beads
Total_area = RadiusBeads * RadiusBeads * math.pi * 2
# (2) sample coefficients from Gaussian
i = 0
j = 0
while i < len(coefficients) and j < len(s):
    if s[j] > 0:
        coefficients[i] = s[j]
        i += 1
    j += 1
# Check that all coefficients are postive
for coef in coefficients:
    assert coef > 0

# (3) Convert Coefficients to Real Radius of Beads
convertFactor = Total_area / (1.0 * sum(coefficients))
for i in range(len(radius)):
    radius[i] = np.sqrt(convertFactor * coefficients[i])
    radius_square[i] = radius[i] * radius[i]
    radius_quadra[i] = radius_square[i] * radius_square[i]
# (4) Report polydispersity index (PI)
#     PDI = M_w / M_n
# Mw  is the weight average molecular weight ,
# Mn is the number average molecular weight. 
# Mn is more sensitive to molecules of low molecular mass, while Mw is more sensitive to molecules of high molecular mass. 
# Two assumptions:
# same density, Mass ~ Area ~ r_i^2
# number of each mass is 1, coefficients are not likely to the same
#      PDI = [sum(r_i^2) / sum(r_i)] / [sum(r_i) / totalBeadNum ]
#          = sum(r_i^2) * totalBeadNum / (sum(r_i)**2)
with open("PDI.csv", "a") as PDIcsv:
    PDIcsv.write("%d, %f\n"%(NumBeads,sum(radius_quadra) * NumBeads / (sum(radius_square)**2) ))

# Loop through all configurations

for i in range(NumConfigures):
    with open("Beads_"+str(NumBeads)+"/BeadsNum_"+str(NumBeads)+"Config_"+str(i)+".txt","w") as ConfigureResults:
        # Loop through all beads
        # (1) Generate Rand Positions
        # (2) Check with any overlappping
        random.seed(time.time())
        rx = [0] * NumBeads
        ry = [0] * NumBeads

        for j in range(NumBeads):
            flagOverLapping = True
            while flagOverLapping:
                tmpX = random.uniform(RadiusBeads, Lx-RadiusBeads)
                tmpY = random.uniform(RadiusBeads, Ly-RadiusBeads)
                # print("tmpX %f, tmpY %f"%(tmpX, tmpY))
                # check with already generated Positions
                for k in range(j):
                    dist = np.sqrt((tmpX - rx[k])**2 + (tmpY-ry[k])**2)
                    if dist < ( radius[j] + radius[k] + CutOffDistance):
                        print("Conficits!!!")
                        flagOverLapping = True
                        break
                    flagOverLapping = False
                if j == 0: flagOverLapping = False
            rx[j] = tmpX
            ry[j] = tmpY
        # Finish Current Configue
        # Save ConfigureResults
        for ind in range(NumBeads):
            ConfigureResults.write(" %6.3f, %6.3f, %6.3f\n"%(rx[ind], ry[ind], radius[ind]))