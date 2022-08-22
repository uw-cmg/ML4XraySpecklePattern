"""

Generate ALL Needed Configures

creator: Mingren Shen
created: 2020-08-17
modifiy: 2020-08-22

"""
import random
import time
import numpy as np

Lx=2000 #system Lx
Ly=2000 #system Ly

RadiusBeads=90 #Beads Radius

NumBeads=19 #Total number of Beads

NumConfigures=1000 #Total Number of configures of Beads
CutOffDistance=2 #Cut-off distance

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
                    if dist < (2*RadiusBeads + CutOffDistance):
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
            ConfigureResults.write(" %6.3f, %6.3f\n"%(rx[ind], ry[ind]))