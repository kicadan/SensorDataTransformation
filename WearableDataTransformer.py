import os
import csv
import math
import statistics
import numpy as np
from scipy.stats.stats import pearsonr

pathDLA = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\dla\\CSV"
pathFalls = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\falls\\CSV"
pathResultFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\wearable_all_features.csv"

frameDuration = 6 #frame duration in seconds

#DLA
files = os.listdir(pathDLA)
with open(pathResultFile, 'w', newline='') as resultfile:
    writer = csv.writer(resultfile, delimiter=';')
    writer.writerow(["FE1W", "FE2W", "FE3W", "FE4W", "FE7XW", "FE7YW", "FE7ZW", "FE8XW", "FE8YW", "FE8ZW", "FE9XW", "FE9YW", "FE9ZW",
                     "FE10XW", "FE10YW", "FE10ZW", "FE11XW", "FE11YW", "FE11ZW", "F12XW", "F12YW", "F12ZW", "F5W", "F6W", "Y"])

    for filename in files:
        xw = []
        yw = []
        zw = []
        xp = []
        yp = []
        zp = []
        magw = []
        magp = []
        #arrays for jerk
        jerkxw = []
        jerkyw = []
        jerkzw = []
        jerkxp = []
        jerkyp = []
        jerkzp = []
        with open(pathDLA + "\\" + filename, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            next(reader, None)
            for row in reader:
                if row[0] != '':
                    xw.append(int(row[0]))
                    yw.append(int(row[1]))
                    zw.append(int(row[2]))
                    magw.append(np.linalg.norm([int(row[0]), int(row[1]), int(row[2])]))
                if row[3] != '':
                    xp.append(float(row[3]))
                    yp.append(float(row[4]))
                    zp.append(float(row[5]))
                    magp.append(np.linalg.norm([float(row[3]), float(row[4]), float(row[5])]))
            roww = iter([xw, yw, zw])
            # no possibility to count for first sample
            x = xw[0]
            y = yw[0]
            z = zw[0]
            next(roww)
            timeStepw = float(frameDuration/len(xw))
            timeStepp = float(frameDuration/len(xp))
            for row in roww:
                jerkxw.append( abs((row[0] - x)/(timeStepw)) )
                jerkyw.append( abs((row[1] - y)/(timeStepw)) )
                jerkzw.append( abs((row[2] - z)/(timeStepw)) )
                x = row[0]
                y = row[1]
                z = row[2]
            rowp = iter([xp, yp, zp])
            x = xp[0]
            y = yp[0]
            z = zp[0]
            next(rowp)
            for row in rowp:
                jerkxp.append( abs((row[0] - x)/(timeStepp)) )
                jerkyp.append( abs((row[1] - y)/(timeStepp)) )
                jerkzp.append( abs((row[2] - z)/(timeStepp)) )
                x = row[0]
                y = row[1]
                z = row[2]
            writer.writerow([max(magw), pearsonr(xw, yw)[0], pearsonr(xw, zw)[0], pearsonr(yw, zw)[0],
                             statistics.mean(xw), statistics.mean(yw), statistics.mean(zw),
                             statistics.variance(xw), statistics.variance(yw), statistics.variance(zw),
                             statistics.stdev(xw), statistics.stdev(yw), statistics.stdev(zw),
                             sum(np.absolute(xw)), sum(np.absolute(yw)), sum(np.absolute(zw)),
                             (max(xw) - min(xw)), (max(yw) - min(yw)), (max(zw) - min(zw)),
                             max(jerkxw), max(jerkyw), max(jerkzw),
                             np.linalg.norm([statistics.stdev(xw), statistics.stdev(zw)]),
                             np.linalg.norm([statistics.stdev(xw), statistics.stdev(yw), statistics.stdev(zw)]), "0"])
    resultfile.close();

#Falls
files = os.listdir(pathFalls)
with open(pathResultFile, 'a+', newline='') as resultfile:
    writer = csv.writer(resultfile, delimiter=';')

    for filename in files:
        xw = []
        yw = []
        zw = []
        xp = []
        yp = []
        zp = []
        magw = []
        magp = []
        #arrays for jerk
        jerkxw = []
        jerkyw = []
        jerkzw = []
        jerkxp = []
        jerkyp = []
        jerkzp = []
        with open(pathFalls + "\\" + filename, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            next(reader, None)
            for row in reader:
                if row[0] != '':
                    xw.append(int(row[0]))
                    yw.append(int(row[1]))
                    zw.append(int(row[2]))
                    magw.append(np.linalg.norm([int(row[0]), int(row[1]), int(row[2])]))
                if row[3] != '':
                    xp.append(float(row[3]))
                    yp.append(float(row[4]))
                    zp.append(float(row[5]))
                    magp.append(np.linalg.norm([float(row[3]), float(row[4]), float(row[5])]))
            roww = iter([xw, yw, zw])
            # no possibility to count for first sample
            x = xw[0]
            y = yw[0]
            z = zw[0]
            next(roww)
            timeStepw = float(frameDuration/len(xw))
            timeStepp = float(frameDuration/len(xp))
            for row in roww:
                jerkxw.append( abs((row[0] - x)/(timeStepw)) )
                jerkyw.append( abs((row[1] - y)/(timeStepw)) )
                jerkzw.append( abs((row[2] - z)/(timeStepw)) )
                x = row[0]
                y = row[1]
                z = row[2]
            rowp = iter([xp, yp, zp])
            x = xp[0]
            y = yp[0]
            z = zp[0]
            next(rowp)
            for row in rowp:
                jerkxp.append( abs((row[0] - x)/(timeStepp)) )
                jerkyp.append( abs((row[1] - y)/(timeStepp)) )
                jerkzp.append( abs((row[2] - z)/(timeStepp)) )
                x = row[0]
                y = row[1]
                z = row[2]
            writer.writerow([max(magw), pearsonr(xw, yw)[0], pearsonr(xw, zw)[0], pearsonr(yw, zw)[0],
                             statistics.mean(xw), statistics.mean(yw), statistics.mean(zw),
                             statistics.variance(xw), statistics.variance(yw), statistics.variance(zw),
                             statistics.stdev(xw), statistics.stdev(yw), statistics.stdev(zw),
                             sum(np.absolute(xw)), sum(np.absolute(yw)), sum(np.absolute(zw)),
                             (max(xw) - min(xw)), (max(yw) - min(yw)), (max(zw) - min(zw)),
                             max(jerkxw), max(jerkyw), max(jerkzw),
                             np.linalg.norm([statistics.stdev(xw), statistics.stdev(zw)]),
                             np.linalg.norm([statistics.stdev(xw), statistics.stdev(yw), statistics.stdev(zw)]), "1"])
    resultfile.close();