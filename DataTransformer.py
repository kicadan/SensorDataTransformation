import os
import csv
import math
import statistics
import numpy as np
from scipy.stats.stats import pearsonr

pathDLA = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\dla\\CSV"
pathFalls = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\falls\\CSV"
pathResultFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\features.csv"

frameDuration = 6 #frame duration in seconds

#DLA
files = os.listdir(pathDLA)
with open(pathResultFile, 'w', newline='') as resultfile:
    writer = csv.writer(resultfile, delimiter=';')
    writer.writerow(["FE1W", "FE1P", "FE3W", "FE3P", "FE4W", "FE4P", "FE5W", "FE5P", "FE9XW", "FE9XP", "FE9YW", "FE9YP", "FE9ZW", "FE9ZP", "FE10XW", "FE10XP", "FE10YW", "FE10YP", "FE10ZW", "FE10ZP", "FE11XW", "FE11XP", "FE11YW", "FE11YP", "FE11ZW", "FE11ZP",
                     "FE12XW", "FE12XP", "FE12YW", "FE12YP", "FE12ZW", "FE12ZP", "FE13XW", "FE13XP", "FE13YW", "FE13YP", "FE13ZW", "FE13ZP", "F14XW", "F14XP", "F14YW", "F14YP", "F14ZW", "F14ZP", "F15W", "F15P", "F16W", "F16P", "Y"])

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
            writer.writerow([max(magw), max(magp), pearsonr(xw, yw)[0], pearsonr(xp, yp)[0], pearsonr(xw, zw)[0], pearsonr(xp, zp)[0], pearsonr(yw, zw)[0], pearsonr(yp, zp)[0],
                             statistics.mean(xw), statistics.mean(xp), statistics.mean(yw), statistics.mean(yp), statistics.mean(zw), statistics.mean(zp),
                             statistics.variance(xw), statistics.variance(xp), statistics.variance(yw), statistics.variance(yp), statistics.variance(zw), statistics.variance(zp),
                             statistics.stdev(xw), statistics.stdev(xp), statistics.stdev(yw), statistics.stdev(yp), statistics.stdev(zw), statistics.stdev(zp),
                             sum(np.absolute(xw)), sum(np.absolute(xp)), sum(np.absolute(yw)), sum(np.absolute(yp)), sum(np.absolute(zw)), sum(np.absolute(zp)),
                             (max(xw) - min(xw)), (max(xp) - min(xp)), (max(yw) - min(yw)), (max(yp) - min(yp)), (max(zw) - min(zw)), (max(zp) - min(zp)),
                             max(jerkxw), max(jerkxp), max(jerkyw), max(jerkyp), max(jerkzw), max(jerkzp),
                             np.linalg.norm([statistics.stdev(xw), statistics.stdev(zw)]), np.linalg.norm([statistics.stdev(xp), statistics.stdev(zp)]),
                             np.linalg.norm([statistics.stdev(xw), statistics.stdev(yw), statistics.stdev(zw)]), np.linalg.norm([statistics.stdev(xp), statistics.stdev(yp), statistics.stdev(zp)]), "0"])
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
            writer.writerow([max(magw), max(magp), pearsonr(xw, yw)[0], pearsonr(xp, yp)[0], pearsonr(xw, zw)[0], pearsonr(xp, zp)[0], pearsonr(yw, zw)[0], pearsonr(yp, zp)[0],
                             statistics.mean(xw), statistics.mean(xp), statistics.mean(yw), statistics.mean(yp), statistics.mean(zw), statistics.mean(zp),
                             statistics.variance(xw), statistics.variance(xp), statistics.variance(yw), statistics.variance(yp), statistics.variance(zw), statistics.variance(zp),
                             statistics.stdev(xw), statistics.stdev(xp), statistics.stdev(yw), statistics.stdev(yp), statistics.stdev(zw), statistics.stdev(zp),
                             sum(np.absolute(xw)), sum(np.absolute(xp)), sum(np.absolute(yw)), sum(np.absolute(yp)), sum(np.absolute(zw)), sum(np.absolute(zp)),
                             (max(xw) - min(xw)), (max(xp) - min(xp)), (max(yw) - min(yw)), (max(yp) - min(yp)), (max(zw) - min(zw)), (max(zp) - min(zp)),
                             max(jerkxw), max(jerkxp), max(jerkyw), max(jerkyp), max(jerkzw), max(jerkzp),
                             np.linalg.norm([statistics.stdev(xw), statistics.stdev(zw)]), np.linalg.norm([statistics.stdev(xp), statistics.stdev(zp)]),
                             np.linalg.norm([statistics.stdev(xw), statistics.stdev(yw), statistics.stdev(zw)]), np.linalg.norm([statistics.stdev(xp), statistics.stdev(yp), statistics.stdev(zp)]), "1"])
    resultfile.close();