import os
import csv
import math
import statistics
import numpy as np
from scipy.stats.stats import pearsonr

pathDLA = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\dla\\CSV"
pathFalls = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\falls\\CSV"
pathResultFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\single_features.csv"

frameDuration = 6 #frame duration in seconds

#DLA
files = os.listdir(pathDLA)
with open(pathResultFile, 'w', newline='') as resultfile:
    writer = csv.writer(resultfile, delimiter=';')
    writer.writerow(["FE7XW", "FE7XP", "FE7YW", "FE7YP", "FE7ZW", "FE7ZP", "FE8XW", "FE8XP", "FE8YW", "FE8YP", "FE8ZW", "FE8ZP", "FE9XW", "FE9XP", "FE9YW", "FE9YP", "FE9ZW", "FE9ZP",
                     "FE10XW", "FE10XP", "FE10YW", "FE10YP", "FE10ZW", "FE10ZP", "FE11XW", "FE11XP", "FE11YW", "FE11YP", "FE11ZW", "FE11ZP", "F12XW", "F12XP", "F12YW", "F12YP", "F12ZW", "F12ZP", "Y"])

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
            roww = list(zip(xw[1:], yw[1:], zw[1:]))
            # no possibility to count for first sample
            x = xw[0]
            y = yw[0]
            z = zw[0]
            timeStepw = float(frameDuration/len(xw))
            timeStepp = float(frameDuration/len(xp))
            for row in roww:
                jerkxw.append( abs((row[0] - x)/(timeStepw)) )
                jerkyw.append( abs((row[1] - y)/(timeStepw)) )
                jerkzw.append( abs((row[2] - z)/(timeStepw)) )
                x = row[0]
                y = row[1]
                z = row[2]
            rowp = list(zip(xp[1:], yp[1:], zp[1:]))
            x = xp[0]
            y = yp[0]
            z = zp[0]
            for row in rowp:
                jerkxp.append( abs((row[0] - x)/(timeStepp)) )
                jerkyp.append( abs((row[1] - y)/(timeStepp)) )
                jerkzp.append( abs((row[2] - z)/(timeStepp)) )
                x = row[0]
                y = row[1]
                z = row[2]
            writer.writerow([statistics.mean(xw), statistics.mean(xp), statistics.mean(yw), statistics.mean(yp), statistics.mean(zw), statistics.mean(zp),
                             statistics.variance(xw), statistics.variance(xp), statistics.variance(yw), statistics.variance(yp), statistics.variance(zw), statistics.variance(zp),
                             statistics.stdev(xw), statistics.stdev(xp), statistics.stdev(yw), statistics.stdev(yp), statistics.stdev(zw), statistics.stdev(zp),
                             sum(np.absolute(xw)), sum(np.absolute(xp)), sum(np.absolute(yw)), sum(np.absolute(yp)), sum(np.absolute(zw)), sum(np.absolute(zp)),
                             (max(xw) - min(xw)), (max(xp) - min(xp)), (max(yw) - min(yw)), (max(yp) - min(yp)), (max(zw) - min(zw)), (max(zp) - min(zp)),
                             max(jerkxw), max(jerkxp), max(jerkyw), max(jerkyp), max(jerkzw), max(jerkzp), "0"])
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
            roww = list(zip(xw[1:], yw[1:], zw[1:]))
            # no possibility to count for first sample
            x = xw[0]
            y = yw[0]
            z = zw[0]
            timeStepw = float(frameDuration/len(xw))
            timeStepp = float(frameDuration/len(xp))
            for row in roww:
                jerkxw.append( abs((row[0] - x)/(timeStepw)) )
                jerkyw.append( abs((row[1] - y)/(timeStepw)) )
                jerkzw.append( abs((row[2] - z)/(timeStepw)) )
                x = row[0]
                y = row[1]
                z = row[2]
            rowp = list(zip(xp[1:], yp[1:], zp[1:]))
            x = xp[0]
            y = yp[0]
            z = zp[0]
            for row in rowp:
                jerkxp.append( abs((row[0] - x)/(timeStepp)) )
                jerkyp.append( abs((row[1] - y)/(timeStepp)) )
                jerkzp.append( abs((row[2] - z)/(timeStepp)) )
                x = row[0]
                y = row[1]
                z = row[2]
            writer.writerow([statistics.mean(xw), statistics.mean(xp), statistics.mean(yw), statistics.mean(yp), statistics.mean(zw), statistics.mean(zp),
                             statistics.variance(xw), statistics.variance(xp), statistics.variance(yw), statistics.variance(yp), statistics.variance(zw), statistics.variance(zp),
                             statistics.stdev(xw), statistics.stdev(xp), statistics.stdev(yw), statistics.stdev(yp), statistics.stdev(zw), statistics.stdev(zp),
                             sum(np.absolute(xw)), sum(np.absolute(xp)), sum(np.absolute(yw)), sum(np.absolute(yp)), sum(np.absolute(zw)), sum(np.absolute(zp)),
                             (max(xw) - min(xw)), (max(xp) - min(xp)), (max(yw) - min(yw)), (max(yp) - min(yp)), (max(zw) - min(zw)), (max(zp) - min(zp)),
                             max(jerkxw), max(jerkxp), max(jerkyw), max(jerkyp), max(jerkzw), max(jerkzp), "1"])
    resultfile.close();