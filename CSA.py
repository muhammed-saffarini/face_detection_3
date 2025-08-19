# -*- coding: utf-8 -*-
"""
Created on Thirsday September 29 2022

@author: Thaer Thaher
"""

import random
import numpy

from DataCSA import DataCSA
import time
import pandas as pd


def check_feasibility(pos, lb, ub):
    if pos[0] < lb[0] or pos[0] > ub[0] or pos[2] < lb[2] or pos[2] > ub[2]:
        return False
    return True


def CSA(objf, lb, ub, dim, PopSize, tmax):
    # CSA Internal parameters

    AP = 0.1  # Awareness probability
    fl = 2  # Flight length (fl)

    s = DataCSA()
    # if not isinstance(lb, list):
    #     lb = [lb] * dim
    # if not isinstance(ub, list):
    #     ub = [ub] * dim

    ######################## Initializations
    ub = numpy.array(ub)  # Convert to numpy array
    lb = numpy.array(lb)
    # gBest = numpy.zeros(dim)  # Solutin of the problem

    gBestScore = float("inf")  # Best fitness value found so far
    t_accuracy = 0
    t_precision = 0
    t_recall = 0
    t_f1 = 0
    tr_accuracy = 0
    tr_precision = 0
    tr_recall = 0
    tr_f1 = 0
    tr_loss = 0

    ft = numpy.zeros((PopSize, 1))  # fitness values
    test_accuracy = numpy.zeros((PopSize, 1))
    test_precision = numpy.zeros((PopSize, 1))
    test_recall = numpy.zeros((PopSize, 1))
    test_f1 = numpy.zeros((PopSize, 1))
    train_accuracy = numpy.zeros((PopSize, 1))
    train_precision = numpy.zeros((PopSize, 1))
    train_recall = numpy.zeros((PopSize, 1))
    train_f1 = numpy.zeros((PopSize, 1))
    train_loss = numpy.zeros((PopSize, 1))
    ###################### Initialize position and memory of crows

    X = numpy.zeros((PopSize, dim))
    xnew = numpy.zeros((PopSize, dim))
    # for i in range(dim):
    #    X[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]
    for i in range(PopSize):
        X[i, :] = numpy.random.uniform(0, 1, dim) * (ub - lb) + lb

    xn = X.copy()  # the position of the crow
    mem = X.copy()  # Memory initialization

    for i in range(0, PopSize):
        # Calculate objective function for each crow
        if xn[i, 0] != 0:
            ft[i], test_accuracy[i], test_precision[i], test_recall[i], test_f1[i], train_accuracy[i], train_precision[i], train_recall[i], train_f1[i], train_loss[i] = objf(xn[i, :])
            if gBestScore > ft[i]:
                gBestScore = ft[i]
                gBest = xn[i, :].copy()


    fit_mem = ft.copy()  # % Fitness of memory positions
    convergence_curve = numpy.zeros(tmax)

    ############################################
    print('CSA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for t in range(0, tmax):

        # print(t)# no of iterations
        # No clipping

        for i in range(0, PopSize):
            r = random.random()
            num = random.randint(0, PopSize - 1)  # Generation of random candidate crows for following (chasing)
            if r >= AP:
                xnew[i, :] = xn[i, :] + fl * r * (
                        mem[num, :] - xn[i, :])  # Generation of a new position for crow i (state 1)
            else:  # Generation of a new position for crow i (state 2)
                xnew[i, :] = numpy.random.uniform(0, 1, dim) * (ub[1] - lb[1]) + lb[1]

        # Evaluate new positions
        for i in range(0, PopSize):
            if check_feasibility(xnew[i, :], lb, ub) and xnew[i, 0] != 0:
                ft[i], test_accuracy[i], test_precision[i], test_recall[i], test_f1[i], train_accuracy[i], \
                train_precision[i], train_recall[i], train_f1[i], train_loss[i] = objf(xnew[i, :])

        # Check the feasibility of new positions
        for i in range(0, PopSize):
            if check_feasibility(xnew[i, :], lb, ub):
                xn[i, :] = xnew[i, :].copy()  # update position
                if ft[i] < fit_mem[i]:
                    mem[i, :] = xnew[i, :].copy()
                    fit_mem[i] = ft[i]
        print(fit_mem)
        gBestScore = min(fit_mem)
        min_index = fit_mem.argmin()
        gBest = mem[min_index, :].copy()
        t_accuracy = test_accuracy[min_index]
        t_precision = test_precision[min_index]
        t_recall = test_recall[min_index]
        t_f1 = test_f1[min_index]
        tr_accuracy = train_accuracy[min_index]
        tr_precision = train_precision[min_index]
        tr_recall = train_recall[min_index]
        tr_f1 = train_f1[min_index]
        tr_loss = train_loss[min_index]
        convergence_curve[t] = gBestScore

        if t % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(t + 1)
                    + " the best fitness is "
                    + str(gBestScore)
                ]
            )
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "CSA"
    s.objfname = "ELM"  # objf.__name__
    s.best = gBestScore
    s.bestIndividual = gBest
    s.accuracy = t_accuracy
    s.precision = t_precision
    s.recall = t_recall
    s.f1_score = t_f1
    # s.confusion_matrix = confusion_matrix
    s.tr_accuracy = tr_accuracy
    s.tr_precision = tr_precision
    s.tr_recall = tr_recall
    s.tr_f1_score = tr_f1

    return s
