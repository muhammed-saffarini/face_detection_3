class DataCSA:
    def __init__(self):
        self.best = 0
        self.bestIndividual = []
        self.convergence = []
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.lb = 0
        self.ub = 0
        self.dim = 0
        self.popnum = 0
        self.maxiers = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.confusion_matrix = []
        self.tr_accuracy = 0
        self.tr_precision = 0
        self.tr_recall = 0
        self.tr_f1_score = 0
        self.tr_confusion_matrix = []