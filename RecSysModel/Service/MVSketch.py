import numpy as np
import pandas as pd
SEED = 2019
np.random.seed(SEED)
class HHtracer():
    def __init__(self, sketch_width, sketch_deep, phi, S):
        self.sketch_width = sketch_width
        self.sketch_deep  = sketch_deep 
        self.phi = phi
        self.S   = S
        self.sketch = [[(0,0,0) for x in range(sketch_width)] for y in range(sketch_deep)]
        return
    
    def processStream_HH(self, dataset):
        for record in dataset:
            item = (record[1], 1)
            self.update(item)
        return
    
    def update(self, item):
        x  = item[0]
        vx = item[1]
        for i in range(self.sketch_deep):
            np.random.seed(i + x)
            j = np.random.choice(self.sketch_width)
            V = self.sketch[i][j][0] + vx
            K = self.sketch[i][j][1]
            C = self.sketch[i][j][2]
            if K == x:
                C += vx
            else:
                C -= vx
                if C < 0:
                    K = x
                    C = -C
            self.sketch[i][j] = (V, K, C)
        return
    
    def queryU(self, x):
        res_list = list()
        for i in range(self.sketch_deep):
            np.random.seed(i + x)
            j = np.random.choice(self.sketch_width)
            V = self.sketch[i][j][0]
            K = self.sketch[i][j][1]
            C = self.sketch[i][j][2] 
            if K == x:
                S = (V + C) / 2
            else:
                S = (V - C) / 2
            res_list.append(S)
        return min(res_list)   
    
    def hitter(self):
        print("heavy hitter threshold: ", self.phi * self.S)
        hh = set()
        for i in range(self.sketch_deep):
            for j in range(self.sketch_width):
                if self.sketch[i][j][0] >= self.phi * self.S:
                    x = self.sketch[i][j][1]
                    freq = self.queryU(x)
                    if freq >= self.phi * self.S:
                        hh.add(freq)
        return np.array(list(hh))
    
    def getHH(self, dataset):
        self.processStream_HH(dataset)
        hh = self.hitter()
        return np.array(list(hh))
    
    def evaluateHH(self, res, real):
        tp = fp = fn = 0
        for i in res:
            if i in real:
                tp += 1
            else:
                fp += 1
        for j in real:
            if j not in res:
                fn += 1
        print("TP =",tp,"   FP =", fp,"   FN =", fn)
        recall = tp / (tp + fn)
        print('reacall:', recall)
        precision = tp / (tp + fp)
        print('precision:',precision)
        f1 = (2 * recall * precision) / (precision + recall)
        print('F1-score:',f1)
        return
    
    def rank(self, hhDict, test, topK=100):
        res = sorted(hhDict.items(), key=lambda item:item[1], reverse=True)[:topK]
        ranklist = list()
        for line in res:
            ranklist.append(line[0])
        hr, ndcg = self.evaluate(ranklist, test)
        print('HR@', topK, ' = %.4f' %  hr)
        print('NDCG@', topK, ' = %.4f' % ndcg)
        return hr, ndcg, ranklist
    
    def getHitRatio(self, ranklist, gtItem):
        #HR击中率，如果topk中有正例ID即认为正确
        if gtItem in ranklist:
            return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        #NDCG归一化折损累计增益
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return np.log(2) / np.log(i+2)
        return 0

    def evaluate(self, ranklist, test):
        hit_list = list()
        undcg_list = list()
        for line in test:
            user = line[0]
            pos_item = line[1]
            hit_list.append(self.getHitRatio(ranklist, pos_item))
            undcg_list.append(self.getNDCG(ranklist, pos_item))
        hr = np.mean(hit_list)
        ndcg = np.mean(undcg_list)
        return hr, ndcg