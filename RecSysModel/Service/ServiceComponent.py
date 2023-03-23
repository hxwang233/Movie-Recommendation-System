#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
import pandas as pd
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import json


# In[ ]:


from RcmModelClass import NCF
from RcmModelClass import ENMF
from MVSketch import HHtracer


# In[ ]:


app = Flask(__name__)

hh_tracer = None
enmf = None
ncf = None

@app.route('/model/active_user',methods=['GET','POST'])
def getActiveUser_list():
    user_topK = np.loadtxt("./User_TopK.csv", delimiter=",", skiprows=1, dtype=int).tolist()
    data = dict()
    data['status'] = 200
    data['message'] = 'success'
    data['result'] = user_topK
    return json.dumps(data)

@app.route('/model/hot_movie',methods=['GET','POST'])
def getHotMovie_list():
    movie_topK = np.loadtxt("./Movie_TopK.csv", delimiter=",", skiprows=1, dtype=int).tolist()
    data = dict()
    data['status'] = 200
    data['message'] = 'success'
    data['result'] = movie_topK
    return json.dumps(data)

@app.route('/model/user_age',methods=['GET','POST'])
def getUserAgeDistribute():
    user_age = np.load('./age.npy').item()
    data = dict()
    data['status'] = 200
    data['message'] = 'success'
    data['result'] = user_age
    return json.dumps(data)

@app.route('/model/user_gender',methods=['GET','POST'])
def getUserGenderDistribute():
    user_gender = np.load('./gender.npy').item()
    data = dict()
    data['status'] = 200
    data['message'] = 'success'
    data['result'] = user_gender
    return json.dumps(data)

@app.route('/model/user_occupation',methods=['GET','POST'])
def getUserOccupationDistribute():
    user_occupation = np.load('./occupation.npy').item()
    data = dict()
    data['status'] = 200
    data['message'] = 'success'
    data['result'] = user_occupation
    return json.dumps(data)

@app.route('/hot/<uid>',methods=['GET','POST'])
def getHotRcm(uid):
    global ncf, hh_tracer
    hotMovie = hh_tracer.hitter()
    user = np.array([int(uid) for i in range(len(hotMovie))]) - 1
    user = user.reshape(-1,1)
    movie = hotMovie.reshape(-1,1)
    rcmlist = list(callNCF(ncf, user, movie, hotMovie) + 1)
    data = dict()
    data['status'] = 200
    data['message'] = 'success'
    data['result'] = [int(temp) for temp in rcmlist]
    return json.dumps(data)

def callNCF(model, user, movie, hotMovie):
    torch_x1 = torch.from_numpy(user).type(torch.LongTensor)
    torch_x2 = torch.from_numpy(movie).type(torch.LongTensor)
    if torch.cuda.is_available():
        torch_x1, torch_x2 = torch_x1.cuda(), torch_x2.cuda()
    prediction = model(torch_x1, torch_x2)
    pred_vector = -1 * (prediction.cpu().data.numpy().reshape(-1))
    ranklist = hotMovie[np.argsort(pred_vector)][:8]
    print(ranklist)
    return ranklist

@app.route('/model/<uid>',methods=['GET','POST'])
def getPersonalRcm(uid):
    global enmf
    user = np.array([int(uid)]) - 1
    rcmlist = list(callENMF(enmf, user) + 1)
    data = dict()
    data['status'] = 200
    data['message'] = 'success'
    data['result'] = [int(temp) for temp in rcmlist]
    return json.dumps(data)

def callENMF(model, user):
    torch_x1 = torch.from_numpy(user).type(torch.LongTensor)
    if torch.cuda.is_available():
        torch_x1 = torch_x1.cuda()
    prediction = model.rank(torch_x1)
    pred_vector = -1 * (prediction.cpu().data.numpy())[0]
    ranklist = np.argsort(pred_vector)[:8]
    print(ranklist)
    return ranklist
    
if __name__ == '__main__':
    enmf = torch.load('./model/ENMF.pkl')
    ncf  = torch.load('./model/NCF.pkl')
    dataset = np.loadtxt('./ml-1m/ratings.dat', delimiter='::', usecols=[0,1], dtype=int)
    dataset[:,0] = dataset[:,0]-1
    dataset[:,1] = dataset[:,1]-1
    phi, S  = round(1.0 / np.max(dataset[:,1]), 5), dataset.shape[0]
    delta   = 0.05
    epsilon = 0.002
    r = round(np.log2(1 / delta)).astype(np.int)
    w = round(2 / epsilon)
    hh_tracer = HHtracer(w, r, phi, S)
    hh_tracer.processStream_HH(dataset)
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=8081)


# In[ ]:





# In[ ]:




