import time
import os
import multiprocessing
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import itertools
from matplotlib.finance import candlestick_ohlc
import matplotlib.pyplot as plt
from matplotlib import gridspec,style
import matplotlib.dates as mdates

# Just so we know in which step of the analysis we are
runAnalysis = False
runGetMax = True

# Analysis
def analysis(x, barSize, chartLen, cooldown):
    """ General Parameters:
            'x'='bb, rsi, ma, psar, macd' lists that contain every indicators' column number
            'barSize' how much it's gotta reach to place an order
            'cooldown' lower weight of indicators after order"""

    # Decompose x (which is a list) into 5 variables
    bb = x[0]
    rsi = x[1]
    ma = x[2]
    psar = x[3]
    macd = x[4]

    # Buy/sell function
    def bs():
        profit=1
        buy=0
        for i in range(chartLen):
            if(buy==0):
                if(dataSum[i]>=barSize):
                    buy=df['close'][i]
            else:
                if(dataSum[i]<=barSize):
                    profit = profit*(df['close'][i]/buy)*0.998
                    buy=0
        return profit

    # Use numpy to sum the 5 values, 1 for every type of indicator
    #ch = [sum(x) for x in itertools.product(bb,rsi,ma,psar,macd)]
    data = np.array([dfi.iloc[:,bb],dfi.iloc[:,rsi],dfi.iloc[:,ma],dfi.iloc[:,psar],dfi.iloc[:,macd]])
    dataSum = np.sum(data, 0)
    bs = bs()
    p = (round(bs*1000))/1000
    return p

# Controls what analysis will be run, and allows for multiprocessing
def analysisControler(something):
    chartLen = dfi.shape[0]

    bb = []
    rsi = []
    ma = []
    psar = []
    macd = []

    # Appends the index of each indicator (from dfi) to the respectful variable
    for b in range(1,55):
        bb.append(b)
    for r in range(55,127):
        rsi.append(r)
    # decided to not use MA here, instead I'm using it to multiprocess. each cpu thread handles 1 variation of 'm'
    #for m in range(127,511):
    #    ma.append(m)
    for p in range(511,538):
        psar.append(p)
    for c in range(538,541):
        macd.append(c)

    # Devides all the analysis into 512 (number of MA analysis), to ensure no the analysis as a whole is made via safe steps
    for m in [something]:
        mAsList = [m]
        start3 = time.time()
        listOfProfits = ['-'.join([str(y) for y in x]) for x in itertools.product(bb,rsi,mAsList,psar,macd)]
        end3 = time.time()
        print(end3 - start3)


        # Create Dataframe that will store the actual analysis of the profits. Creates multiple archives to later combine than. This method is used to avoid trouble with multiprocessing
        dfb = pd.DataFrame()
        dfb[f'ma{m}'] = listOfProfits
        dfb.to_csv(f'{chartLen}/reverseBarAnalysis{m}.csv', index=False)

# Get highest values in dataframe, to later
def getMax():
    dfp = pd.read_csv('1008/barAnalysis.csv')
    topProfits = pd.DataFrame()
    for y in range(384):
        print(y)
        x = y+127
        dfr = pd.read_csv(f'1008/reverseBarAnalysis{x}.csv')
        idxmax = dfp[f'{y}'].nlargest(10)
        for i in range(len(idxmax)):
            topProfits[f'{y}-{i}'] = [idxmax.index[i], dfp[f'{y}'][idxmax.index[i]], dfr[f'ma{x}'][idxmax.index[i]]]
    topProfits.to_csv('1008/topProfits1.csv', index=False)

# Multiprocessing
if __name__ == '__main__':
    if(runAnalysis):
        data=[]
        for m in range(127,511):
            data.append(m)
        p=multiprocessing.Pool(4)
        p.map(analysisControler, data)

    if(runGetMax):
        getMax()
    #os.system('shutdown -s')
