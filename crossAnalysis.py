import time
import os
import multiprocessing
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import itertools
from matplotlib.finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import gridspec,style
import matplotlib.dates as mdates

# Just so we know in which step of the analysis we are
runIndicators = False
runProfits = False
runAnalysis = True
runStatisticsCsv = False

# Motors
if(runIndicators):
    for ind in range(15):
        # Load DataFrame
        df = pd.read_csv(f'data/completeData{ind}.csv')

        # Create DataFrame to save indicatoranalysis
        dfi = pd.DataFrame()

        # MA cross tracker
        maCross = 1
        macdCross = 1

        # Step 3. Calculate the motors. PS: could've used kwargs here
        def indicatoranalysis(bbBool, bbLen, bbDev, bbA, bbBSOnReturnToMA, bbWeight, rsiBool, rsiLen, rsiA, rsiThreshold, rsiSProfitSecuring, rsiWeight, maBool, maThresh, ma1Len, ma1Type, ma2Len, ma2Type, maWeight, psarBool, psarInc, psarMax, psarWeight, macdBool, macdType, macdThresh, macdWeight):
            """ General Parameters:
                BB = 'bbBool' tells if should use BB or not
                     'bbLen' amount of days to be considered. df = 20
                     'bbDev' standard deviation. df = 2
                     'bbA' defines the slope of the function. Goes from 0 to 100, tops
                     'bbBSOnReturnToMA' B/S when hits line and goes back to the MA
                     'bbWeight' max bar fluctuation
                RSI = 'rsiBool' tells if should use RSI or not
                      'rsiLen' rsi(days), 14 days usually the standard
                      'rsiA' slope of the function. Goes from 0 to 2, max
                      'rsiThreshold' distance from 50. 15 means that a rsi of 65 will signal overbought
                      'rsiSProfitSecuring' sell after 'x%' of profit
                      'rsiWeight' max bar fluctuation
                MA = 'maBool' tells if should use RSI or not
                     'maThresh' B/S after crossing 'x%' up or below. Values given in 0.01=1%
                     'ma1Len' quicker MA lenght
                     'ma1Type' 'E'ma or 'S'ma
                     'ma2Len' slower MA lenght
                     'ma2Type' 'E'ma or 'S'ma
                     'maWeight' max bar fluctuation.
                PSAR = 'psarBool' whether to use PSAR
                       'psarInc' increment over time df = 0.02
                       'psarMax' max value. df = 0.2
                       'psarWeight' the amount that will be added/remove from the bar
                MACD = 'macdBool' whether to use MACD
                       'macdType' type of analysis in use (cross/hist)
                       'macdThresh' how much macdHist must hit before making any changes to the bar
                       'macdWeight' macdHist multiplier."""

            # BB analysis
            def bb(price, bb, bbHigh, a, c):
                """ MOM """
                """ a: defines the slope. a~>0 = straight line. Goes from 0 to 100, tops
                    b: activation value, always 1 in this case
                    c: max bar fluctuation. default = 100 """
                bbDev = bbHigh-bb
                if(price > bb):
                    barChange = c/(1+np.exp((-a*((price-bb)/bbDev)+a*1)))
                    return barChange
                elif(price < bb):
                    barChange = -c/(1+np.exp((-(-a)*((price-bb)/bbDev)+(-a)*-1)))
                    return barChange
                else:
                    return 0

            # Rsi analysis
            def rsi(rsi, a, b, c):
                """ MOM """
                """ a: defines the slope. a~>0 = straight line. Goes from 0 to 2, tops
                    b: activation value, rsiThreshold
                    c: max bar fluctuation. default = 100"""
                rsiX = rsi-50
                if(rsiX >= 0):
                    barChange = c/(1+np.exp((-a*rsiX+a*b)))
                    #barChange = (((rsiX-(rsiThreshold))*50)/(a+abs(rsiX-(rsiThreshold))))+40
                    return barChange
                else:
                    barChange = -c/(1+np.exp((-(-a)*rsiX+(-a)*(-b))))
                    #barChange = -(((-rsiX-(rsiThreshold))*50)/(a+abs(-rsiX-(rsiThreshold))))-40
                    return barChange

            # MACD analysis
            def macd(price, macdHist):
                """ TREND """
                """ 1st option = use macdHist to increase the bar
                    2nd option = use MACD to signal cross """
                global macdCross
                if(macdType == 1):
                    barChange = (macdHist*10000/price)*macdWeight
                    return barChange
                if(macdType == 2):
                    if(macdHist >= 5):
                        if(macdCross == -1):
                            macdCross = 1
                            return macdWeight*100
                        elif(macdCross==0):
                            macdCross = 1
                            return macdWeight*100
                        else:
                            return 0
                    elif(macdHist <= 5):
                        if(macdCross == 1):
                            macdCross = -1
                            return -macdWeight*100
                        elif(macdCross==0):
                            macdCross = -1
                            return macdWeight*100
                        else:
                            return 0
                    else:
                        return 0

            # MA analysis
            def ma(ma1, ma2):
                """ TREND. ma1=fast, ma2=slow """
                """ 1st option = use MA to signal cross """
                global maCross
                if(ma1*(1-maThresh) > ma2):
                    if(maCross == -1):
                        maCross = 1
                        return maWeight
                    elif(maCross==0):
                        macdCross = 1
                        return maWeight
                    else:
                        return 0
                elif(ma1*(1+maThresh) < ma2):
                    if(maCross == 1):
                        maCross = -1
                        return -maWeight
                    elif(maCross==0):
                        macdCross = -1
                        return -maWeight
                    else:
                        return 0
                else:
                    return 0

            # PSAR analysis
            def psar(psar, price):
                """ TREND """
                if(psar>price):
                    barChange = psarWeight
                    return barChange
                else:
                    barChange = -psarWeight
                    return barChange

            # Run the analysis
            if __name__ == "__main__":
                global balanceBTC, balanceUSD, barList

                barChanges = []
                # Go through every row and analise all the info
                for index,row in df.iterrows():

                    # Check indicators for each of the rows. Calculate based on variables passed to the function
                    if(bbBool):
                        change = bb(row['close'], row[f'middleband{bbLen}n{bbDev}'], row[f'upperband{bbLen}n{bbDev}'], bbA, bbWeight)
                        if(change<-1 or change>1):
                            barChanges.append(round(change))
                        else:
                            barChanges.append(0)
                    if(rsiBool):
                        change = rsi(row[f'rsi{rsiLen}'], rsiA, rsiThreshold, rsiWeight)
                        if(change<-1 or change>1):
                            barChanges.append(round(change))
                        else:
                            barChanges.append(0)
                    if(macdBool):
                        change = macd(row['close'], row['macdhist'])
                        if(change<-1 or change>1):
                            barChanges.append(round(change))
                        else:
                            barChanges.append(0)
                    if(maBool):
                        change = ma(row[f'{ma1Type}ma{ma1Len}'], row[f'{ma2Type}ma{ma2Len}'])
                        if(change<-1 or change>1):
                            barChanges.append(round(change))
                        else:
                            barChanges.append(0)
                    if(psarBool):
                        change = psar(row[f'sar_a{psarInc}m{psarMax}'], row['close'])
                        barChanges.append(round(change))

                # Save the the list that waas generated from the analysis to the Dataframe. Each indicator provides a list equal to the length of the chart, for each row there's a change made to the 'bar'
                if(bbBool):
                    dfi[f'bb{bbLen}-d{bbDev}-a{bbA}-{bbWeight}'] = barChanges
                if(rsiBool):
                    dfi[f'rsi{rsiLen}-t{rsiThreshold}-a{rsiA}-{rsiWeight}'] = barChanges
                if(macdBool):
                    dfi[f'macd-{macdType}-{macdWeight}'] = barChanges
                if(maBool):
                    dfi[f'{ma1Type}ma{ma1Len}+{ma2Type}ma{ma2Len}-t{maThresh}-{maWeight}'] = barChanges
                if(psarBool):
                    dfi[f'psar-i{psarInc}-m{psarMax}-{psarWeight}'] = barChanges

        # Run analysis for each individual indicator. Avg time on December 26: 0.058s
        t1 = time.time()

        # Run all bb analysis, saves column as: 'bb{bbLen}-d{bbDev}-a{bbA}-{bbWeight}'
        for l in [12,20,35]:
            for d in [2.0,3.0,4.0]:
                for a in [20,40]:
                    for w in [50,100,150]:
                        indicatoranalysis(True, l, d, a, 0, w, False, 0, 0, 0, 0, 0, False, 0, 0, 0, 0, 0, 0, False, 0, 0, 0, False, 0, 0, 0)
        t2 = time.time()
        print(t2 - t1)

        # Run all rsi analysis, saves column as: 'rsi{rsiLen}-t{rsiThreshold}-a{rsiA}-{rsiWeight}'
        for l in [9,14,22]:
            for t in [15,20,25,30]:
                for a in [0.5,0.1]:
                    for w in [50,100,150]:
                        indicatoranalysis(False, 0, 0, 0, 0, 0, True, l, a, t, 0, w, False, 0, 0, 0, 0, 0, 0, False, 0, 0, 0, False, 0, 0, 0)
        t3 = time.time()
        print(t3 - t2)

        # Run all ma analysis, saves column as: '{ma1Type}ma{ma1Len}+{ma2Type}ma{ma2Len}-t{maThresh}-{maWeight}'
        for t in [0,0.015]:
            for fl in [10,20,30,40]:
                for ft in ['e', 's']:
                    for sl in [10,20,35]:
                        for st in ['e', 's']:
                            for w in [50,100,150,200]:
                                indicatoranalysis(False, 0, 0, 0, 0, 0, False, 0, 0, 0, 0, 0, True, t, fl, ft, sl+fl, st, w, False, 0, 0, 0, False, 0, 0, 0)
        t4 = time.time()
        print(t4 - t3)

        # Run all psar analysis, saves column as: 'psar-i{psarInc}-m{psarMax}-{psarWeight}'
        for i in [0.01,0.02,0.03]:
            for m in [3,6,10]:
                    for w in [20,50,100]:
                        indicatoranalysis(False, 0, 0, 0, 0, 0, False, 0, 0, 0, 0, 0, False, 0, 0, 0, 0, 0, 0, True, i, m*i, w, False, 0, 0, 0)
        t5 = time.time()
        print(t5 - t4)

        # Run all macd analysis, saves column as: 'macd-{macdWeight}'
        for t in [1,2]:
            for w in [0.5,1,1.5]:
                indicatoranalysis(False, 0, 0, 0, 0, 0, False, 0, 0, 0, 0, 0, False, 0, 0, 0, 0, 0, 0, False, 0, 0, 0, True, t, 5, w)
        t6 = time.time()
        print(t6 - t5)

        t7 = time.time()
        print(t7 - t1)
        print('Average time = '+str((t7-t1)/668))
        dfi.to_csv(f'data/indicatorAnalysis{ind}.csv', index=True)

if(runProfits):
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
    def analysisControler(index):
        chartLen = dfi.shape[0]

        # Create DF to store profits
        dfb = pd.DataFrame()

        # Calculate profit for each column of topProfits.csv separately
        for m in range(384):
            for i in range(10):
                # Current column being analysed
                config = dfp[f'{m}-{i}'][2]

                # Read each fourth line from topProfits.csv and decompose it
                bb, rsi, ma, psar, macd = map(int, config.split("-"))

                # Calculate profit
                profit = analysis([bb,rsi,ma,psar,macd], 100, chartLen, 0)

                # Add profit to DF
                dfb[f'{m}-{i}'] = [profit]
        print('Saving')
        dfb.to_csv(f'data/dataProfits{index}.csv', index=False)

    # Multiprocessing
    if __name__ == '__main__':
        for ind in range(15):
            dfp = pd.read_csv(f'1008/topProfits.csv')
            dfi = pd.read_csv(f'data/indicatorAnalysis{ind}.csv')
            df = pd.read_csv(f'data/data{ind}.csv')
            analysisControler(ind)

if(runAnalysis):
    dfp = pd.read_csv(f'data/topProfits.csv')
    # Create DF for statistics
    dfs = pd.DataFrame()

    # Work on statistics
    if(runStatisticsCsv):
        for x in range(384):
            for i in range(10):
                print(x)
                columnProfit = []
                for y in range(15):
                    df = pd.read_csv(f'data/dataProfits{y}.csv')
                    columnProfit.append(df[f'{x}-{i}'][0])
                columnProfitDF = pd.Series(columnProfit)
                dfs[f'{x}-{i}'] = columnProfitDF.describe()
        print('Saving Statistics')
        dfs.to_csv(f'data/statistics.csv', index=True)
    else:
        dfs = pd.read_csv('data/statistics.csv')

    # Get best strategies
    std = []
    mean = []
    for x in range(384):
        for i in range(10):
            std.append(dfs[f'{x}-{i}'][2])
            mean.append(dfs[f'{x}-{i}'][1])

    print("Best STD: " + str(min(std)) + "  Index: " +  str(std.index(min(std))))
    print("Best Mean: " + str(max(mean)) + "  Index: " + str(mean.index(max(mean))))

    # Plot
    fig, (ax1) = plt.subplots(1, sharex=True, gridspec_kw={'height_ratios':[1]})

    # Plot topProfits
    dfpY = []
    dfpX = []
    for x in range(3840):
        dfpX.append(x)

    for x in range(384):
        for i in range(10):
            dfpY.append(float(dfp[f'{x}-{i}'][1]))
    ax1.scatter(dfpX, dfpY, label="Best Results", c='#00E88C', s=6)

    # Labels
    ax1.set_ylabel('Profit')

    # Save
    plt.savefig('bestProfits.png', bbox_inches='tight')

    # Plot each of the 10 other DFs
    fig2, (ax2) = plt.subplots(1, sharex=True, gridspec_kw={'height_ratios':[1]})
    ax2.set_color_cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue'])
    for y in range(15):
        df = pd.read_csv(f'data/dataProfits{y}.csv')
        dfY = []
        for x in range(384):
            for i in range(10):
                dfY.append(df[f'{x}-{i}'][0])
        ax2.scatter(dfpX, dfY, label=f"Plot {y}", s=4)

    # Labels
    ax2.set_ylabel('Profit')

    # Save
    plt.savefig('crossAnalysis.png', bbox_inches='tight')

    # Plot STD and MEAN
    fig3, (ax3, ax4) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[1,1]})

    meanList = []
    stdList = []
    for x in range(384):
        for i in range(10):
            meanList.append(dfs[f'{x}-{i}'][1])
            stdList.append(dfs[f'{x}-{i}'][2])
    ax3.scatter(dfpX, meanList, label=f"Mean", c='#00E88C', s=7)
    ax4.scatter(dfpX, stdList, label=f"St. Deviation", c='#FF4026', s=7)

    # Labels
    ax3.set_ylabel('Mean')
    ax4.set_ylabel('Standard Deviation')

    # Save
    plt.savefig('MeanAndStd.png', bbox_inches='tight')

    # Allows for legend
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    plt.show()

    #for x in range(384):
    #    df0[f'{x}'][0]
