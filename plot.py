import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from matplotlib.finance import candlestick_ohlc
import matplotlib.pyplot as plt
from matplotlib import gridspec,style
import matplotlib.dates as mdates

style.use('ggplot')

# Load DataFrame
df = pd.read_csv('data.csv')

# Create new DataFrame for Analisys
dfan = pd.DataFrame()

# Create new DataFrame to store analisys results
dfr = pd.DataFrame(columns=['profit', 'barSize', 'cooldown', 'bbBool', 'bbLen', 'bbDev', 'bbA', 'bbBSOnReturnToMA', 'bbWeight', 'rsiBool', 'rsiLen', 'rsiA', 'rsiThreshold', 'rsiSProfitSecuring', 'rsiWeight', 'maBool', 'maThresh', 'ma1Len', 'ma1Type', 'ma2Len', 'ma2Type', 'maWeight', 'psarBool', 'psarInc', 'psarMax', 'psarWeight', 'macdBool', 'macdWeight'])
#dfr = pd.read_csv('dataAnalisysResults.csv')


# Set buy/sell bar. Once reached the cap, currency will be bought/sold. 100=buy, -100=sell
bar = 0
balanceUSD = 1000
balanceBTC = 0

# barList will keep track of the bar chages made by each indicator.
# bar = 0 , bb = 1 , rsi = 2 , macd = 3 , ma = 4 , psar = 5
barList = [[],[],[],[],[],[]]

# MA cross tracker
maCross = 1

# Analisys
def analisys(barSize, cooldown, bbBool, bbLen, bbDev, bbA, bbBSOnReturnToMA, bbWeight, rsiBool, rsiLen, rsiA, rsiThreshold, rsiSProfitSecuring, rsiWeight, maBool, maThresh, ma1Len, ma1Type, ma2Len, ma2Type, maWeight, psarBool, psarInc, psarMax, psarWeight, macdBool, macdWeight):
    """ General Parameters:
            'barSize' how much it's gotta reach to place an order
            'cooldown' lower weight of indicators after order
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
               'macdWeight' how important BB should be """

    # Buy/Sell. 0.2% fee
    def buy(date, close, bar, bb, rsi, macdhist, ma1, ma2, psar):
        global balanceBTC, balanceUSD, dfan

        balanceBTC += (balanceUSD/close)*0.998
        balanceUSD = 0
        profit = (balanceBTC * close)/1000
        dfan = dfan.append({'date': date, 'close': close, 'bar': bar, 'bb':bb, 'rsi': rsi, 'macd': macdhist, 'maFast': ma1, 'maSlow': ma2, 'psar': psar, 'type': 'BUY', 'profit': profit}, ignore_index=True)

    def sell(date, close, bar, bb, rsi, macdhist, ma1, ma2, psar):
        global balanceBTC, balanceUSD, dfan

        balanceUSD += (balanceBTC*close)*0.998
        balanceBTC = 0
        profit = balanceUSD/1000
        dfan = dfan.append({'date': date, 'close': close, 'bar': bar, 'bb':bb, 'rsi': rsi, 'macd': macdhist, 'maFast': ma1, 'maSlow': ma2, 'psar': psar, 'type': 'SELL', 'profit': profit}, ignore_index=True)

    # BB analisys
    def bb(price, bb, bbHigh, a, c):
        """ MOM """
        """ a: defines the slope. a~>0 = straight line. Goes from 0 to 100, tops
            b: activation value, always 1 in this case
            c: max bar fluctuation. default = 100 """
        global bar
        bbDev = bbHigh-bb
        if(price > bb):
            barChange = c/(1+np.exp((-a*((price-bb)/bbDev)+a*1)))
            bar += barChange
            barList[1].append(barChange)
        elif(price < bb):
            barChange = -c/(1+np.exp((-(-a)*((price-bb)/bbDev)+(-a)*-1)))
            bar += barChange
            barList[1].append(barChange)
        else:
            barList[1].append(0)

    # Rsi analisys
    def rsi(rsi, a, b, c):
        """ MOM """
        """ a: defines the slope. a~>0 = straight line. Goes from 0 to 2, tops
            b: activation value, rsiThreshold
            c: max bar fluctuation. default = 100"""
        global bar
        rsiX = rsi-50
        if(rsiX >= 0):
            barChange = c/(1+np.exp((-a*rsiX+a*b)))
            #barChange = (((rsiX-(rsiThreshold))*50)/(a+abs(rsiX-(rsiThreshold))))+40
            bar += barChange
            barList[2].append(barChange)
        else:
            barChange = -c/(1+np.exp((-(-a)*rsiX+(-a)*(-b))))
            #barChange = -(((-rsiX-(rsiThreshold))*50)/(a+abs(-rsiX-(rsiThreshold))))-40
            bar += barChange
            barList[2].append(barChange)

    # MACD analisys
    def macd(macdHist):
        """ TREND """
        """ 1st option = use macdHist to increase the bar
            2nd option = use MACD to signal cross """
        global bar
        barChange = macdHist
        bar += barChange
        barList[3].append(barChange)

    # MA analisys
    def ma(ma1, ma2):
        """ TREND. ma1=fast, ma2=slow """
        """ 1st option = use MA to signal cross """
        global bar
        global maCross
        if(ma1*(1-maThresh) > ma2):
            if(maCross == -1):
                bar += maWeight
                maCross = 1
                barList[4].append(maWeight)
            else:
                barList[4].append(0)
        elif(ma1*(1-maThresh) < ma2):
            if(maCross == 1):
                bar -= maWeight
                maCross = -1
                barList[4].append(-maWeight)
            else:
                barList[4].append(0)
        else:
            barList[4].append(0)

    # PSAR analisys
    def psar(psar, price):
        """ TREND """
        global bar
        if(psar>price):
            barChange = psarWeight
            bar += barChange
            barList[5].append(barChange)
        else:
            barChange = -psarWeight
            bar += barChange
            barList[5].append(barChange)

    # Run the analisys
    if __name__ == "__main__":
        global balanceBTC, balanceUSD, barList, dfan

        # Go through every row and analise all the info
        for index,row in df.iterrows():
            global bar

            if(bbBool):
                bb(row['close'], row[f'middleband{bbLen}n{bbDev}'], row[f'upperband{bbLen}n{bbDev}'], bbA, bbWeight)
            if(rsiBool):
                rsi(row[f'rsi{rsiLen}'], rsiA, rsiThreshold, rsiWeight)
            if(macdBool):
                macd(row['macdhist'])
            if(maBool):
                ma(row[f'{ma1Type}ma{ma1Len}'],row[f'{ma2Type}ma{ma2Len}'])
            if(psarBool):
                psar(row[f'sar_a{psarInc}m{psarMax}'], row['close'])

            # End each iteration by checking if bar is high/low enough to Buy/Sell. Takes about 0.0025s
            if(bar>=barSize and balanceUSD>0):
                buy(row['date'], row['close'], bar, row[f'lowerband{bbLen}n{bbDev}'], row[f'rsi{rsiThreshold}'], row['macdhist'], row[f'{ma1Type}ma{ma1Len}'],row[f'{ma2Type}ma{ma2Len}'], row[f'sar_a{psarInc}m{psarMax}'])
            elif(bar<=-barSize and balanceBTC>0):
                sell(row['date'], row['close'], bar, row[f'upperband{bbLen}n{bbDev}'], row[f'rsi{rsiThreshold}'], row['macdhist'], row[f'{ma1Type}ma{ma1Len}'],row[f'{ma2Type}ma{ma2Len}'], row[f'sar_a{psarInc}m{psarMax}'])

            barList[0].append(bar)

            # Zeroes bar for next iteration
            bar = 0

        # Sell at the end to check profits .012 from here to end
        #if(balanceBTC>0):
        #    sell(df.iloc[-1]['date'], df.iloc[-1]['close'], bar, df.iloc[-1][f'upperband{bbLen}n{bbDev}'], df.iloc[-1][f'rsi{rsiThreshold}'], df.iloc[-1]['macdhist'], df.iloc[-1][f'{ma1Type}ma{ma1Len}'],df.iloc[-1][f'{ma2Type}ma{ma2Len}'], df.iloc[-1][f'sar_a{psarInc}m{psarMax}'])

        # Save profits to dfr
        profit = balanceUSD/1000
        dfrLen = dfr.shape[0]
        dfr.loc[dfrLen] = [profit, barSize, cooldown, bbBool, bbLen, bbDev, bbA, bbBSOnReturnToMA, bbWeight, rsiBool, rsiLen, rsiA, rsiThreshold, rsiSProfitSecuring, rsiWeight, maBool, maThresh, ma1Len, ma1Type, ma2Len, ma2Type, maWeight, psarBool, psarInc, psarMax, psarWeight, macdBool, macdWeight]

        # Save detailed analisys of the biggest profit
        df['bar'] = barList[0]
        df['bbBarChange'] = barList[1]
        df['rsiBarChange'] = barList[2]
        df['macdBarChange'] = barList[3]
        df['maBarChange'] = barList[4]
        df['psarBarChange'] = barList[5]

        # Define plot function and plot
        def plot(ma1Len, ma1Type, ma2Len, ma2Type, bbDev, bbLen, rsiLen, rsiThreshold, psarInc, psarMax):
            # First tab
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios':[3,1,1,1]})
            quotes = df[['date', 'open', 'high', 'low', 'close']]
            candlestick_ohlc(ax1, quotes.values, width=0.4, colorup='g', colordown='r')

            ax1.set_ylabel('Chart')
            ax1.plot(df['date'], df[f'{ma1Type}ma{ma1Len}'], 'r', df['date'], df[f'{ma2Type}ma{ma2Len}'], 'b', df['date'], df[f'upperband{bbLen}n{bbDev}'], 'y--', df['date'], df[f'lowerband{bbLen}n{bbDev}'], 'y--')
            ax1.scatter(df['date'], df[f'sar_a{psarInc}m{psarMax}'], s=2, color='k')

            ax2.set_ylabel('Rsi')
            ax2.plot(df['date'], df[f'rsi{rsiLen}'], color='k')
            ax2.fill_between(df['date'], rsiThreshold+50, df[f'rsi{rsiLen}'], where=df[f'rsi{rsiLen}'] >= rsiThreshold+50, facecolor='green')
            ax2.fill_between(df['date'], 50-rsiThreshold, df[f'rsi{rsiLen}'], where=df[f'rsi{rsiLen}'] <= 50-rsiThreshold, facecolor='green', interpolate=True)

            ax3.set_ylabel('Macd')
            ax3.plot(df['date'], df['macdsignal'], 'r', df['date'], df['macd'], 'k')

            ax4.set_ylabel('Bar')
            ax4.plot(df['date'], df['bar'], 'g')

            for index,row in dfan.iterrows():
                if (row['type'] == 'BUY'):
                    ax1.scatter(row['date'], row['close'], s=40, color='g')
                    ax4.scatter(row['date'], row['bar'], s=40, color='g')
                elif (row['type'] == 'SELL'):
                    ax1.scatter(row['date'], row['close'], s=40, color='r')
                    ax4.scatter(row['date'], row['bar'], s=40, color='r')

            # Second tab
            fig2, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, sharey=True, gridspec_kw={'height_ratios':[1,1,1,1,1,1]})

            ax1.set_ylabel('BB')
            ax1.plot(df['date'], df['bbBarChange'])

            ax2.set_ylabel('Rsi')
            ax2.plot(df['date'], df['rsiBarChange'])

            ax3.set_ylabel('Macd')
            ax3.plot(df['date'], df['macdBarChange'])

            ax4.set_ylabel('Ma')
            ax4.plot(df['date'], df['maBarChange'])

            ax5.set_ylabel('Psar')
            ax5.plot(df['date'], df['psarBarChange'])

            ax6.set_ylabel('Sum = bar')
            ax6.plot(df['date'], df['bar'], 'g')

            fig.subplots_adjust(hspace=0.09, top=.95, right=.95, bottom=.05, left=.08)
            fig2.subplots_adjust(hspace=0.09, top=.95, right=.95, bottom=.05, left=.08)
            plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        plot(ma1Len, ma1Type, ma2Len, ma2Type, bbDev, bbLen, rsiLen, rsiThreshold, psarInc, psarMax)


# Run analisys
start = time.time()

################# Change this to whatever values you want as long as they are included in addData.py #################
analisys(100, None, True, 20, 2.0, 25, None, 100, True, 14, 0.4, 20, None, 100, True, 0, 10, 's', 25, 'e', 50, True, 0.02, 0.2, 10, True, 50)

end = time.time()
print(end - start)

# Show chart
plt.show()

# Save
#dfan.to_csv('dataAnalisys.csv', index=True)
#dfr.to_csv('dataAnalisysResults.csv', index=True)
