import time
import pandas as pd
from talib.abstract import *

# Decide what to add. Prevents rework
doDate=True
doSma=True
doEma=True
doRsi=True
doMacd=True
doBB=True
doSAR=True
doMOM=False

for ind in range(15):
    # Load DataFrame
    df = pd.read_csv(f'data/data{ind}.csv')
    inputs = {
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume'],
    }

    # Add 'strDate' column
    if(doDate):
        print("ADD: Readable date")
        for index,row in df.iterrows():
            df.loc[index,'strDate'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row['date']))

    # Add SMA
    if(doSma):
        print("ADD: SMA")
        for x in range(5,49):
            print("    ...Adding SMA"+str(x))
            sma = SMA(inputs, timeperiod=x, price='close')
            df['sma'+str(x)] = sma
        for x in [5*n for n in range(10,40)]:
            print("    ...Adding SMA"+str(x))
            sma = SMA(inputs, timeperiod=x, price='close')
            df['sma'+str(x)] = sma

    # Add EMA
    if(doEma):
        print("ADD: EMA")
        for x in range(5,49):
            print("    ...Adding EMA"+str(x))
            ema = EMA(inputs, timeperiod=x, price='close')
            df['ema'+str(x)] = ema
        for x in [5*n for n in range(10,40)]:
            print("    ...Adding EMA"+str(x))
            ema = EMA(inputs, timeperiod=x, price='close')
            df['ema'+str(x)] = ema

    # Add RSI
    if(doRsi):
        print("ADD: RSI")
        for x in range(4,30):
            print("    ...Adding RSI"+str(x))
            rsi = RSI(inputs, timeperiod=x, price='close')
            df['rsi'+str(x)] = rsi

    # Add MACD
    if(doMacd):
        print("ADD: MACD")
        macd, macdsignal, macdhist = MACD(inputs, fastperiod=12, slowperiod=26, signalperiod=9, price='close')
        df['macd'] = macd
        df['macdsignal'] = macdsignal
        df['macdhist'] = macdhist

    # Add BB
    if(doBB):
        print("ADD: Bollinger Bands")
        for x in [n for n in range(10,40)]:
            for y in [5*n for n in range(2,10)]:
                print("    ...Adding BB"+str(x)+"n"+str(y/10))
                upperband, middleband, lowerband = BBANDS(inputs, timeperiod=x, nbdevup=y/10, nbdevdn=y/10, matype=0, price='close')
                df['upperband'+str(x)+"n"+str(y/10)] = upperband
                df['middleband'+str(x)+"n"+str(y/10)] = middleband
                df['lowerband'+str(x)+"n"+str(y/10)] = lowerband

    # Add SAR
    if(doSAR):
        print("ADD: SAR")
        for x in [n for n in range(1,5)]:
            for y in [n for n in range(3,35)]:
                print("    ...Adding SAR_a"+str(x/100)+"m"+str(y/100))
                sar = SAR(inputs, high='high', low='low', acceleration=x/100, maximum=y/100)
                df['sar_a'+str(x/100)+"m"+str(y/100)] = sar

    # Add Momentum
    if(doMOM):
        print("ADD: Mom")
        for x in range(4, 26):
            print("    ...Adding Mom"+str(x))
            mom = MOM(inputs, timeperiod=x, price='close')
            df['mom'+str(x)] = mom

    # Save to .csv
    print("Number of rows: {0}, columns: {1}, total values: {2}".format(df.shape[0], df.shape[1], df.shape[0]*df.shape[1]))
    print("Done. Saving...")
    df.to_csv(f'data/completeData{ind}.csv', index=False)
    df = df.iloc[0:0]
