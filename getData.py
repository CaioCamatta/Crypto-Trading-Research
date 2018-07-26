import poloniex
import time
import pandas as pd
import os

# Set time labels
MINUTE, HOUR, DAY, WEEK, MONTH, YEAR = \
    60, 60 * 60, 60 * 60 * 24, 60 * 60 * 24 * \
    7, 60 * 60 * 24 * 30, 60 * 60 * 24 * 365

# Connect to poloniex
print('Connecting to Poloniex')
polo = poloniex.Poloniex('your key','secret')

# Get data from poloniex
print('Getting data')
data0 = polo.returnChartData('USDT_BTC', 900, time.time() - YEAR*1.5, time.time())# Create DataFrame
print('Creating DataFrame')
df0 = pd.DataFrame(data0)

# Save to .csv
print('Saving to .csv')
df0.to_csv('data/data0.csv')

# Exit
print('Done')
