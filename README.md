# CRYPTOCURRENCY TRADING STUDY
A "bot" that uses technical indicators tests the profit of multiple trading strategies.

## Important
This is not the smartest way to find good trading strategies. **Use Machine Learning**.
I ran over 150 million different combinations across different periods of time to see their profits (40 hours to run). This is not the correct solution to the problem of finding a good trading strategy. The difficulty and time required increase exponentially.
But it was a nice challenge and I learned a lot from it. After I was done I proceeded to learn Machine Learning and Deep Learning. If you are thinking about doing a project similar to the one I did here, I'd recommend you go straight to Machine Learning (it's awesome).

## Short Explanation
  A brief explanation of the research, method and code.

### Goal
  The purpose of this research is to **determine the most profitable cryptocurrency trading strategy using 5 popular [technical indicators](https://en.wikipedia.org/wiki/Technical_indicator)**: Bollinger Bands, RSI, Moving Averages, MACD and Parabolic SAR. These are called "Momentum" and "Trend" type indicators.

### Method and Process
  In order to achieve this goal, hundreds of millions of strategies were tested on a Bitcoin chart of predetermined length. Then, the most profitable strategies were selected and tested on 15 different other time frames, all with the same length.

  For the computer to decide when to make trades, a bar of variable length was created. It ranges from -X to X. Each indicator impacts the bar in some amount, and if the sum of these impacts reaches X the computer places a buy order, and vice-versa. Here's a demonstration (X not defined).

  ![alt text](https://raw.githubusercontent.com/CaioCamatta/Crypto-Trading-Research/master/imgs/Figure_2.png)

  A strategy is composed of 5 motors (one for each indicator) and a value for X (bar size). These motors were created in order to decide how much each indicator will influence the bar. Every technical indicator requires you to input some variable values in order to output a value. The motors hold not only these values but also a Weight and some extra variables that help them make more realistic decisions.

  After calculating various possibilities for each motor (about 100 for each), the final step was to calculate every possible combination and its respective profit, get the best ones and apply them to other time frames. Looks simple, but this part required the computer to run more than a trillion calculations, so it took weeks to perfect the code in order to make this research viable.

## Results and conclusion
After some months working on this project, and after analyzing hundreds of millions of possible trading strategies with the five chosen indicators, I was able to reach an interesting conclusion. I was looking for a strategy that would give me the most profit, and fail the least amount of times. That was not the result I encountered, that's why I'm publishing this research. If I had found such, there would be no point in publishing the results, as it would ruin the whole research.

### Results
  After running the analysis on a 3week/30min (1008 ticks) Bitcoin chart, I was able to pick the 3840 best results (10 from each Moving Average option). These results were plotted in the following figure.

  ![alt text](https://raw.githubusercontent.com/CaioCamatta/Crypto-Trading-Research/master/imgs/bestProfits.png)

  Then, I reverse-engineered these values to find out which strategy was used in each of them, and applied those strategies in 15 other BTC charts, all with the same length of 1008 ticks - e.g. 0.5week/5min from one week ago or 1.5week/15min from one month ago. In the next messy figure, you can see the outcome of each strategy on every chart.

  ![alt text](https://raw.githubusercontent.com/CaioCamatta/Crypto-Trading-Research/master/imgs/crossAnalysis.png)

  You may reach some conclusions just by looking at the picture above, but for better analysis, here's a figure with the mean result and standard deviation from each strategy.

  ![alt text](https://raw.githubusercontent.com/CaioCamatta/Crypto-Trading-Research/master/imgs/MeanAndStd.png)

  The lowest standard deviation was 4.33% (Index: 439, Mean: 96.57%) and the highest mean was 99.86% (Index 3187, Std: 6.42%). Needless to say, the standard deviations are extremely high, thus removing all reliability from any strategy.

### Conclusion
  Unfortunately, I was not able to find a trading strategy that would allow me to effortlessly make millions by using this method. However, this research may be useful for some people in the trading community. Between all of the analysis, there wasn't a single fixed strategy that was inherently profitable and solid.

  This study only involved 5 indicators, but I have the impression that the results obtained here would be very similar for any other combinations of "momentum" and "trend" indicators. My conclusion is:

  **"Momentum" and "Trend" Technical Indicators are useless for longing cryptocurrency if used alone. You should not rely completely on them.**

  These types of indicators are the most used ones and based on this research I can say that they require extra rules or technics in order to reach a fixed profitable strategy. Therefore, if your strategy is based exclusively on "momentum" and "trend" technical indicators with fixed variables, I would recommend using other methods, because you there's a chance you're just trading randomly.

  In the future, I might do a study on Price Action trading. I have the impression that I can use it to identify a solid profitable strategy. Maybe because it inherently works, or maybe because everyone uses it, thus causing it to work.
