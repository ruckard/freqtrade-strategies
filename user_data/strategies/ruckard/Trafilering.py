# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from datetime import datetime

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
    stoploss_from_absolute,
)

from freqtrade.exchange import (
    timeframe_to_prev_date
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class Trafilering(IStrategy):

    INTERFACE_VERSION = 3
    timeframe = "1h"
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {"60": 0.1, "30": 0.2, "0": 0.2}
    minimal_roi = {"0": 1}

    stoploss = -0.05
    can_short = True

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # This function should return traileringLong so that we can later on decide to end into a trend or not
    def populate_trailering_long(self, dataframe: DataFrame, metadata: dict, offset: int) -> DataFrame:

        df = dataframe.copy()

        i_trailering_atr_take_profit_length = 14 # TODO: Input
        i_virt_trailering_offset = offset
        i_virt_trailering_period = 24 # TODO: Input
        i_traileringAtrTakeProfitMultiplier = 1.25 # TODO: Input
        i_virtTakeProfitMaximumProfitExcursionRatio = 1.00 # TODO: Input
        i_virtTakeProfitMaximumLossExcursionRatio = 1.00 # TODO: Input
        i_breakEven = 0.20 # TODO: Input
        i_breakEvenPerOne = i_breakEven * 0.01

        # Add new columns
        df["virtLongTakeProfitIsReached"] = None
        df["virtShortTakeProfitIsReached"] = None
        df["virtTraileringBoth"] = None
        df["virtInsideAPosition"] = None
        df["virtLongMaximumProfitPrice"] = None
        df["virtShortMaximumProfitPrice"] = None
        df["virtLongMaximumLossPrice"] = None
        df["virtShortMaximumLossPrice"] = None
        df["virtLongTakeProfit"] = None
        df["virtShortTakeProfit"] = None
        df["traileringLong"] = None
        df["traileringShort"] = None

        for i in range(i_trailering_atr_take_profit_length, len(df)):
            # Only once init
            if (i == i_trailering_atr_take_profit_length):
                virtLongMaximumProfitPrice = None
                virtLongMaximumLossPrice = None
                virtShortMaximumProfitPrice = None
                virtShortMaximumLossPrice = None
                traileringAtrLongTakeProfitPrice = None
                traileringAtrShortTakeProfitPrice = None
                traileringLongTakeProfitPerOne = None
                traileringShortTakeProfitPerOne = None
                virtLongTakeProfit = None
                virtShortTakeProfit = None
                virtExpectedTakeProfit = None
                resetVirtVariables = False

                virtInsideAPosition = False
                virtPositionPrice = df["close"].iat[i]

            # Init on each loop
            virtLongMaximumProfitExcursion = 0.0
            virtShortMaximumProfitExcursion = 0.0
            virtShortMaximumLossExcursion = 0.0
            i_traileringAtrTakeProfitSource = df["close"].iat[i] # TODO: Input

            virtTraileringBoth = ((df["date"].dt.hour.iat[i] + i_virt_trailering_offset) % i_virt_trailering_period) == 0

            if (virtLongTakeProfit is not None):
                virtLongTakeProfitIsReached = (virtLongTakeProfit >= df["low"].iat[i]) and (virtLongTakeProfit <= df["high"].iat[i])
            else:
                virtLongTakeProfitIsReached = False
            if (virtShortTakeProfit is not None):
                virtShortTakeProfitIsReached =  (virtShortTakeProfit >= df["low"].iat[i]) and (virtShortTakeProfit <= df["high"].iat[i])
            else:
                virtShortTakeProfitIsReached = False

            if (not (df["virtLongTakeProfitIsReached"].iat[i - 1] or df["virtShortTakeProfitIsReached"].iat[i - 1])):
                if (True):
                #if ((not insideASellOrder) and (not insideABuyOrder)):
                    if ((not df["virtTraileringBoth"].iat[i - 1]) and (virtTraileringBoth)):
                        if (virtInsideAPosition):
                            resetVirtVariables = True
                        virtInsideAPosition = True
                        virtPositionPrice = df["close"].iat[i]

            if (df["virtInsideAPosition"].iat[i - 1] and (virtLongTakeProfitIsReached or virtShortTakeProfitIsReached)):
                virtInsideAPosition = False

            tmpTraileringAtrLongTakeProfitPrice = i_traileringAtrTakeProfitSource + (df["traileringAtr"].iat[i] * i_traileringAtrTakeProfitMultiplier)
            tmpTraileringAtrShortTakeProfitPrice = i_traileringAtrTakeProfitSource - (df["traileringAtr"].iat[i] * i_traileringAtrTakeProfitMultiplier)

            if ((not virtInsideAPosition) or ((not df["virtInsideAPosition"].iat[i - 1]) and virtInsideAPosition) or (resetVirtVariables)):
                traileringAtrLongTakeProfitPrice = tmpTraileringAtrLongTakeProfitPrice
                traileringAtrShortTakeProfitPrice = tmpTraileringAtrShortTakeProfitPrice
                virtLongTakeProfit = traileringAtrLongTakeProfitPrice
                virtShortTakeProfit = traileringAtrShortTakeProfitPrice

            # maximumProfitExcursion - BEGIN

            # Long
            if ((not df["virtInsideAPosition"].iat[i - 1]) and virtInsideAPosition):
                if (df["close"].iat[i] >= virtPositionPrice):
                    virtLongMaximumProfitPrice = df["close"].iat[i]
                else:
                    virtLongMaximumProfitPrice = virtPositionPrice
            elif (virtInsideAPosition):
                if (df["close"].iat[i] >= df["virtLongMaximumProfitPrice"].iat[i - 1]):
                    virtLongMaximumProfitPrice = df["close"].iat[i]
            if (virtLongMaximumProfitPrice is not None):
                virtLongMaximumProfitExcursion = abs(virtLongMaximumProfitPrice - virtPositionPrice)
            else:
                virtLongMaximumProfitExcursion = 0.0

            # Short
            if ((not df["virtInsideAPosition"].iat[i - 1]) and virtInsideAPosition):
                if (df["close"].iat[i] <= virtPositionPrice):
                    virtShortMaximumProfitPrice = df["close"].iat[i]
                else:
                    virtShortMaximumProfitPrice = virtPositionPrice
            elif (virtInsideAPosition):
                if (df["close"].iat[i] <= df["virtShortMaximumProfitPrice"].iat[i - 1]):
                    virtShortMaximumProfitPrice = df["close"].iat[i]
            if (virtShortMaximumProfitPrice is not None):
                virtShortMaximumProfitExcursion = abs(virtShortMaximumProfitPrice - virtPositionPrice)
            else:
                virtShortMaximumProfitExcursion = 0.0

            # maximumProfitExcursion - END

            # maximumLossExcursion - BEGIN

            # Long
            if (resetVirtVariables or ((not df["virtInsideAPosition"].iat[i - 1]) and virtInsideAPosition)):
                if (df["close"].iat[i] <= virtPositionPrice):
                    virtLongMaximumLossPrice = df["close"].iat[i]
                else:
                    virtLongMaximumLossPrice = virtPositionPrice
            elif (virtInsideAPosition):
                if (df["close"].iat[i] <= df["virtLongMaximumLossPrice"].iat[i - 1]):
                    virtLongMaximumLossPrice = df["close"].iat[i]
            if (virtLongMaximumLossPrice is not None):
                virtLongMaximumLossExcursion = abs(virtLongMaximumLossPrice - virtPositionPrice)
            else:
                virtLongMaximumLossExcursion = 0.0

            # Short
            if (resetVirtVariables or ((not df["virtInsideAPosition"].iat[i - 1]) and virtInsideAPosition)):
                if (df["close"].iat[i] >= virtPositionPrice):
                    virtShortMaximumLossPrice = df["close"].iat[i]
                else:
                    virtShortMaximumLossPrice = virtPositionPrice
            elif (virtInsideAPosition):
                if (df["close"].iat[i] >= df["virtShortMaximumLossPrice"].iat[i - 1]):
                    virtShortMaximumLossPrice = df["close"].iat[i]
            if (virtShortMaximumLossPrice is not None):
                virtShortMaximumLossExcursion = abs(virtShortMaximumLossPrice - virtPositionPrice)
            else:
                virtShortMaximumLossExcursion = 0.0

            if (resetVirtVariables):
                resetVirtVariables = False

            # maximumLossExcursion - END

            if (virtInsideAPosition):
                # Long
                virtExpectedTakeProfit = traileringAtrLongTakeProfitPrice + ((i_virtTakeProfitMaximumProfitExcursionRatio * virtLongMaximumProfitExcursion) - (i_virtTakeProfitMaximumLossExcursionRatio * virtLongMaximumLossExcursion))
                if (virtExpectedTakeProfit >= (virtPositionPrice * (1 + i_breakEvenPerOne))):
                    virtLongTakeProfit = virtExpectedTakeProfit
                else:
                    virtLongTakeProfit = df["virtLongTakeProfit"].iat[i - 1]
                # Short
                virtExpectedTakeProfit = traileringAtrShortTakeProfitPrice - ((i_virtTakeProfitMaximumProfitExcursionRatio * virtShortMaximumProfitExcursion) - (i_virtTakeProfitMaximumLossExcursionRatio * virtShortMaximumLossExcursion))
                if (virtExpectedTakeProfit <= (virtPositionPrice * (1 - i_breakEvenPerOne))):
                    virtShortTakeProfit = virtExpectedTakeProfit
                else:
                    virtShortTakeProfit = df["virtShortTakeProfit"].iat[i - 1]

            df["traileringLong"].iat[i] = df["virtInsideAPosition"].iat[i - 1] and virtShortTakeProfitIsReached and (df["close"].iat[i] <= virtPositionPrice) and (virtPositionPrice >= (df["close"].iat[i] * (1 + i_breakEvenPerOne)))
            df["traileringShort"].iat[i] = df["virtInsideAPosition"].iat[i - 1] and virtLongTakeProfitIsReached and (df["close"].iat[i] >= virtPositionPrice) and (virtPositionPrice <= (df["close"].iat[i] * (1 - i_breakEvenPerOne)))

            # Save tmp variables into df
            df["virtLongTakeProfitIsReached"].iat[i] = virtLongTakeProfitIsReached
            df["virtShortTakeProfitIsReached"].iat[i] = virtShortTakeProfitIsReached
            df["virtTraileringBoth"].iat[i] = virtTraileringBoth
            df["virtInsideAPosition"].iat[i] = virtInsideAPosition
            df["virtLongMaximumProfitPrice"].iat[i] = virtLongMaximumProfitPrice
            df["virtShortMaximumProfitPrice"].iat[i] = virtShortMaximumProfitPrice
            df["virtLongMaximumLossPrice"].iat[i] = virtLongMaximumLossPrice
            df["virtShortMaximumLossPrice"].iat[i] = virtShortMaximumLossPrice
            df["virtLongTakeProfit"].iat[i] = virtLongTakeProfit
            df["virtShortTakeProfit"].iat[i] = virtShortTakeProfit

        return ([ df["traileringLong"], df["traileringShort"] ])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        print("DataframeSize: " + str(dataframe.size))

        i_traileringAtrTakeProfitLength = 14 # TODO: Input
        dataframe['traileringAtr'] = ta.ATR(dataframe, timeperiod=i_traileringAtrTakeProfitLength)

        for traileringOffset in range(0,24):
            dataframe["traileringLong" + "-" + str(traileringOffset)], dataframe["traileringShort" + "-" + str(traileringOffset)] = self.populate_trailering_long(dataframe, metadata, offset=traileringOffset)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        dataframe["adx"] = ta.ADX(dataframe)
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]

        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # MFI
        dataframe["mfi"] = ta.MFI(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]

        # Parabolic SAR
        dataframe["sar"] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe["htsine"] = hilbert["sine"]
        dataframe["htleadsine"] = hilbert["leadsine"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                (
                    dataframe["traileringLong"]
                )
                & (  # Guard: tema is raising
                    dataframe["volume"] > 0
                )  # Make sure Volume is not 0
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(dataframe["rsi"], 70))
                & (dataframe["tema"] > dataframe["bb_middleband"])
                & (  # Guard: tema above BB middle
                    dataframe["tema"] < dataframe["tema"].shift(1)
                )
                & (  # Guard: tema is falling
                    dataframe["volume"] > 0
                )  # Make sure Volume is not 0
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(dataframe["rsi"], 70))
                & (dataframe["tema"] > dataframe["bb_middleband"])
                & (  # Guard: tema above BB middle
                    dataframe["tema"] < dataframe["tema"].shift(1)
                )
                & (  # Guard: tema is falling
                    dataframe["volume"] > 0
                )  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(dataframe["rsi"], 30))
                &
                # Guard: tema below BB middle
                (dataframe["tema"] <= dataframe["bb_middleband"])
                & (dataframe["tema"] > dataframe["tema"].shift(1))
                & (  # Guard: tema is raising
                    dataframe["volume"] > 0
                )  # Make sure Volume is not 0
            ),
            "exit_short",
        ] = 1

        return dataframe

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        if (trade.open_date_utc is not None):
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)

            # Obtain pair dataframe.
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

            # Look up trade candle.
            trade_candle = dataframe.loc[dataframe['date'] == trade_date]
            pre_trade_candle = dataframe.shift()
            pre_trade_candle = pre_trade_candle.tail(1).squeeze()

            if ('atr' in pre_trade_candle):
                if (trade.is_short):
                    pre_trade_candle_stoploss = stoploss_from_absolute(current_rate + (pre_trade_candle['atr'] * 1.5), current_rate, is_short=trade.is_short)
                else:
                    pre_trade_candle_stoploss = stoploss_from_absolute(current_rate - (pre_trade_candle['atr'] * 1.5), current_rate, is_short=trade.is_short)
                return pre_trade_candle_stoploss
        return 1.00

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        risk_per_trade = 0.01
        # TODO: Set this 1% risk per trade as a config option

        max_money_loss_per_trade = self.wallets.get_total_stake_amount() * risk_per_trade

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        pre_trade_candle = dataframe.shift().tail(1).squeeze()
        if ('atr' in pre_trade_candle):
            # TODO: 1.5 needs to be a setting
            atr_stoploss_value = pre_trade_candle['atr'] * 1.5
            atr_stoploss_perone = (atr_stoploss_value / self.wallets.get_total_stake_amount())
            atr_stoploss_percentage = atr_stoploss_perone * 100.0

            if (side == 'long'):
                stop_price = current_rate * ( 1 - atr_stoploss_perone ) # loss whatever% of price
                volume_for_buy = max_money_loss_per_trade / (current_rate - stop_price)
                use_money = volume_for_buy * current_rate
            else:
                stop_price = current_rate * ( 1 + atr_stoploss_perone ) # loss whatever% of price
                volume_for_buy = max_money_loss_per_trade / (stop_price - current_rate)
                use_money = volume_for_buy * current_rate
            # TODO What happens when volume is greater than available amount

            #print ("atr_stoploss_percentage: " + str(atr_stoploss_percentage))
            #print ("stop_price: " + str(stop_price))
            #print ("volume_for_buy: " + str(volume_for_buy))
            #print ("use_money: " + str(use_money))

        return use_money
