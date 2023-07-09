# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from datetime import datetime, timedelta

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

from freqtrade.persistence import Trade

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

    stoploss = -1.00
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

    # ATR Length for the Trailering ATR
    i_traileringAtrTakeProfitLength = 14 # TODO: Probably an INPUT from the config
    i_longVolumeBounceMinimumRatio = 2.5 # TODO: Probably an INPUT from the config
    i_longVolumeBounceMaximumRatio = 7.0 # TODO: Probably an INPUT from the config
    i_shortVolumeBounceMinimumRatio = 2.5 # TODO: Probably an INPUT from the config
    i_shortVolumeBounceMaximumRatio = 7.0 # TODO: Probably an INPUT from the config

    # Custom dictionary for saving custom TakeProfitPrice and other data
    custom_info = {}

    def populate_volume_bounce(self, dataframe: DataFrame, metadata: dict, longVolumeBounceMinimumRatio : float, longVolumeBounceMaximumRatio : float, shortVolumeBounceMinimumRatio : float, shortVolumeBounceMaximumRatio : float) -> DataFrame:

        df = dataframe.copy()

        # Add new columns
        df["volumeBounceLong"] = None
        df["volumeBounceShort"] = None

        for i in range(1, len(df)):
            # Only once init
            if (i == 1):
                volumeBounceRatio = None
                volumeBounceLong = None
                volumeBounceShort = None

            volumeBounceRatio = df["volume"].iat[i] / df["volume"].iat[i - 1]
            volumeBounceLong = (volumeBounceRatio >= longVolumeBounceMinimumRatio) and (volumeBounceRatio <= longVolumeBounceMaximumRatio)
            volumeBounceShort = (volumeBounceRatio >= shortVolumeBounceMinimumRatio) and (volumeBounceRatio <= shortVolumeBounceMaximumRatio)

            # Save tmp variables into df
            df["volumeBounceLong"].iat[i] = volumeBounceLong
            df["volumeBounceShort"].iat[i] = volumeBounceShort

        return ([ df["volumeBounceLong"], df["volumeBounceShort"] ])

    # This function should return traileringLong so that we can later on decide to end into a trend or not
    def populate_trailering_long(self, dataframe: DataFrame, metadata: dict, offset: int, traileringAtrTakeProfitLength : int) -> DataFrame:

        df = dataframe.copy()

        i_trailering_atr_take_profit_length = traileringAtrTakeProfitLength
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

        dataframe['traileringAtr'] = ta.ATR(dataframe, timeperiod=self.i_traileringAtrTakeProfitLength)

        for traileringOffset in range(0,24):
            dataframe["traileringLong" + "-" + str(traileringOffset)], dataframe["traileringShort" + "-" + str(traileringOffset)] = self.populate_trailering_long(dataframe, metadata, offset=traileringOffset, traileringAtrTakeProfitLength=self.i_traileringAtrTakeProfitLength)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        dataframe["volumeBounceLong"], dataframe["volumeBounceShort"] = self.populate_volume_bounce(dataframe, metadata, longVolumeBounceMinimumRatio=self.i_longVolumeBounceMinimumRatio, longVolumeBounceMaximumRatio=self.i_longVolumeBounceMaximumRatio, shortVolumeBounceMinimumRatio=self.i_shortVolumeBounceMinimumRatio, shortVolumeBounceMaximumRatio=self.i_shortVolumeBounceMaximumRatio)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        i_virt_trafilering_minimum_longs = 5 # TODO: Input
        i_virt_trafilering_maximum_longs = 10 # TODO: Input
        dataframe["enter_long"] = None
        for i in range(self.i_traileringAtrTakeProfitLength, len(dataframe)):
            trafilering_longs = 0
            for traileringOffset in range(0,24):
                if (dataframe["traileringLong" + "-" + str(traileringOffset)].iat[i]):
                    trafilering_longs += 1
            if ((trafilering_longs >= i_virt_trafilering_minimum_longs) and (trafilering_longs <= i_virt_trafilering_maximum_longs)):
                if (dataframe["volumeBounceLong"].iat[i]):
                    dataframe["enter_long"].iat[i] = 1

        i_virt_trafilering_minimum_shorts = 5 # TODO: Input
        i_virt_trafilering_maximum_shorts = 10 # TODO: Input
        dataframe["enter_short"] = None
        for i in range(self.i_traileringAtrTakeProfitLength, len(dataframe)):
            trafilering_shorts = 0
            for traileringOffset in range(0,24):
                if (dataframe["traileringShort" + "-" + str(traileringOffset)].iat[i]):
                    trafilering_shorts += 1
            if ((trafilering_shorts >= i_virt_trafilering_minimum_shorts) and (trafilering_shorts <= i_virt_trafilering_maximum_shorts)):
                if (dataframe["volumeBounceShort"].iat[i]):
                    dataframe["enter_short"].iat[i] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exits will be handled either by the custom_stoploss function
        # Or by the custom_exit function

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:

        i_breakEven = 0.20 # TODO: Input
        i_breakEvenPerOne = i_breakEven * 0.01
        i_longTakeProfitRatio = 0.68
        i_shortTakeProfitRatio = 0.68

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.shift()
        last_candle = last_candle.tail(1).squeeze()

        # 1. Check if it is worth trading the trade
        if (side == 'long'):
            longTakeProfit = last_candle['close'] + (i_longTakeProfitRatio * (last_candle['high'] - last_candle['close']))
            if ((longTakeProfit) >= ((last_candle['close']) * (1 + i_breakEvenPerOne))):
                pass
            else:
                return (False)

        if (side == 'short'):
            shortTakeProfit = last_candle['close'] - (i_longTakeProfitRatio * (last_candle['close'] - last_candle['low']))
            if ((shortTakeProfit) <= ((last_candle['close']) * (1 - i_breakEvenPerOne))):
                pass
            else:
                return (False)

        if (side == 'long'):
            newTakeProfit = longTakeProfit
        if (side == 'short'):
            newTakeProfit = shortTakeProfit

        # 2. Update custom values
        if not pair in self.custom_info:
            self.custom_info[pair] = {
                'takeProfitPrice' : float(newTakeProfit)
            }
        else:
            self.custom_info[pair]['takeProfitPrice'] = float(newTakeProfit)

        return (True)

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        if (trade.is_short):
            exitTakeProfit = 0.0
        else:
            exitTakeProfit = 99999999.0

        if not pair in self.custom_info:
            self.custom_info[pair] = {
                'takeProfitPrice' : float(exitTakeProfit)
            }
        else:
            self.custom_info[pair]['takeProfitPrice'] = float(exitTakeProfit)

        return True

    use_exit_signal = True

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        i_trade_maximum_duration_hours = 4 # TODO: Input
        expiredTrade = ((current_time - trade.open_date_utc) >= timedelta(hours=i_trade_maximum_duration_hours))
        if (expiredTrade):
            return ('expired_trade')

        if pair in self.custom_info:
            if ((trade.is_short) and (current_rate <= self.custom_info[pair]['takeProfitPrice'])):
                return ('short_take_profit_reached')

            if ((not (trade.is_short)) and (current_rate >= self.custom_info[pair]['takeProfitPrice'])):
                return ('long_take_profit_reached')

        return (False)

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
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
