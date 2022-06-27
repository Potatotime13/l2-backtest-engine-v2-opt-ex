# general imports
from __future__ import annotations
import random as rn
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

# project imports
from env.market import Trade
from agent.schedule import Schedule

# quality of life imports
if TYPE_CHECKING:
    from agent.order_management import OrderManagementSystem


class ParentOrder:
    """
    This is the Parent Order class wich is used to simulate incoming
    parent orders
    and manege corresponding child orders
    """

    def __init__(self, ts: pd.Timestamp, relative_volume: float, stock: str,
                 avg_vol: float, time_window_length: int,
                 order_management: OrderManagementSystem, child_window: int,
                 pattern: list) -> None:
        """
        initialization of an Parent order object

        :param ts:
            pd.Timestamp, timestamp of the moment the initialization is called
        :param volume_range:
            list, containing percentual values of daily volume representing
            the interval from which the order size is sampled
        :param stock:
            string, stock symbol of the stock ordered
        :param avg_vol:
            float, average volume trade volume of the stock
        :param time_window_length:
            int, time in which the order should be filled in houres
        :param order_management:
            order management object
        :param child_window:
            int, minutes of initial schedule window
        :param pattern:
            list, pattern of initial schedule
        """
        # parameters
        self.MARKET_END = pd.Timestamp(ts.year, ts.month, ts.day, 16, 30)
        self.NEXT_MARKET_START = pd.Timestamp(ts.year, ts.month, ts.day+1, 8)
        self.market_id = stock
        self.volume = int(relative_volume * avg_vol)
        self.order_management = order_management

        # WARNING static start at 8:xx for testing
        start_h = max(8, ts.hour)  # rn.randint(8, self.MARKET_END.hour-1)
        start_m = min(ts.minute+2, 59)  # rn.randint(15, 59)
        ############################################

        self.start_time = pd.Timestamp(
            ts.year, ts.month, ts.day, start_h, start_m)
        if self.start_time \
                + pd.DateOffset(minutes=time_window_length*60) \
                > self.MARKET_END:
            end_time = self.NEXT_MARKET_START \
                + (self.start_time
                   + pd.DateOffset(minutes=time_window_length*60)
                   - self.MARKET_END)
            self.time_window = [self.start_time, end_time]
        else:
            self.time_window = [self.start_time,
                                pd.Timestamp(ts.year, ts.month, ts.day,
                                             start_h + time_window_length,
                                             start_m)]
        self.side = rn.choice(['buy', 'sell'])
        self.schedule = Schedule(self, child_window, pattern)

        # order status
        self.active = False
        self.child_orders = []
        self.volume_left = self.volume
        self.vwap = None
        self.stats = pd.DataFrame({'time': [], 'volume': [], 'price': [], 'midpoint': [
        ], 'vwap': [], 'market_vwap': [], 'twap': [], 'market_twap': []})
        self.last_midpoint = None
        self.market_vwap = None
        self.market_volume = None
        # TODO
        self.twap = None
        self.market_twap = None

        print(self.volume, self.market_id, self.side, self.time_window[0])

    def execution(self, exec_trade: Trade):
        """
        method which is called when a child order is (partly) executed to
        track the current standing of the parent order

        :param exec_trade:
            Trade, trade object representing the trade who triggered the method
        """
        volume_old = self.volume-self.volume_left
        self.volume_left -= exec_trade.quantity
        self.schedule.reduce_schedule(exec_trade.quantity)
        if self.vwap is None:
            self.vwap = exec_trade.price
        else:
            self.vwap = (volume_old * self.vwap + exec_trade.quantity
                         * exec_trade.price) / (volume_old+exec_trade.quantity)
        if self.twap is None:
            self.twap_time = exec_trade.timestamp
            self.twap = exec_trade.price
            self.time_sum = self.twap_time - self.start_time
        else:
            delta_t = exec_trade.timestamp - self.twap_time
            self.twap = ((self.twap * self.time_sum.microseconds +
                          exec_trade.price * delta_t.microseconds) /
                         (self.time_sum.microseconds+delta_t.microseconds))
            self.time_sum += delta_t
            self.twap_time = exec_trade.timestamp
        if self.market_vwap is not None:
            if self.side == 'sell':
                print(self.market_id, ' standing : ',
                      self.vwap/self.market_vwap)
            else:
                print(self.market_id, ' standing : ',
                      self.market_vwap/self.vwap)

        # update stats
        self.stats = pd.concat([self.stats,
                                pd.DataFrame({
                                    'time': [exec_trade.timestamp],
                                    'volume':[exec_trade.quantity],
                                    'price':[exec_trade.price],
                                    'midpoint':[self.last_midpoint],
                                    'vwap':[self.vwap],
                                    'market_vwap':[self.market_vwap],
                                    'twap':[self.twap],
                                    'market_twap':[self.market_twap]
                                })])

        # order filled
        if self.volume_left <= 0:
            self.active = False
            self.order_management.order_filled(self, exec_trade.timestamp)

    def set_last_midpoint(self, midpoint):
        """
        soos
        """
        self.last_midpoint = midpoint

    def actualize_market_vwap(self, trades_state: pd.Series):
        """
        method to track the market vwap in the same time window
        as the parent order

        :param trades_state:
            pd.Series, pandas series containing the latest trade history
            format: TIMESTAMP_UTC :pd.Timestamp,
                    Price :[float], Volume :[float]
        """
        trade_volume = np.sum(trades_state['Volume'])
        trade_vwap = np.sum(np.array(trades_state['Price'])
                            * np.array(trades_state['Volume'])) / trade_volume
        if self.market_vwap is None:
            self.market_volume = trade_volume
            self.market_vwap = trade_vwap
        else:
            self.market_vwap = (self.market_volume * self.market_vwap
                                + trade_volume * trade_vwap) \
                / (self.market_volume+trade_volume)
            self.market_volume += trade_volume

    def actualize_market_twap(self, trade_state: pd.Series):
        trade_price = np.mean(trade_state['Price'])
        if self.market_twap is None:
            self.market_twap_time = trade_state['TIMESTAMP_UTC']
            self.market_twap = trade_price
            self.market_time_sum = self.market_twap_time - self.start_time
        else:
            delta_t = trade_state['TIMESTAMP_UTC'] - self.market_twap_time
            self.market_twap = ((self.market_twap * self.market_time_sum.microseconds +
                                 trade_price * delta_t.microseconds) /
                                (self.market_time_sum.microseconds +
                                 delta_t.microseconds))
            self.market_time_sum += delta_t
            self.market_twap_time = trade_state['TIMESTAMP_UTC']
