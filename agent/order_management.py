# general imports
from __future__ import annotations
import pandas as pd
from typing import Tuple, TYPE_CHECKING

# project imports
from agent.parent_order import Parent_order

# quality of life imports
if TYPE_CHECKING:
    from main import Agent


class Order_Management_System():

    def __init__(self, stock_list: list, agent:Agent, ) -> None:
        self.stock_list = stock_list
        self.agent = agent
        self.parent_orders = {stock:[] for stock in self.stock_list}
        self.initialized = False
        self.support = agent.support

    def check_status_on_time(self, timestamp: pd.Timestamp) -> None:
        '''
        method to check at a moment in time if orders are active depending on their time_window

        :param timestamp:
            pd.Timestamp, timestamp of the moment the method is called
        '''
        for stock in self.parent_orders:
            tmp_order = self.get_recent_parent_order(stock)
            if tmp_order.active == False and tmp_order.time_window[0] < timestamp and tmp_order.volume_left > 0:
                tmp_order.active = True
            elif tmp_order.time_window[1] < timestamp and tmp_order.volume_left <= 0:
                tmp_order.active = False
                self.generate_parent_order(stock, timestamp)

    def initialize_parent_orders(self, timestamp: pd.Timestamp) -> None:
        '''
        method to initialize parent orders at the begining of the simulation

        :param timestamp:
            pd.Timestamp, timestamp of the moment the method is called
        '''
        if not self.initialized:
            self.initialized = True
            for market_id in self.stock_list:
                self.generate_parent_order(market_id, timestamp)

    def order_filled(self, parent: Parent_order, timestamp: pd.Timestamp):
        self.generate_parent_order(parent.symbol, timestamp)

    def generate_parent_order(self, market_id: str, timestamp: pd.Timestamp) -> None:
        '''
        method to generate a parent order, called in case the previous order is finished

        :param market_id:
            str, symbol name of the stock

        :param timestamp:
            pd.Timestamp, timestamp of the moment the method is called
        '''
        volume_pattern = self.support.get_volume_distribution(market_id,timestamp,self.agent.time_window)
        volume_avg = self.support.get_daily_volume(market_id)
        self.parent_orders[market_id].append(
            Parent_order(timestamp, 
                        self.agent.vol_range,
                        market_id, 
                        volume_avg, 
                        self.agent.time_window, 
                        self, 
                        self.agent.child_window, 
                        volume_pattern))

    def update_vwap(self, market_id: str, trades_state:pd.Series) -> None:
        '''
        method to update the vwap of a parent order of the corresponding stock

        :param market_id:
            str, name of the stock

        :param timestamp:
            pd.Timestamp, timestamp of the moment the method is called
        '''
        if self.parent_orders[market_id][-1].active:
            self.parent_orders[market_id][-1].actualize_vwap(trades_state)

    def get_recent_parent_order(self, market_id: str) -> Parent_order:
        '''
        method to get the latest parent order of a symbol

        :param market_id:
            str, name of the stock
        :return Parent_order:
            Paretn_order
        '''
        return self.parent_orders[market_id][-1]

    def stay_scheduled(self, timestamp: pd.Timestamp, market_states) -> Tuple[list[Parent_order], list[dict]]:
        '''
        method to get the latest parent order of a symbol

        :param timestamp:
            pd.Timestamp, timestamp of the moment the method is called
        :param market_states:
            market_interface.statelist
        :return parent_list:
            list[Paretn_order]
        :return scheduling_list:
            list[dict], list of dicts containing limit order infos
        '''
        parent_list = []
        scheduling_list = []
        for market_id in self.stock_list:
            market_state = market_states[market_id]
            parent = self.get_recent_parent_order(market_id)
            scheduling_order = parent.schedule.stay_scheduled(market_state, timestamp)
            if not scheduling_order is None:
                parent_list.append(parent)
                scheduling_list.append(scheduling_order)
        return parent_list, scheduling_list

    def get_combined_vwap(self):
        pass