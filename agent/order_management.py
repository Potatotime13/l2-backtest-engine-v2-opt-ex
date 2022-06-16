# general imports
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, TYPE_CHECKING

# project imports
from agent.parent_order import ParentOrder

# quality of life imports
if TYPE_CHECKING:
    from main import Agent
    from env.market import Order
    from env.market import MarketState


# TODO Alter Name war Order_Management_System()
class OrderManagementSystem:

    def __init__(self, stock_list: list, agent: Agent, ) -> None:
        self.stock_list = stock_list
        self.agent = agent
        self.parent_orders = {stock: [] for stock in self.stock_list}
        self.order_stats = pd.DataFrame({'key':[],'time':[],'volume':[],'price':[],'midpoint':[],'vwap':[],'market_vwap':[],'twap':[],'market_twap':[]})
        self.stats_path = './agent/save_games/mapping.csv'
        self.save_path = './agent/save_games/'
        self.stats_mapping = pd.read_csv(self.stats_path, index_col=None)
        self.initialized = False
        self.support = agent.support

    def check_status_on_time(self, timestamp: pd.Timestamp) -> None:
        """
        method to check at a moment in time if orders are active depending on
        their time_window

        :param timestamp:
            pd.Timestamp, timestamp of the moment the method is called
        """
        for stock in self.parent_orders:
            tmp_order = self.get_recent_parent_order(stock)
            if tmp_order.active is False \
                    and tmp_order.time_window[0] < timestamp \
                    and tmp_order.volume_left > 0:
                tmp_order.active = True
            elif tmp_order.time_window[1] < timestamp \
                    and tmp_order.volume_left <= 0:
                tmp_order.active = True
                #self.generate_parent_order(stock, timestamp)

    def initialize_parent_orders(self, timestamp: pd.Timestamp) -> None:
        """
        method to initialize parent orders at the beginning of the simulation

        :param timestamp:
            pd.Timestamp, timestamp of the moment the method is called
        """
        if not self.initialized:
            self.initialized = True
            for market_id in self.stock_list:
                self.generate_parent_order(market_id, timestamp)

    def order_filled(self, parent: ParentOrder, timestamp: pd.Timestamp):
        # save stats: append to order_stats dataframe; append row in mapping table first for new key
        key = self.stats_mapping['key'].to_list()[-1]+1
        self.stats_mapping = pd.concat([self.stats_mapping, pd.DataFrame({
            'key':[key],
            'stock':[parent.symbol],
            'window':[self.agent.time_window],
            'childwindow':[self.agent.child_window],
            'agent':[self.agent.level],
            'volume':[self.agent.vol_range],
            'abs_vol':[parent.volume],
            'testing':[1]
        })])
        tmp_df = parent.stats.copy()
        tmp_df['key'] = [key for _ in range(len(tmp_df))]
        self.order_stats = pd.concat([self.order_stats,tmp_df])
        orders = self.agent.market_interface.get_filtered_orders(parent.symbol, status="ACTIVE")
        for order in orders:
            if order.limit is not None:
                self.agent.market_interface.cancel_order(order)
        print('parant order completed')
        self.generate_parent_order(parent.symbol, timestamp)

    def save_stats(self):
        self.stats_mapping.to_csv(self.stats_path, index=False)
        k_start = self.order_stats['key'].min()
        k_end = self.order_stats['key'].max()
        self.order_stats.to_csv(self.save_path+'save_games_'+str(k_start)+'_'+str(k_end)+'.csv', index=False)

    @staticmethod
    def get_order_book_position(market_state: MarketState, order: Order):
        """
        method wich returns relative queue position of an order on a
        limit level

        :param market_state:
            MarketState, returned by the market_interface
        :param order:
            Order, order created through submit_order
        """
        if order.status == 'ACTIVE':
            limit = order.limit
            time = order.timestamp
            side = order.side
            buy_dict, sell_dict = market_state.state
            position = 0
            volume = 0
            last = True
            if side == 'buy':
                try:
                    level = buy_dict[limit]
                except KeyError:
                    return 0.0
            else:
                try:
                    level = sell_dict[limit]
                except KeyError:
                    return 0.0
            for order_ in level:
                volume += order_[1]
                if order_[0] > time and last:
                    last = False
                    position = volume
            if last:
                position = volume
            if volume == 0:
                return 0.0
            else:
                return position/volume
        else:
            return None

    def generate_parent_order(self, market_id: str,
                              timestamp: pd.Timestamp) -> None:
        """
        method to generate a parent order, called in case the previous order
        is finished

        :param market_id:
            str, symbol name of the stock

        :param timestamp:
            pd.Timestamp, timestamp of the moment the method is called
        """
        volume_pattern = \
            self.support.get_volume_distribution(market_id, timestamp,
                                                 self.agent.time_window)
        volume_avg = self.support.get_daily_volume(market_id)
        self.parent_orders[market_id].append(
            ParentOrder(timestamp,
                        self.agent.vol_range,
                        market_id,
                        volume_avg,
                        self.agent.time_window,
                        self,
                        self.agent.child_window,
                        volume_pattern))

    def update_vwap(self, market_id: str, trades_state: pd.Series) -> None:
        """
        method to update the vwap of a parent order of the corresponding stock

        :param market_id:
            str, name of the stock

        :param trades_state:
            pd.Timestamp, timestamp of the moment the method is called
        """
        if self.parent_orders[market_id][-1].active:
            self.parent_orders[market_id][-1].actualize_market_vwap(
                trades_state)

    def set_midpoint(self, market_id, book_state: pd.Series):
        self.parent_orders[market_id][-1].set_last_midpoint(
            np.mean(book_state.loc[['L1-BidPrice', 'L1-AskPrice']]))

    def get_recent_parent_order(self, market_id: str) -> ParentOrder:
        """
        method to get the latest parent order of a symbol

        :param market_id:
            str, name of the stock
        :return Parent_order:
            Parent_order
        """
        return self.parent_orders[market_id][-1]

    def stay_scheduled(self, timestamp: pd.Timestamp, market_states) -> \
            Tuple[list[ParentOrder], list[dict]]:
        """
        method to get the latest parent order of a symbol

        :param timestamp:
            pd.Timestamp, timestamp of the moment the method is called
        :param market_states:
            market_interface.statelist
        :return parent_list:
            list[Parent_order]
        :return scheduling_list:
            list[dict], list of dicts containing limit order infos
        """
        parent_list = []
        scheduling_list = []
        for market_id in self.stock_list:
            market_state = market_states[market_id]
            parent = self.get_recent_parent_order(market_id)
            scheduling_order = parent.schedule.stay_scheduled(market_state,
                                                              timestamp)
            if scheduling_order is not None:
                parent_list.append(parent)
                scheduling_list.append(scheduling_order)
        return parent_list, scheduling_list

    def get_combined_vwap(self):
        pass
