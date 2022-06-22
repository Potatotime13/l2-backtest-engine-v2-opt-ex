# general imports
from __future__ import annotations
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
# import random as rn
from typing import Union, TYPE_CHECKING, List

# project imports
from env.market import Order

# quality of life imports
if TYPE_CHECKING:
    from agent.parent_order import ParentOrder
    from main import Agent
    from env.market import MarketState


class DecisionSupport:

    def __init__(self, stock_list, agent: Agent) -> None:
        self.ml_direction = {stock:None for stock in stock_list}
        self.ml_intensity = {stock:None for stock in stock_list}
        self.agent = agent
        self.stock_list = stock_list
        self.vol_dist = pd.read_csv(
            './agent/resources/daily_volume.csv').set_index('Unnamed: 0')
        self.last_level1 = {stock: None for stock in stock_list}
        if agent.level > 0:
            self.minute_vol = {stock: pd.read_csv('./agent/resources/'+stock+'_trade_summary.csv', index_col='Unnamed: 0')
                               for stock in stock_list}

    def get_volume_distribution(self, stock: str, timestamp: pd.Timestamp,
                                window: int) -> list:
        if self.agent.level > 0:
            hour = timestamp.hour
            min = timestamp.minute
            c_window = self.agent.child_window
            trades = self.minute_vol[stock].copy()
            return self.summary_to_dist(trades,window,c_window,hour,min)
        else:
            return [1]

    def summary_to_dist(self, trades, window, c_window, hour, min):
        data = trades.iloc[:,:22].groupby(['21','20']).sum()
        data = data.sum(axis=1)
        data = data[(hour,min):(hour+window,min)]
        print(data)
        w = 2*c_window
        dist = []
        for i in range(len(data)//w):
            dist.append(int(np.sum(data[w*i:w*(i+1)])))
        return dist

    def get_daily_volume(self, stock):
        return self.vol_dist.mean()[stock]

    # evaluate strategy

    def update_needed(self, market_state, order: ParentOrder, orders: list[Order]) -> bool:
        if self.agent.level == 1:
            side_name = 'Bid' if order.side == 'buy' else 'Ask'
            act_level1 = market_state['L1-'+side_name+'Price']
            if self.last_level1[order.market_id] == None or self.last_level1[order.market_id] != act_level1:
                self.last_level1[order.market_id] = act_level1
                new_strat = self.get_strat(market_state, order, orders)
                cancel, submit = self.match_strat(orders, new_strat, order.side)
                return cancel, submit
            else:
                return [], []
        elif self.agent.level == 2:
            return None

    def get_strat(self, market_state: pd.Series, p_order: ParentOrder, orders: list[Order]) -> np.array:
        midpoint = np.mean(market_state.loc[['L1-BidPrice', 'L1-AskPrice']])
        side_name = 'Bid' if p_order.side == 'buy' else 'Ask'
        window_left = max(p_order.schedule.get_left_window_time(market_state['TIMESTAMP_UTC']).total_seconds()//60+1, 1)
        volume = p_order.schedule.get_outstanding(market_state['TIMESTAMP_UTC'])
        volume = int(volume / window_left)
        opt_level, new_score = self.determinate_price(market_state, p_order, midpoint, 1, volume)
        opt_level = self.current_best_order(orders, (opt_level, volume), p_order, market_state, midpoint, new_score, 1)

        if volume > 0:
            strat = [[market_state['L'+str(level)+'-'+side_name+'Price'], volume]
                     for level in range(opt_level, min(opt_level+3,10))]
        else:
            opt_level += 2
            strat = [[market_state['L'+str(level)+'-'+side_name+'Price'], volume]
                     for level in range(opt_level, min(opt_level+3,10))]
        return np.array(strat)

    def match_strat(self, orders: List[Order], strat: np.array, side: str) -> tuple(list, list):
        cancelations = []
        if side == 'buy':
            for order in orders:
                if order.limit > np.max(strat[:, 0]):
                    cancelations.append(order)
                else:
                    ind = np.argmin(abs(strat[:, 0]-order.limit))
                    strat[ind, 1] -= order.quantity
            fullfill = strat[:, 1] > 0
        else:
            for order in orders:
                if order.limit < np.max(strat[:, 0]):
                    cancelations.append(order)
                else:
                    ind = np.argmin(abs(strat[:, 0]-order.limit))
                    strat[ind, 1] -= order.quantity
            fullfill = strat[:, 1] > 0
        return cancelations, strat[fullfill, :].tolist()

    def determinate_price(self, market_state, p_order: ParentOrder, midpoint: float, window_left, volume):
        if self.agent.level == 1:
            df = self.minute_vol[p_order.market_id].copy()
            timestamp = market_state['TIMESTAMP_UTC']
            hour = timestamp.hour

            
            if p_order.side == 'buy':
                volume_to_fill = volume + market_state.to_numpy()[2::4]
                prob = self.new_prob(df, 0, volume_to_fill, hour, window_left)
                price = market_state.to_numpy()[1::4]
            else:
                volume_to_fill = volume + market_state.to_numpy()[4::4]
                prob = self.new_prob(df, 10, volume_to_fill, hour, window_left)
                price = market_state.to_numpy()[3::4]
            score = self.get_score(midpoint, price, prob, volume, p_order.side)
            level = np.argmax(score)+1
            return level, np.max(score)

        elif self.agent.level == 2:
            return None

    def current_best_order(self, orders: list[Order], new_order, p_order: ParentOrder, market_state: pd.Series, midpoint, new_score, window_left):
        if len(orders)>0:
            df = self.minute_vol[p_order.market_id].copy()
            timestamp = market_state['TIMESTAMP_UTC']
            hour = timestamp.hour
            prices = market_state.iloc[np.arange(1, 40, 2)]
            level = []
            vol_queue = []
            volume = []
            if p_order.side == 'buy':
                for order in orders:
                    if order.limit<prices[(new_order[0]-1)*2]:
                        tmp_lev = np.argmin(abs(order.limit-prices))//2
                        if not tmp_lev in level:
                            vol_tmp = self.get_order_book_position(self.agent.market_interface.market_state_list[order.market_id], order)
                            vol_queue.append(vol_tmp+order.quantity)
                            level.append(tmp_lev)
                            volume.append(order.quantity)
            else:
                for order in orders:
                    if order.limit>prices[(new_order[0]-1)*2+1]:
                        tmp_lev = np.argmin(abs(order.limit-prices))//2
                        if not tmp_lev in level:
                            vol_tmp = self.get_order_book_position(self.agent.market_interface.market_state_list[order.market_id], order)
                            vol_queue.append(vol_tmp+order.quantity)
                            level.append(tmp_lev)
                            volume.append(order.quantity)

            level = np.array(level)
            vq = np.zeros(10)
            vl = np.zeros(10)            
            vq[level-1] = np.array(vol_queue)
            vl[level-1] = np.array(volume)
            if p_order.side == 'buy':
                prob = self.new_prob(df, 0, vq, hour, window_left)
                score = self.get_score(midpoint, prices[0::2], prob, vl, p_order.side)
            else:
                prob = self.new_prob(df, 10, vq, hour, window_left)
                score = self.get_score(midpoint, prices[1::2], prob, vl, p_order.side)
            score = score * (vl>0)
            if np.max(score) > new_score:
                ind = np.argmax(score)
                return ind+1
            else:
                return new_order[0]
        else:
            return new_order[0]

    def get_score(self, midpoint: float, price: float, prob: float, volume, side):
        if side == 'buy':
            return (midpoint-price) * prob
        else:
            return (price-midpoint) * prob

    def get_prob(self, df, count, hour, minute, window, level, vol):
        data = df.copy().to_numpy()
        filter_ = np.logical_and(data[:, 20] == hour, np.logical_and(
            data[:, 21] >= minute, data[:, 21] < minute+window))
        dist = data[filter_, level]
        return np.sum(dist >= vol)/count

    def new_prob(self, df, level, vol, hour, window):
        count = 0
        data = df.copy().to_numpy()

        def func(day):
            sub_count = np.zeros(10)
            for min_ in range(60-int(window)):
                filter_ = np.logical_and(data[:, -1] == day,
                                         np.logical_and(data[:, 21] == hour,
                                         np.logical_and(data[:, 20] >= min_,
                                                        data[:, 20] < min_+window)))
                sub_count += 1 * \
                    (np.sum(data[filter_, level:level+10], axis=0) >= vol)
            return sub_count/(60-window)
        out = Parallel(n_jobs=6)(delayed(func)(d)
                                 for d in np.unique(data[:, -1]))
        count = np.sum(np.array(out), axis=0)
        return count/len(np.unique(data[:, -1]))

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
                return position
        else:
            return None

    # ai support functions

    def determinate_price_TODO(market_state, order: Union[Order, ParentOrder]):
        """
        This function is an empty frame for the limit price calculation using the different
        signals of the decission supports
        """
        price_movement = np.zeros((5))
        execution_probs = np.zeros((20))
        long_term_movement = np.zeros((5))
        limit = 2
        return limit

    def get_intensity(self):
        pass

    def get_direction(self):
        pass
