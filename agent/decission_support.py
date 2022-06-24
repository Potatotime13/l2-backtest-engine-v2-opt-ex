# general imports
from __future__ import annotations
from tabnanny import verbose
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
import keras as keras
from keras import backend as K
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
        self.ml_direction = {stock: self.load_ml_model(
            'direction', stock) for stock in stock_list}
        self.ml_intensity = {stock: self.load_ml_model(
            'intensity', stock) for stock in stock_list}
        self.limit_adj_hist = {stock:0 for stock in stock_list}
        self.volume_adj_hist = {stock:0 for stock in stock_list}
        self.current_books = {stock: pd.DataFrame([]) for stock in stock_list}
        self.agent = agent
        self.stock_list = stock_list
        self.vol_dist = pd.read_csv(
            './agent/resources/daily_volume.csv').set_index('Unnamed: 0')
        self.last_level1 = {stock: None for stock in stock_list}
        if agent.level > 0:
            self.minute_vol = {stock: pd.read_csv('./agent/resources/'+stock+'_trade_summary.csv', index_col='Unnamed: 0')
                               for stock in stock_list}
        
        ### Hardcoded stats
        self.ml_model_stats = {
            'Allianz': {'barrier_int':0.54, 'barrier_dir':0.98, 'horizon':2, 'secs_per_signal':20},
            'Adidas': {'barrier_int':0.54, 'barrier_dir':0.98, 'horizon':2, 'secs_per_signal':20},
            'Continental': {'barrier_int':0.54, 'barrier_dir':0.98, 'horizon':2, 'secs_per_signal':20}
        }

    def load_ml_model(self, model_type, market_id):
        path = './agent/resources/'+model_type+'_'+market_id+'.hp5'
        return tf.keras.models.load_model(path)

    def get_volume_distribution(self, stock: str, timestamp: pd.Timestamp,
                                window: int) -> list:
        if self.agent.level > 0:
            hour = timestamp.hour
            min = timestamp.minute
            c_window = self.agent.child_window
            trades = self.minute_vol[stock].copy()
            return self.summary_to_dist(trades, window, c_window, hour, min)
        else:
            return [1]

    def summary_to_dist(self, trades, window, c_window, hour, min):
        data = trades.iloc[:, :22].groupby(['21', '20']).sum()
        data = data.sum(axis=1)
        data = data[(hour, min):(hour+window, min)]
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
        tick_size = self.agent.market_interface.market_state_list[p_order.market_id].tick_size
        side_name = 'Bid' if p_order.side == 'buy' else 'Ask'
        window_left = max(p_order.schedule.get_left_window_time(
            market_state['TIMESTAMP_UTC']).total_seconds()//60+1, 1)
        volume = p_order.schedule.get_outstanding(
            market_state['TIMESTAMP_UTC'])
        volume = int(volume / window_left)

        opt_level, new_score = self.determinate_price(
                                                    market_state,
                                                    p_order, midpoint,
                                                    5, volume)
        opt_level = self.current_best_order(
                                            orders, (opt_level, volume), 
                                            p_order, market_state, 
                                            midpoint, new_score, 5)
        limit_adj = 0
        if p_order.schedule.get_outstanding(
                market_state['TIMESTAMP_UTC']+pd.Timedelta(
                    minutes=p_order.schedule.child_window)) > 0:
            limit_adj, volume_adj = self.get_ml_limit_adj(
                                            p_order, market_state['TIMESTAMP_UTC'], 
                                            tick_size, market_state, 
                                            market_state['L'+str(opt_level)+'-'+side_name+'Price'])
            if limit_adj == 0:
                limit_adj = self.limit_adj_hist[p_order.market_id] * (2/3)
                self.limit_adj_hist[p_order.market_id] = limit_adj
                limit_adj = limit_adj//tick_size*tick_size
                volume_adj = self.volume_adj_hist[p_order.market_id] * (2/3)
                self.volume_adj_hist[p_order.market_id] = volume_adj
            else:
                self.limit_adj_hist[p_order.market_id] = limit_adj
                self.volume_adj_hist[p_order.market_id] = volume_adj
            volume = min(int(volume+volume_adj), p_order.schedule.get_outstanding(
                market_state['TIMESTAMP_UTC']+pd.Timedelta(minutes=p_order.schedule.child_window)))
        if volume > 0:
            strat = [[round(market_state['L'+str(level)+'-'+side_name+'Price']+limit_adj, 2), volume]
                        for level in range(opt_level, min(opt_level+3, 10))]
        else:
            strat = [[round(market_state['L'+str(level)+'-'+side_name+'Price']+limit_adj, 2), volume]
                        for level in range(opt_level, min(opt_level+3, 10))]
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
            mins = timestamp.minute
            if p_order.side == 'buy':
                volume_to_fill = volume + market_state.to_numpy()[2::4]
                prob = self.new_prob(df, 0, volume_to_fill, hour, mins, window_left)
                price = market_state.to_numpy()[1::4]
            else:
                volume_to_fill = volume + market_state.to_numpy()[4::4]
                prob = self.new_prob(df, 10, volume_to_fill, hour, mins, window_left)
                price = market_state.to_numpy()[3::4]
            score = self.get_score(midpoint, price, prob, volume, p_order.side)
            level = np.argmax(score)+1
            return level, np.max(score)

        elif self.agent.level == 2:
            return None

    def current_best_order(self, orders: list[Order], new_order, p_order: ParentOrder, market_state: pd.Series, midpoint, new_score, window_left):
        if len(orders) > 0:
            df = self.minute_vol[p_order.market_id].copy()
            timestamp = market_state['TIMESTAMP_UTC']
            hour = timestamp.hour
            mins = timestamp.minute
            prices = market_state.iloc[np.arange(1, 40, 2)]
            level = []
            vol_queue = []
            volume = []
            if p_order.side == 'buy':
                for order in orders:
                    if order.limit < prices[(new_order[0]-1)*2]:
                        tmp_lev = np.argmin(abs(order.limit-prices))//2
                        if not tmp_lev in level:
                            vol_tmp = self.get_order_book_position(
                                self.agent.market_interface.market_state_list[order.market_id], order)
                            vol_queue.append(vol_tmp+order.quantity)
                            level.append(tmp_lev)
                            volume.append(order.quantity)
            else:
                for order in orders:
                    if order.limit > prices[(new_order[0]-1)*2+1]:
                        tmp_lev = np.argmin(abs(order.limit-prices))//2
                        if not tmp_lev in level:
                            vol_tmp = self.get_order_book_position(
                                self.agent.market_interface.market_state_list[order.market_id], order)
                            vol_queue.append(vol_tmp+order.quantity)
                            level.append(tmp_lev)
                            volume.append(order.quantity)
            if len(level)>0:
                level = np.array(level)
                vq = np.zeros(10)
                vl = np.zeros(10)
                vq[level-1] = np.array(vol_queue)
                vl[level-1] = np.array(volume)
                if p_order.side == 'buy':
                    prob = self.new_prob(df, 0, vq, hour, mins, window_left)
                    score = self.get_score(
                        midpoint, prices[0::2], prob, vl, p_order.side)
                else:
                    prob = self.new_prob(df, 10, vq, hour, mins, window_left)
                    score = self.get_score(
                        midpoint, prices[1::2], prob, vl, p_order.side)
                score = score * (vl > 0)
            else:
                score = 0
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

    def new_prob(self, df, level, vol, hour, mins, window):
        count = 0
        data = df.copy().to_numpy()
        
        def func(day):
            count_len = 0
            sub_count = np.zeros(10)
            for min_ in range(int(max(0,mins-5)), int(min(mins+5+window,60))):
                count_len += 1
                filter_ = np.logical_and(data[:, -1] == day,
                                         np.logical_and(data[:, 21] == hour,
                                         np.logical_and(data[:, 20] >= min_,
                                                        data[:, 20] < min_+window)))
                sub_count += 1 * \
                    (np.sum(data[filter_, level:level+10], axis=0) >= vol)
            return sub_count/(count_len)
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

    def get_book_file(self, timestamp, market_id, sek_window):
        from env.replay import SOURCE_DIRECTORY
        path = SOURCE_DIRECTORY
        path_book = path+'/book/'
        day = str(timestamp.day) if timestamp.day > 9 else '0' + \
            str(timestamp.day)
        date = str(timestamp.year)+'0'+str(timestamp.month)+day
        file_name = 'Book_'+market_id+'_DE_'+date+'_'+date+'.csv.gz'
        if self.current_books[market_id].empty or self.current_books[market_id].index[0].day != timestamp.day:
            data_tmp = pd.read_csv(path_book+file_name, compression='gzip')
            data_tmp['TIMESTAMP_UTC'] = pd.to_datetime(
                data_tmp['TIMESTAMP_UTC'])
            data_tmp = data_tmp.resample(pd.Timedelta(
                seconds=1), on='TIMESTAMP_UTC').mean()
            self.current_books[market_id] = data_tmp.dropna()
        data = self.current_books[market_id]
        ind = data.index.get_loc(data.index[data.index <= timestamp][-1])
        data = data.iloc[ind-sek_window:ind, :]
        data = data.to_numpy()
        return data.copy()

    def get_model_inputs(self, inp_book: np.ndarray, model_type):
        data = inp_book.copy()

        data[:, 0::2] = data[:, 0::2]-np.min(data[:, 0::2])
        data[:, 1::2] = data[:, 1::2]-np.min(data[:, 1::2])
        data[:, 0::2] = data[:, 0::2]/np.max(data[:, 0::2])
        data[:, 1::2] = data[:, 1::2]/np.max(data[:, 1::2])
        en_input = np.zeros((1, 200, 40, 1))
        en_input[0, :, :, 0] = data

        midpoints = np.mean(data[:, [0, 2]], axis=1)
        delta = np.mean(midpoints[0:200]/midpoints[0]-1)
        if model_type == 'direction':
            de_input = np.zeros((1, 1, 2))
            if delta > 0.0:
                de_input[0, 0, 0] = 1
            else:
                de_input[0, 0, 1] = 1
        elif model_type == 'intensity':
            de_input = np.zeros((1, 1, 1))
            decoder_delta = np.std(delta)
            de_input[0, 0, 0] = decoder_delta

        return en_input, de_input

    def get_intensity(self, market_id, timestamp):
        book = self.get_book_file(timestamp, market_id, 200)
        inp1, inp2 = self.get_model_inputs(book, 'intensity')
        pred = self.ml_intensity[market_id].predict((inp1, inp2),verbose=0)
        return pred

    def get_direction(self, market_id, timestamp):
        book = self.get_book_file(timestamp, market_id, 200)
        inp1, inp2 = self.get_model_inputs(book, 'direction')
        pred = self.ml_direction[market_id].predict((inp1, inp2),verbose=0)
        return pred

    def get_ml_limit_adj(self, p_order: ParentOrder, timestamp, tick_size, market_state, limit):
        market_id = p_order.market_id        
        intensity = self.get_intensity(market_id, timestamp)
        side = p_order.side
        limit_adj = 0
        volume_adj = 1
        horizon = self.ml_model_stats[market_id]['horizon']
        if intensity[0, horizon, 0] > self.ml_model_stats[market_id]['barrier_int']:
            direction = self.get_direction(market_id, timestamp)
            best_bid = market_state['L1-BidPrice']
            best_ask = market_state['L1-AskPrice']
            tmp_vol = p_order.volume / (self.agent.time_window*3600/self.ml_model_stats[market_id]['secs_per_signal'])
            if direction[0, horizon, 0]-direction[0, horizon, 1] > self.ml_model_stats[market_id]['barrier_dir']:
                print('up')
                if side=='buy':
                    # act agressive -> marketble order best ask
                    limit_adj = best_ask-limit
                    volume_adj = tmp_vol
                else:
                    # act passive -> increase limit and volume
                    limit_adj = -tick_size*3
                    volume_adj = 0
            elif direction[0, horizon, 0]-direction[0, horizon, 1] < -self.ml_model_stats[market_id]['barrier_dir']:
                print('down')
                if side=='buy':
                    # act passive -> reduce limit and volume
                    limit_adj = -tick_size*3
                    volume_adj = tmp_vol
                else:
                    # act agressive -> marketable order best bid
                    limit_adj = best_bid - limit
                    volume_adj = 10
        return limit_adj, volume_adj
