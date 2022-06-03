# general imports
from __future__ import annotations
import pandas as pd
import numpy as np
import random as rn
from typing import Union, TYPE_CHECKING

# project imports
from env.market import Order
from data_handling.ai_letsgo import everknowing_entity

# quality of life imports
if TYPE_CHECKING:
    from agent.parent_order import Parent_order


class Decission_support():

    def __init__(self, stock_list, agressivness) -> None:
        self.ml_model_a = None
        self.ml_model_b = None
        self.stock_list = stock_list
        self.agressivness = agressivness
        self.vol_dist = pd.read_csv('./agent/resources/daily_volume.csv').set_index('Unnamed: 0')

    def get_volume_distribution(self, stock:str, timestamp:pd.Timestamp, window:int) -> list:
        hour = timestamp.hour-8
        return self.vol_dist.loc[hour:hour+window,stock].astype(int)
    
    def get_daily_volume(self, stock):
        return self.vol_dist.mean()[stock]

    def update_needed(self, market_state, order:Order, timestamp):
        trigger = False
        level = '3'
        side = 'Bid' if order.side == 'buy' else 'Ask'
        if  market_state['L'+level+'-'+side+'Price'] != order.limit and order.parent.schedule.get_outstanding(timestamp)>0:
            trigger = True
        return trigger

    def determinate_price(self, market_state, order: Union[Order, Parent_order]):
        level = '3'
        side = 'Bid' if order.side == 'buy' else 'Ask'
        return market_state['L'+level+'-'+side+'Price']
    
    def get_alpha(self, side, stock, book_state):
        timestamp = book_state['TIMESTAMP_UTC']
        direction = everknowing_entity(timestamp, stock, 500)
        # edge would be the standard deviation used for the label
        if side == 'sell':
            alpha = direction * -1
        return alpha

    def get_intensity():
        pass

    def get_execution_prob():
        pass
