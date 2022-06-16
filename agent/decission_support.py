# general imports
from __future__ import annotations
import pandas as pd
import numpy as np
# import random as rn
from typing import Union, TYPE_CHECKING

# project imports
from env.market import Order
from data_handling.ai_letsgo import everknowing_entity

# quality of life imports
if TYPE_CHECKING:
    from agent.parent_order import ParentOrder


# TODO Prüfen, ob das einen Fehler gibt.
# Alter name war Decission_support()
class DecisionSupport:

    def __init__(self, stock_list, aggressiveness) -> None:
        self.ml_model_a = None
        self.ml_model_b = None
        self.stock_list = stock_list
        self.aggressiveness = aggressiveness
        self.vol_dist = pd.read_csv('./agent/resources/'
                                    'daily_volume.csv').set_index('Unnamed: 0')

    def get_volume_distribution(self, stock: str, timestamp: pd.Timestamp,
                                window: int) -> list:
        hour = timestamp.hour-8
        return self.vol_dist.loc[hour:hour+window, stock].astype(int)
    
    def get_daily_volume(self, stock):
        return self.vol_dist.mean()[stock]

    @staticmethod
    def update_needed(market_state, order: Order, timestamp) -> bool:
        trigger = False
        level = '3'
        side = 'Bid' if order.side == 'buy' else 'Ask'
        if market_state['L'+level+'-'+side+'Price'] != \
                order.limit and \
                order.parent.schedule.get_outstanding(timestamp) > 0:
            trigger = True
        return trigger

    @staticmethod
    def determinate_price(market_state,
                          order: Union[Order, ParentOrder]):
        level = '3'
        side = 'Bid' if order.side == 'buy' else 'Ask'
        return market_state['L'+level+'-'+side+'Price']

    def determinate_price_TODO(market_state,
                          order: Union[Order, ParentOrder]):
        """
        This function is an empty frame for the limit price calculation using the different
        signals of the decission supports
        """
        price_movement = np.zeros((5))
        execution_probs = np.zeros((20))
        long_term_movement = np.zeros((5))
        limit = 2
        return limit
    
    @staticmethod
    def get_alpha(side, stock, book_state):
        timestamp = book_state['TIMESTAMP_UTC']
        direction = everknowing_entity(timestamp, stock, 500)
        # edge would be the standard deviation used for the label
        if side == 'sell':
            alpha = direction * -1
        # TODO Ist das hier richtig? Alpha war vorher nur im if Teil zugewiesen
        else:
            alpha = direction
        return alpha

    def get_intensity(self):
        pass

    def get_execution_prob(self):
        pass
