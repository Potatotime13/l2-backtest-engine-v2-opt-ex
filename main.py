# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# general imports
import tensorflow as tf
import keras as keras
from keras import backend as K
import pandas as pd
import re
import sys

# project imports
from agent.agent import BaseAgent
from env.replay import Backtest
from agent.decission_support import DecisionSupport
from agent.order_management import OrderManagementSystem


class Agent(BaseAgent):

    def __init__(self, name, identifies, rel_vol, t_window, c_window, level, *args, **kwargs):
        """
        Trading agent implementation.

        The backtest iterates over a set of sources and alerts the trading
        agent whenever a source is updated. These updates are conveyed
        through method calls to, all of which you are expected to
        implement yourself:

        - on_quote(self, market_id, book_state)
        - on_trade(self, market_id, trade_state)
        - on_time(self, timestamp, timestamp_next)

        In order to interact with a market, the trading agent needs to use the 
        market_interface instance available at `self.market_interface` that 
        provides the following methods to create and delete orders waiting to 
        be executed against the respective market's order book: 

        - submit_order(self, market_id, side, quantity, limit=None)
        - cancel_order(self, order)

        Besides, the market_interface implements a set of attributes that may 
        be used to monitor trading agent performance: 

        - exposure (per market)
        - pnl_realized (per market)
        - pnl_unrealized (per market)
        - exposure_total
        - pnl_realized_total
        - pnl_unrealized_total
        - exposure_left
        - transaction_costs

        The agent may also access attributes of related class instances, using 
        the container attributes: 

        - order_list -> [<Order>, *]
        - trade_list -> [<Trade>, *]
        - market_state_list -> {<market_id>: <Market>, *}

        For more information, you may list all attributes and methods as well
        as access the docstrings available in the base class using
        `dir(BaseAgent)` and `help(BaseAgent.<method>)`, respectively.

        :param name:
            str, agent name
        """
        self.reset(name, identifies, rel_vol, t_window, c_window, level, *args, **kwargs)
    
    def reset(self, name, identifies, rel_vol, t_window, c_window, level, *args, **kwargs):
        super(Agent, self).__init__(name, *args, **kwargs)

        # static parameter set
        self.relalitve_volume = rel_vol     # percent of daily vol
        self.time_window = t_window         # hours
        self.child_window = c_window        # minutes
        self.level = level                  # agent level

        # extract stock list out of identifier list
        self.stock_list = []
        self.identifies = identifies
        for val in identifies:
            stock = re.split(r'\.(?!\d)', val)[0]
            if len(self.stock_list) == 0 or self.stock_list[-1] != stock:
                self.stock_list.append(stock)

        # Decision support
        self.support = DecisionSupport(self.stock_list, self)

        # variable parameter set
        self.order_management = OrderManagementSystem(self.stock_list, self)
        self.check_status = None

    def on_quote(self, market_id: str, book_state: pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp,
            bid/ask price/quantity for 10 levels
        """
        if self.order_management.initialized:
            if self.order_management.get_recent_parent_order(market_id).active:
                if self.level > 0:
                    self.update_limit_order(market_id, book_state['TIMESTAMP_UTC'],
                                            book_state)
                self.order_management.set_midpoint(market_id, book_state)

    def on_trade(self, market_id: str, trades_state: pd.Series):
        """
        This method is called after a new trade.

        :param market_id:
            str, market identifier
        :param trades_state:
            pd.Series: TIMESTAMP_UTC :pd.Timestamp,
            Price :[float], Volume :[float]
        """

        # calculate vwap for active order
        self.order_management.update_AP_metrics(market_id, trades_state)

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        """ 
        This method is called with every iteration and provides the timestamps
        for both current and next iteration. The given interval may be used to
        submit orders before a specific point in time.

        :param timestamp:
            pd.Timestamp, timestamp recorded
        :param timestamp_next:
            pd.Timestamp, timestamp recorded in next iteration
        """

        # Produce parent orders at the beginning
        self.order_management.initialize_parent_orders(timestamp)

        # check every minute if orders are outdated
        # check if they are in their schedule
        if self.check_status is None \
                or self.check_status + pd.DateOffset(seconds=20) < timestamp:
            self.check_status = timestamp
            self.order_management.check_status_on_time(timestamp)
            parents, scheduling_orders = self.order_management.stay_scheduled(
                timestamp, self.market_interface.market_state_list)
            for parent, scheduling_order in zip(parents, scheduling_orders):
                parent.child_orders.append(
                    self.market_interface.submit_order(
                        market_id=scheduling_order['symbol'],
                        limit=scheduling_order['limit'],
                        side=scheduling_order['side'],
                        quantity=scheduling_order['quantity'],
                        parent=scheduling_order['parent'],
                    )
                )

    def update_limit_order(self, market_id, timestamp, market_state):
        """
        method to send limit orders to the market,
        dependent on the agent strategy
        """
        orders_all = self.market_interface.get_filtered_orders(
            market_id, status="ACTIVE")
        parent_order = self.order_management.get_recent_parent_order(market_id)
        orders = []

        for order in orders_all:
            if order.limit is not None:
                orders.append(order)
        cancel, submit = self.support.update_needed(
            market_state, parent_order, orders)

        for c_order in cancel:
            self.market_interface.cancel_order(c_order)

        for s_order in submit:
            if parent_order.volume_left > 0:
                parent_order.child_orders.append(
                    self.market_interface.submit_order(
                        market_id=market_id,
                        side=parent_order.side,
                        quantity=s_order[1],
                        limit=s_order[0],
                        parent=parent_order
                    )
                )

    def save_stats(self):
        self.order_management.save_stats()


if __name__ == "__main__":

    # TODO: SELECT SOURCES. You may delete or comment out the rest.

    identifier_list = [
        # ADIDAS
        #"Adidas.BOOK", "Adidas.TRADES",
        # ALLIANZ
        "Allianz.BOOK", "Allianz.TRADES",
        # BASF
        #"BASF.BOOK", "BASF.TRADES",
        # Bayer
        #"Bayer.BOOK", "Bayer.TRADES",
        # BMW
        #"BMW.BOOK", "BMW.TRADES",
        # Continental
        #"Continental.BOOK", "Continental.TRADES",
        # Covestro
        #"Covestro.BOOK", "Covestro.TRADES",
        # Daimler
        #"Daimler.BOOK", "Daimler.TRADES",
        # Deutsche Bank
        #"DeutscheBank.BOOK", "DeutscheBank.TRADES",
        # DeutscheBörse
        #"DeutscheBörse.BOOK", "DeutscheBörse.TRADES",
    ]

    # TODO: INSTANTIATE AGENT. Please refer to the corresponding file for more
    # information.

    rel_vol = 0.05
    t_window = 1
    level = 1

    if len(sys.argv) > 1:
        rel_vol = float(sys.argv[1])
        t_window = int(sys.argv[2])
        level = int(sys.argv[3])
        day = int(sys.argv[4])

    agent = Agent(name="test_agent", identifies=identifier_list,
                  rel_vol=rel_vol, t_window=t_window, c_window=10, level=level)

    # TODO: INSTANTIATE BACKTEST. Please refer to the corresponding file for
    # more information.

    backtest = Backtest(
        agent=agent,
    )

    # TODO: RUN BACKTEST. Please refer to the corresponding file for more
    # information.
    '''
    # Option 1: run agent against a series of generated episodes, that is, 
    # generate episodes with the same episode_buffer and episode_length
    backtest.run_episode_generator(identifier_list=identifier_list,
        date_start="2021-01-01", 
        date_end="2021-02-28", 
        episode_interval=30, 
        episode_shuffle=True, 
        episode_buffer=5, 
        episode_length=30, 
        num_episodes=10, 
    )

    # Option 2: run agent against a series of broadcast episodes, that is, 
    # broadcast the same timestamps for every date between date_start and 
    # date_end
    backtest.run_episode_broadcast(identifier_list=identifier_list,
        date_start="2021-01-01", 
        date_end="2021-02-28", 
        time_start_buffer="08:00:00", 
        time_start="08:30:00", 
        time_end="16:30:00",
    )
    '''
    # Option 3: run agent against a series of specified episodes, that is,
    # list a tuple (episode_start_buffer, episode_start, episode_end) for each
    # episode
    if len(sys.argv) > 1:
        backtest.run_episode_list(identifier_list=identifier_list, episode_list=[
            ("2021-02-"+ str(day) +"T08:00:00", "2021-02-"+ str(day) +"T08:15:00", "2021-02-"+ str(day) +"T14:20:00"),
            ("2021-02-"+ str(day+3) +"T10:00:00", "2021-02-"+ str(day+3) +"T10:15:00", "2021-02-"+ str(day+3) +"T16:20:00"),
            ("2021-02-"+ str(day+5) +"T08:00:00", "2021-02-"+ str(day+5) +"T08:15:00", "2021-02-"+ str(day+5) +"T14:20:00"),
            ("2021-02-"+ str(day+7) +"T10:00:00", "2021-02-"+ str(day+7) +"T10:15:00", "2021-02-"+ str(day+7) +"T16:20:00"),
            ("2021-02-"+ str(day+10) +"T08:00:00", "2021-02-"+ str(day+10) +"T08:15:00", "2021-02-"+ str(day+10) +"T14:20:00"),
            ("2021-02-"+ str(day+12) +"T10:00:00", "2021-02-"+ str(day+12) +"T10:15:00", "2021-02-"+ str(day+12) +"T16:20:00"),
            ("2021-02-"+ str(day+14) +"T08:00:00", "2021-02-"+ str(day+14) +"T08:15:00", "2021-02-"+ str(day+14) +"T14:20:00"),
            ("2021-02-"+ str(day+17) +"T10:00:00", "2021-02-"+ str(day+17) +"T10:15:00", "2021-02-"+ str(day+17) +"T16:20:00"),
            ("2021-02-"+ str(day+19) +"T08:00:00", "2021-02-"+ str(day+19) +"T08:15:00", "2021-02-"+ str(day+19) +"T14:20:00"),
            ("2021-02-"+ str(day+21) +"T10:00:00", "2021-02-"+ str(day+21) +"T10:15:00", "2021-02-"+ str(day+21) +"T16:20:00"),
            #  ...
            ],
        )

    else:
        backtest.run_episode_list(identifier_list=identifier_list, episode_list=[
            ("2021-02-01T08:00:00", "2021-02-01T08:15:00", "2021-02-01T09:20:00"),
            #  ...
            ],
        )
