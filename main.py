# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# general imports
import pandas as pd
import re

# project imports
from agent.agent import BaseAgent
from env.replay import Backtest
from agent.decission_support import DecisionSupport
from agent.order_management import OrderManagementSystem


class Agent(BaseAgent):

    def __init__(self, name, identifies, *args, **kwargs):
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
        super(Agent, self).__init__(name, *args, **kwargs)

        # extract stock list out of identifier list
        self.stock_list = []
        for val in identifies:
            stock = re.split(r'\.(?!\d)', val)[0]
            if len(self.stock_list) == 0 or self.stock_list[-1] != stock: 
                self.stock_list.append(stock)

        # Decision support
        self.support = DecisionSupport(self.stock_list, None)

        # variable parameter set
        self.order_management = OrderManagementSystem(self.stock_list, self)
        self.check_status = None        

        # static parameter set
        self.vol_range = [0.05, 0.07]    # percent of daily vol
        self.time_window = 1            # hours
        self.child_window = 2           # minutes TODO: maybe dynamic per stock

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
                self.update_limit_order(market_id, book_state['TIMESTAMP_UTC'],
                                        book_state)

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
        self.order_management.update_vwap(market_id, trades_state)

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
                or self.check_status + pd.DateOffset(minutes=1) < timestamp:
            self.check_status = timestamp
            print(timestamp)
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
        orders = self.market_interface.get_filtered_orders(market_id,
                                                           status="ACTIVE")
        limit_order = False
        create_new_order = False
        for order in orders:
            if order.limit is not None:
                limit_order = True
                if self.support.update_needed(market_state, order, timestamp):
                    print(
                        'order position: ', 
                        self.order_management.get_order_book_position(
                            self.market_interface.market_state_list[
                                order.market_id], order)
                    )
                    self.market_interface.cancel_order(order)
                    create_new_order = True

        if not limit_order or create_new_order:
            parent_order = \
                self.order_management.get_recent_parent_order(market_id)
            if parent_order.schedule.get_outstanding(timestamp) > 0:
                parent_order.child_orders.append(
                    self.market_interface.submit_order(
                        market_id=market_id,
                        side=parent_order.side,
                        quantity=parent_order.schedule.get_outstanding(
                            timestamp),
                        limit=self.support.determinate_price(market_state,
                                                             parent_order),
                        parent=parent_order
                    )
                )

    def update_schedule(self, parent_order):
        """
        method that allows more flexible scheduling,
        like front and back loading
        """
        pass


if __name__ == "__main__":

    # TODO: SELECT SOURCES. You may delete or comment out the rest.

    identifier_list = [
        # ADIDAS
        # "Adidas.BOOK", "Adidas.TRADES",
        # ALLIANZ
        # "Allianz.BOOK", "Allianz.TRADES",
        # BASF
        # "BASF.BOOK", "BASF.TRADES",
        # Bayer
        # "Bayer.BOOK", "Bayer.TRADES",
        # BMW
        # "BMW.BOOK", "BMW.TRADES",
        # Continental
        # "Continental.BOOK", "Continental.TRADES",
        # Covestro
        # "Covestro.BOOK", "Covestro.TRADES",
        # Daimler
        "Daimler.BOOK", "Daimler.TRADES",
        # Deutsche Bank
        # "DeutscheBank.BOOK", "DeutscheBank.TRADES",
        # DeutscheBörse
        # "DeutscheBörse.BOOK", "DeutscheBörse.TRADES",
    ]

    # TODO: INSTANTIATE AGENT. Please refer to the corresponding file for more 
    # information. 

    agent = Agent(name="test_agent", identifies=identifier_list)

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
    backtest.run_episode_list(identifier_list=identifier_list, episode_list=[
        ("2021-01-04T08:00:00", "2021-01-04T08:15:00", "2021-01-04T09:30:00"),
        #  ...
        ],
    )
