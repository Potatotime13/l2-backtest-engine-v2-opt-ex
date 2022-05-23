# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.agent import BaseAgent
from env.replay import Backtest
from agent.parent_order import Parent_order

import numpy as np
import pandas as pd

class Agent(BaseAgent):

    def __init__(self, name, *args, **kwargs):
        """
        Trading agent implementation.

        The backtest iterates over a set of sources and alerts the trading agent
        whenever a source is updated. These updates are conveyed through method
        calls to, all of which you are expected to implement yourself: 

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

        self.Orders_to_market = []
        self.order_vol_hist = []
        self.stock_hourly_vol = pd.read_csv('./agent/resources/daily_volume.csv').set_index('Unnamed: 0')
        self.stock_mean_vol = self.stock_hourly_vol.mean()
        self.parent_orders = {stock:[] for stock in self.stock_mean_vol.index}
        self.orders_initialized = False
        self.check_status = None        

        ### Params to set
        self.vol_range = [0.05,0.07]
        self.time_window = 1
        self.child_window = 2

    def on_quote(self, market_id:str, book_state:pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/quantity for 10 levels
        """

        # TODO: YOUR IMPLEMENTATION GOES HERE
        
        pass

    def on_trade(self, market_id:str, trades_state:pd.Series):
        """
        This method is called after a new trade.

        :param market_id:
            str, market identifier
        :param trades_state:
            pd.Series: TIMESTAMP_UTC :pd.Timestamp, Price :[float], Volume :[float]
        """

        # calculate vwap for active orders
        # TODO idea store order object as dict with market id
        if self.parent_orders[market_id][-1].active:
            self.parent_orders[market_id][-1].actualize_vwap(trades_state)


    def on_time(self, timestamp:pd.Timestamp, timestamp_next:pd.Timestamp):
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
        if not self.orders_initialized:
            print('create parent orders')
            for stock in self.parent_orders:
                self.parent_orders[stock].append(Parent_order(timestamp, self.vol_range, stock, self.stock_mean_vol[stock], self.time_window, self))
            self.orders_initialized = True

        # check every minute if orders are outdated
        # check if they are in their schedule
        if self.check_status == None:
            self.check_status = timestamp
        elif self.check_status + pd.DateOffset(minutes=1) < timestamp:
            self.check_status = timestamp
            print(timestamp)
            self.set_parent_order_status(timestamp)
            self.stay_scheduled(timestamp)

    def set_parent_order_status(self, timestamp):
        for stock in self.parent_orders:
            if self.parent_orders[stock][-1].active == False and self.parent_orders[stock][-1].time_window[0] < timestamp:
                self.parent_orders[stock][-1].active = True
            elif self.parent_orders[stock][-1].time_window[1] < timestamp:
                self.parent_orders[stock][-1].active = False # generate new parent order
                self.parent_orders[stock].append(Parent_order(timestamp, self.vol_range, stock, self.stock_mean_vol[stock], self.time_window, self))

    def set_schedule(self, parent_order:Parent_order):
        '''
        defines a time schedule per parent order
        '''
        # generate timestamps
        # account for across day time windows
        time_stamps = []
        time_unreached = True
        count = 1
        start_time = parent_order.time_window[0]

        while time_unreached:
            if start_time + pd.DateOffset(minutes=self.child_window * count) < parent_order.MARKET_END:
                offset_time = start_time + pd.DateOffset(minutes=self.child_window * count)
            else:
                offset_time = parent_order.NEXT_MARKET_START + ((start_time + pd.DateOffset(minutes=self.child_window * count)) - parent_order.MARKET_END)
            if   offset_time > parent_order.time_window[1]:
                time_unreached = False
            else:
                time_stamps.append(offset_time)
            count += 1

        # calculate volumes
        volumes = np.ones((len(time_stamps))) * max(parent_order.volume//len(time_stamps),1)
        vol_left = int(parent_order.volume - np.sum(volumes))
        if vol_left > 0:
            volumes[np.arange(vol_left)] += 1
        elif vol_left < 0:
            volumes[np.arange(len(time_stamps)+vol_left,len(time_stamps))] -= 1
        parent_order.schedule = pd.concat([parent_order.schedule, pd.DataFrame({'timestamp':time_stamps,'volume':volumes.tolist()})])


    def update_schedule(self, parent_order):
        pass

    def stay_scheduled(self, timestamp:pd.Timestamp):
        for stock in self.parent_orders:
            if self.parent_orders[stock][-1].active:
                orders_to_fill = self.parent_orders[stock][-1].schedule[self.parent_orders[stock][-1].schedule['timestamp']<timestamp]
                if orders_to_fill['volume'].sum() > 0:
                    self.parent_orders[stock][-1].child_orders.append(self.market_interface.submit_order(
                                                    stock, 
                                                    self.parent_orders[stock][-1].market_side, 
                                                    int(orders_to_fill['volume'].sum()),
                                                    parent=self.parent_orders[stock][-1]
                                                    ))

    def update_limit_order(self, timestamp, market_status):
        pass

if __name__ == "__main__":

    # TODO: SELECT SOURCES. You may delete or comment out the rest.

    identifier_list = [
        # ADIDAS
        "Adidas.BOOK", "Adidas.TRADES",
        # ALLIANZ
        "Allianz.BOOK", "Allianz.TRADES",
        # BASF
        "BASF.BOOK", "BASF.TRADES",
        # Bayer
        "Bayer.BOOK", "Bayer.TRADES",
        # BMW
        "BMW.BOOK", "BMW.TRADES",
        # Continental
        "Continental.BOOK", "Continental.TRADES",
        # Covestro
        "Covestro.BOOK", "Covestro.TRADES",
        # Daimler
        "Daimler.BOOK", "Daimler.TRADES",
        # Deutsche Bank
        "DeutscheBank.BOOK", "DeutscheBank.TRADES",
        # DeutscheBörse
        "DeutscheBörse.BOOK", "DeutscheBörse.TRADES",
    ]

    # TODO: INSTANTIATE AGENT. Please refer to the corresponding file for more 
    # information. 

    agent = Agent(
        name="test_agent",
        # ...
    )

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
    backtest.run_episode_list(identifier_list=identifier_list,
        episode_list=[
            ("2021-01-04T08:00:00", "2021-01-04T08:15:00", "2021-01-04T08:45:00"),
            # ... 
        ],
    )


