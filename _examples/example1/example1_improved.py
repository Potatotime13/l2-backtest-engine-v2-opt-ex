# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.agent import BaseAgent
from env.replay import Backtest
import datetime
import numpy as np
import pandas as pd


class SimpleAgent(BaseAgent):

    def __init__(self, name: str,barrier_open: float,
                    barrier_close: float, stop_loss: float, quantity: int,
                    ):
        """
        Trading agent implementation example. Improved version.

        :param name:
            str, agent name
        :param barrier_open:
            float, if barrier is hit, agent opens a position
        :param barrier_close:
            float, if barrier is hit, agent closes a position with profit
        :param stop_loss:
            float, if stop loss is hit, agent closes a position with loss
        :param quantity:
            int, defines the amount of shares that are traded
        """
        super(SimpleAgent, self).__init__(name)

        # static attributes from arguments
        self.quantity = quantity
        self.barrier_open = barrier_open
        self.barrier_close = barrier_close
        self.stop_loss = stop_loss

        # further static attributes
        self.start_time = datetime.time(8, 15)
        self.end_time = datetime.time(16, 15)
        self.market_interface.transaction_cost_factor = 0

        # dynamic attributes
        self.max_price = {}  # dict that captures max price of each market
        self.min_price = {}  # dict that captures min price of each market
        self.trading_phase = False  # indicates whether algo is ready to trade

    def on_quote(self, market_id:str, book_state:pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/size for 10 levels
        """
        if self.trading_phase:

            # Calculate midpoint
            midpoint = (book_state['L1-BidPrice'] + book_state['L1-AskPrice']) / 2
            midpoint = np.round(midpoint, 4)
            # alternative: midpoint = self.markets[market_id].mid_point

            # Update min and max prices of episode
            if market_id not in self.max_price.keys():
                self.max_price[market_id] = midpoint
            else:
                self.max_price[market_id] = max(midpoint,
                                                self.max_price[market_id])
            if market_id not in self.min_price.keys():
                self.min_price[market_id] = midpoint
            else:
                self.min_price[market_id] = min(midpoint,
                                                self.min_price[market_id])

            # Calculate current drawdown & upswing
            drawdown = min((book_state['L1-AskPrice']/
                            self.max_price[market_id] - 1), 0)
            upswing = max((book_state['L1-BidPrice']/
                            self.min_price[market_id] - 1), 0)

            # Conditions for opening a position
            # (1) no position is open
            # (2) no order is waiting for execution
            if not self.market_interface.exposure[market_id] and \
                not self.market_interface.get_filtered_orders(
                    market_id, status="ACTIVE"):

                # Submit market buy order if drawdown exceeds barrier
                if abs(drawdown) > self.barrier_open:
                    self.market_interface.submit_order(
                        market_id, "buy", self.quantity)

                # Submit market sell order if upswing exceeds barrier
                elif abs(upswing) > self.barrier_open:
                    self.market_interface.submit_order(
                        market_id, "sell", self.quantity)

            # Conditions for Closing
            # (1) Long position exists for this market
            if self.market_interface.exposure[market_id] > 0:
                # Get price of trade
                exec_price = self.market_interface.get_filtered_trades(
                    market_id)[-1].price
                trade_profit = book_state['L1-BidPrice']/exec_price - 1

                # Close long position if
                # (1) barrier_close is hit
                # (2) stop loss is hit
                if trade_profit > self.barrier_close or \
                   trade_profit * -1 > self.stop_loss:
                    self.market_interface.submit_order(
                        market_id, "sell", self.quantity)
                    self.max_price[market_id] = midpoint  # reset current max

            # Conditions to for Closing
            # (2) Short position exists for this market
            elif self.market_interface.exposure[market_id] < 0:
                # Get price of trade
                exec_price = self.market_interface.get_filtered_trades(
                    market_id)[-1].price
                trade_profit = -1 * (book_state['L1-AskPrice']/exec_price - 1)

                # Close short position if
                # (1) barrier_close is hit
                # (2) stop loss is hit
                if trade_profit > self.barrier_close or \
                   trade_profit * -1 > self.stop_loss:
                    self.market_interface.submit_order(
                        market_id, "buy", self.quantity)
                    self.min_price[market_id] = midpoint  # reset current min


    def on_trade(self, market_id:str, trades_state:pd.Series):
        """
        This method is called after a new trade.

        :param market_id:
            str, market identifier
        :param trades_state:
            pd.Series, including timestamp, price, quantity
        """
        pass

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

        trading_time = timestamp.time() > self.start_time and \
                        timestamp.time() < self.end_time

        # Enter trading phase if
        # (1) current time in defined trading_time
        # (2) trading_phase is False up to now
        if trading_time and not self.trading_phase:
            print('Algo is now able to trade...')
            self.trading_phase = True

        # Close trading phase if
        # (1) current time not in defined trading_time
        # (2) trading_phase is True up to now
        elif not trading_time and self.trading_phase:

            for market_id in self.market_interface.market_state_list.keys():

                # cancel active orders for this market
                [self.market_interface.cancel_order(order) for order in
                self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]

                # close positions for this market
                if self.market_interface.exposure[market_id] > 0:
                    self.market_interface.submit_order(
                        market_id, "sell", self.quantity)
                if self.market_interface.exposure[market_id] < 0:
                    self.market_interface.submit_order(
                        market_id, "buy", self.quantity)

            self.trading_phase = False


if __name__ == "__main__":

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

    agent = SimpleAgent(
        name="simpleAgent2",
        barrier_open=0.003,
        barrier_close=0.003,
        stop_loss=0.003,
        quantity=100,
    )
    
    backtest = Backtest(
        agent=agent, 
    )
    """
    # Option 1: run agent against a series of generated episodes, that is, 
    # generate episodes with the same episode_buffer and episode_length
    backtest.run_episode_generator(identifier_list=identifier_list,
        date_start="2016-01-06",
        date_end="2016-01-07",
        episode_interval=30, 
        episode_shuffle=True, 
        episode_buffer=5, 
        episode_length=30, 
        num_episodes=2,
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
    """
    # Option 3: run agent against a series of specified episodes, that is, 
    # list a tuple (episode_start_buffer, episode_start, episode_end) for each 
    # episode
    backtest.run_episode_list(identifier_list=identifier_list,
        episode_list=[
            ("2021-01-03T08:00:00", "2021-01-03T08:15:00", "2021-01-03T08:25:00"),
            ("2021-01-04T08:00:00", "2021-01-04T08:15:00", "2021-01-04T08:25:00"),

            ("2021-02-18T12:00:00", "2021-02-18T12:15:00", "2021-02-18T12:45:00"),
            ("2021-03-23T14:00:00", "2021-03-23T14:15:00", "2021-03-23T14:45:00"),

            # ... 
        ],
    )


