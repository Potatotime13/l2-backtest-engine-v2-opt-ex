# general imports
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

# project imports
from env.market import MarketState

# quality of life imports
if TYPE_CHECKING:
    from agent.parent_order import ParentOrder


class Schedule:
    """
    A Schedule object is a necessary part of each parent order and manages
    the child order
    timing of it
    """

    def __init__(self, parent_order: ParentOrder, child_window: int,
                 pattern: list) -> None:
        """
        method that defines a time schedule per parent order which is called
        by the order itself

        :param parent_order:
            Parent_order, object of the parent order which asks for a schedule
        :param child_window:
            int, length of a one point in the schedule to the next in minutes
        :param pattern:
            list[number], list of numbers which sketch a pattern for the order
            volumes
            example: [1] -> linear pattern, [1,2] -> increasing pattern
        """
        # generate timestamps
        # account for across day time windows
        time_stamps = []
        time_unreached = True
        count = 1
        self.start_time = parent_order.time_window[0]
        self.end_time = parent_order.time_window[1]
        self.child_window = child_window
        self.parent_order = parent_order
        self.pattern = pattern

        while time_unreached:
            if self.start_time \
                    + pd.DateOffset(minutes=self.child_window * count) \
                    < parent_order.MARKET_END:
                offset_time = self.start_time \
                    + pd.DateOffset(
                        minutes=self.child_window * count)
            else:
                offset_time = parent_order.NEXT_MARKET_START \
                    + ((self.start_time
                        + pd.DateOffset(
                            minutes=self.child_window * count))
                       - parent_order.MARKET_END)
            if offset_time > self.end_time:
                time_unreached = False
            else:
                time_stamps.append(offset_time)
            count += 1

        # meta schedule
        diff = int((self.end_time-self.start_time).seconds / 3600)
        self.meta_schedule_time = [self.start_time+pd.DateOffset(hours=i)
                                   for i in range(1, diff)]
        self.meta_schedule_time.append(self.end_time-pd.DateOffset(minutes=5))
        self.meta_schedule_time.append(self.end_time-pd.DateOffset(minutes=max(child_window//2,1)))
        self.meta_schedule = pd.DataFrame(
            {'time': self.meta_schedule_time,
             'Done': [False for _ in range(len(self.meta_schedule_time))]}
        )
        self.meta_schedule = self.meta_schedule.set_index('time')

        # pattern to distribution
        x = np.linspace(0, 1, len(pattern))
        y = np.array(pattern)
        x_inter = np.linspace(0, 1, len(time_stamps))
        y_inter = np.interp(x_inter, x, y)
        percentages = y_inter/np.sum(y_inter)

        # calculate volumes
        volumes = np.maximum(parent_order.volume*percentages, 1).astype(int)
        vol_left = int(parent_order.volume - np.sum(volumes))
        if vol_left > 0:
            volumes[np.arange(vol_left)] += 1
        elif vol_left < 0:
            volumes[np.arange(len(time_stamps)%abs(vol_left),
                            len(time_stamps),
                            len(time_stamps)//abs(vol_left))] -= 1 
        self.scheduling = pd.DataFrame({'timestamp': time_stamps,
                                        'volume': volumes.tolist()})

    def stay_scheduled(self, level, market_state: MarketState,
                       timestamp: pd.Timestamp) -> dict:
        """
        method to send market orders to stay on the planed schedule

        :params timestamp:
            pd.Timestamp, moment of method call usually called out of on_time
        """

        if np.all(self.meta_schedule[self.meta_schedule.index < timestamp]) and level > 0:
            if self.parent_order.side == 'buy':
                limit = None # market_state.best_ask
            else:
                limit = None # market_state.best_bid
        else:
            limit = None
            self.meta_schedule[self.meta_schedule.index < timestamp] = True

        output = None
        if self.parent_order.active:
            orders_to_fill = self.scheduling[self.scheduling['timestamp']<= timestamp]
            if orders_to_fill['volume'].sum() > 0:
                output = {
                    'symbol': self.parent_order.market_id,
                    'limit': limit,
                    'side': self.parent_order.side,
                    'quantity': int(orders_to_fill['volume'].sum()),
                    'parent': self.parent_order
                }
        return output

    def reduce_schedule(self, reduce_volume: float) -> None:
        """
        method to actualize the schedule after an order was executed

        :param reduce_volume:
            float, number of shares traded
        """
        for index, vol in self.scheduling['volume'].iteritems():
            if 0 < vol <= reduce_volume:
                reduce_volume -= vol
                self.scheduling.loc[index, 'volume'] = 0
            elif vol > reduce_volume:
                self.scheduling.loc[index, 'volume'] -= reduce_volume
                reduce_volume = 0

    def get_outstanding(self, timestamp) -> float:
        if self.scheduling[self.scheduling['timestamp']
                           < timestamp].index.empty:
            ind = 0
        else:
            ind = max(self.scheduling[self.scheduling['timestamp']
                                      < timestamp].index)+1
        return self.scheduling.loc[:ind, 'volume'].sum()
    
    def get_left_window_time(self, timestamp:pd.Timestamp) -> pd.Timedelta:
        next_times = self.scheduling.loc[self.scheduling['timestamp']>timestamp,'timestamp'].to_list()
        if len(next_times)<1:
            return pd.Timedelta(value=10, unit='seconds')
        else:
            return next_times[0]-timestamp
