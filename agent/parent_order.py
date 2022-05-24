import random as rn
import numpy as np
import pandas as pd
from env.market import Trade

class Schedule():
    '''
    A Schedule object is a nessesary part of each parant order and manages the child order
    timing of it
    '''
    def __init__(self, parent_order, child_window:int, pattern:list) -> None:
        '''
        mehtod that defines a time schedule per parent order which is called by the order itself

        :param parent_order:
            Parent_order, object of the parent order which asks for a schedule
        :param child_window:
            int, length of a one point in the schedule to the next in minutes
        :param pattern:
            list[number], list of numbers which scetch a pattern for the order volumes
            example: [1] -> linear pattern, [1,2] -> increasing pattern
        '''
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
            if self.start_time + pd.DateOffset(minutes=self.child_window * count) < parent_order.MARKET_END:
                offset_time = self.start_time + pd.DateOffset(minutes=self.child_window * count)
            else:
                offset_time = parent_order.NEXT_MARKET_START + ((self.start_time + pd.DateOffset(minutes=self.child_window * count)) - parent_order.MARKET_END)
            if   offset_time > self.end_time:
                time_unreached = False
            else:
                time_stamps.append(offset_time)
            count += 1

        # pattern to distribution
        x = np.linspace(0,1,len(pattern))
        y = np.array(pattern)
        x_inter = np.linspace(0,1,len(time_stamps))
        y_inter = np.interp(x_inter,x,y)
        percentages = y_inter/np.sum(y_inter)

        # calculate volumes
        volumes = np.maximum(parent_order.volume*percentages,1).astype(int)
        vol_left = int(parent_order.volume - np.sum(volumes))
        if vol_left > 0:
            volumes[np.arange(vol_left)] += 1
        elif vol_left < 0:
            volumes[np.arange(len(time_stamps)+vol_left,len(time_stamps))] -= 1
        self.scheduling = pd.DataFrame({'timestamp':time_stamps,'volume':volumes.tolist()})

    def stay_scheduled(self, timestamp:pd.Timestamp):
        '''
        method to send market orders to stay on the planed schedule

        :params timestamp:
            pd.Timestamp, moment of method call usually called out of on_time
        '''
        if self.parent_order.active:
            orders_to_fill = self.scheduling[self.scheduling['timestamp']<timestamp]
            if orders_to_fill['volume'].sum() > 0:
                self.parent_order.child_orders.append(self.parent_order.agent.market_interface.submit_order(
                                                self.parent_order.symbol, 
                                                self.parent_order.market_side, 
                                                int(orders_to_fill['volume'].sum()),
                                                parent=self.parent_order
                                                ))

    def reduce_schedule(self, reduce_volume:float):
        '''
        method to actualize the schedule after an order was executed

        :param reduce_volume:
            float, number of shares traded
        '''
        for index, vol in self.scheduling['volume'].iteritems():
            if 0 < vol <= reduce_volume:
                reduce_volume -= vol
                self.scheduling.loc[index,'volume'] = 0
            elif vol > reduce_volume:
                self.scheduling.loc[index,'volume'] -= reduce_volume
                reduce_volume = 0

class Parent_order():
    '''
    This is the Parent Order class wich is used to simulate incoming parent orders 
    and manege corresponding child orders
    '''
    def __init__(self, ts:pd.Timestamp, volume_range:list, stock:str, avg_vol:float, time_window_length:int, agent, child_window:int, pattern:list) -> None:
        '''
        initializiation of an Parent order object

        :param ts:
            pd.Timestamp, timestamp of the moment the initialization is called
        :param volume_range:
            list, containing percentual values of daily volume representing the interval from which the order size is sampled
        :param stock:
            string, stock symbol of the stock ordered
        :param avg_vol:
            float, average volume trade volume of the stock
        :param time_window_length:
            int, time in which the order should be filled in houres
        :param agent:
            agent, agent object
        :param child_window:
            int, minutes of initial schedule window
        :param pattern:
            list, pattern of initial schedule
        '''
        # parameters
        self.MARKET_END = pd.Timestamp(ts.year, ts.month, ts.day, 16, 30)
        self.NEXT_MARKET_START = pd.Timestamp(ts.year, ts.month, ts.day+1, 8)
        self.symbol = stock
        self.volume = int(volume_range[0] + rn.random() * (volume_range[1]-volume_range[0]) * avg_vol)

        ### WARNING static start at 8:xx for testing
        start_h = 8 #rn.randint(8, self.MARKET_END.hour-1)
        start_m = rn.randint(15,59)
        ############################################

        start_time = pd.Timestamp(ts.year, ts.month, ts.day, start_h, start_m)
        if start_time + pd.DateOffset(minutes=time_window_length*60)>self.MARKET_END:
            end_time = self.NEXT_MARKET_START + (start_time + pd.DateOffset(minutes=time_window_length*60) - self.MARKET_END)
            self.time_window = [start_time, end_time]
        else:
            self.time_window = [start_time, pd.Timestamp(ts.year, ts.month, ts.day, start_h+time_window_length,start_m)]
        self.market_side = rn.choice(['buy','sell'])
        self.agent = agent        
        self.schedule = Schedule(self, child_window, pattern)

        # order status
        self.active = False
        self.child_orders = []        
        self.volume_left = self.volume
        self.vwap = None
        self.market_vwap = None
        self.market_volume = None

        print(self.volume,self.symbol,self.market_side,self.time_window[0])
    
    def execution(self, exec_trade:Trade):
        '''
        method which is called when a child order is (partly) executed to track the current standing of the parent order

        :param exec_trade:
            Trade, trade object representing the trade who triggerd the method
        '''
        volume_old = self.volume-self.volume_left
        self.volume_left -= exec_trade.quantity
        if self.volume_left <= 0:
            self.active = False
            child_window = self.schedule.child_window
            pattern = self.schedule.pattern
            self.agent.parent_orders[self.symbol].append(Parent_order(exec_trade.timestamp, self.agent.vol_range, self.symbol, self.agent.stock_mean_vol[self.symbol], self.agent.time_window, self.agent, child_window, pattern))
        self.schedule.reduce_schedule(exec_trade.quantity)
        if self.vwap == None:
            self.vwap = exec_trade.price
        else:
            self.vwap =  (volume_old * self.vwap + exec_trade.quantity * exec_trade.price) / (volume_old+exec_trade.quantity)
        if not self.market_vwap is None:
            if self.market_side == 'sell':
                print(self.symbol,' standing : ',self.vwap/self.market_vwap)
            else:
                print(self.symbol,' standing : ',self.market_vwap/self.vwap)

    def actualize_vwap(self, trades_state:pd.Series):
        '''
        method to track the market vwap in the same time window as the parent order

        :param trades_state:
            pd.Series, pandas series containing the latest trade history
            format: TIMESTAMP_UTC :pd.Timestamp, Price :[float], Volume :[float]
        '''
        trade_volume = np.sum(trades_state['Volume'])        
        trade_vwap = np.sum(np.array(trades_state['Price']) * np.array(trades_state['Volume'])) / trade_volume
        if self.market_vwap is None:
            self.market_volume = trade_volume
            self.market_vwap = trade_vwap
        else:
            self.market_vwap = (self.market_volume * self.market_vwap + trade_volume * trade_vwap) / (self.market_volume+trade_volume)
            self.market_volume += trade_volume