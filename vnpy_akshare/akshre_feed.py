import dataclasses
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional,Callable

import pandas as pd
from pytz import timezone

from numpy import ndarray
from pandas import DataFrame

from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData, HistoryRequest
from vnpy.trader.utility import round_to
from vnpy.trader.datafeed import BaseDatafeed

import akshare as ak

INTERVAL_VT2RQ: Dict[Interval, str] = {
    Interval.DAILY: "daily",
    Interval.WEEKLY: "weekly",
    Interval.HOUR: "60min",
    Interval.MINUTE: "1min",
}

INTERVAL_ADJUSTMENT_MAP: Dict[Interval, timedelta] = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta()         # no need to adjust for daily bar
}

CHINA_TZ = timezone("Asia/Shanghai")


def string_to_date(ds: str) -> datetime:
    return datetime.strptime(ds, "%Y-%m-%d")

def string_to_datetime(ds: str) -> datetime:
    return datetime.strptime(ds, "%Y-%m-%d %H:%M:%S")

def date_to_string(dd: datetime) -> str:
    if dd is None:
        return None
    return dd.strftime("%Y%m%d")


@dataclasses.dataclass
class TradeDate:
    start:datetime
    end: datetime
    date_list: List[datetime]


class Country(Enum):
    China = "china"
    US = "us"
    UK = "uk"


country_trade_date: Dict[Country, TradeDate or None] = {
    Country.China: None,
    Country.US: None,
    Country.UK: None,
}


EXCHANGE_COUNTRY = {
    Country.China: {
        Exchange.CFFEX,
        Exchange.SHFE,
        Exchange.CZCE,
        Exchange.DCE,
        Exchange.INE,
        Exchange.SSE,
        Exchange.SZSE,
        Exchange.BSE,
        Exchange.SGE,
        Exchange.WXE,
        Exchange.CFETS,
        Exchange.XBOND,
    },
}


def get_country(exchange: Exchange):
    for country, exchange_set in EXCHANGE_COUNTRY.items():
        if exchange in exchange_set:
            return country

    return None


def get_zh_a_trader_date():
    date_list = list(ak.stock_zh_index_daily_tx("sh000919").date)
    date_list = [string_to_date(d) for d in date_list]
    start = date_list[0]
    end = date_list[-1]
    return TradeDate(start, end, date_list)


def get_trade_date(exchange, start: datetime, end: datetime)-> List[datetime]:
    country = get_country(exchange)
    td = country_trade_date[country]
    if td is None:
        if country == Country.China:
            td = get_zh_a_trader_date()
        country_trade_date[country] = td

    return [d for d in td.date_list if end >= d >= start]


class BaseFeed:
    def query_bar_history(self, req: HistoryRequest) -> pd.DataFrame:

        pass

    def query_tick_history(self, req: HistoryRequest) -> pd.DataFrame:
        pass


class ZhADataFeed(BaseFeed):
    def query_bar_history(self, req: HistoryRequest) -> pd.DataFrame:
        symbol: str = req.symbol
        interval: Interval = req.interval
        start: datetime = req.start
        end: datetime = req.end

        if interval is None:
            interval = Interval.DAILY

        if interval==Interval.MINUTE:
            # 1分钟数据
            df = ak.stock_zh_a_hist_min_em(symbol, date_to_string(start), date_to_string(end), period='1', adjust="qfq")

            df.rename(columns={
                '时间': "datetime",
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'turnover',
            }, inplace=True)
        elif interval==Interval.DAILY:
            # 日频数据
            period = INTERVAL_VT2RQ[interval]
            df = ak.stock_zh_a_hist(symbol, period, date_to_string(start), date_to_string(end), "hfq")

            df.rename(columns={
                '日期': "datetime",
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'turnover',
            }, inplace=True)
        elif interval==Interval.HOUR:
            # 1小时数据
            df = ak.stock_zh_a_hist_min_em(symbol, date_to_string(start), date_to_string(end), period='60', adjust="qfq")

            df.rename(columns={
                '时间': "datetime",
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'turnover',
            }, inplace=True)

        return df

    def query_tick_history(self, req: HistoryRequest) -> pd.DataFrame:
        symbol: str = req.symbol
        start: datetime = req.start
        end: datetime = req.end

        if end is None:
            end = datetime.now()

        date_list = get_trade_date(req.exchange, start, end)
        ret = []
        for d in date_list:
            ret.append(ak.stock_zh_a_tick_163(symbol, date_to_string(d)))

        return pd.concat(ret)


class ZhFutureDataFeed(BaseFeed):
    def query_bar_history(self, req: HistoryRequest) -> pd.DataFrame:
        symbol: str = req.symbol

        start: datetime = req.start
        end: datetime = req.end
        exchange = req.exchange

        df = ak.get_futures_daily(date_to_string(start), date_to_string(end), exchange.value)

        return df

    def query_tick_history(self, req: HistoryRequest) -> pd.DataFrame:
        symbol: str = req.symbol
        start: datetime = req.start
        end: datetime = req.end

        if end is None:
            end = datetime.now()

        date_list = get_trade_date(req.exchange, start, end)
        ret = []
        for d in date_list:
            ret.append(ak.stock_zh_a_tick_163(symbol, date_to_string(d)))

        return pd.concat(ret)


FEEDS = {
    Exchange.CFFEX: ZhFutureDataFeed,
    Exchange.SHFE: ZhFutureDataFeed,
    Exchange.CZCE: ZhFutureDataFeed,
    Exchange.DCE: ZhFutureDataFeed,
    Exchange.INE: ZhFutureDataFeed,

    Exchange.SSE: ZhADataFeed,
    Exchange.SZSE: ZhADataFeed,
    Exchange.BSE: ZhADataFeed,
}


class AKShareDataFeed(BaseDatafeed):
    """AKData数据服务接口"""

    def __init__(self):
        self.inited = False

    def init(self,output: Callable = print) -> bool:
        self.inited = True
        return True

    def convert_df_to_bar(self, req: HistoryRequest, df: DataFrame) -> Optional[List[BarData]]:

        data: List[BarData] = []

        interval: Interval = req.interval if req.interval is not None else Interval.DAILY

        # 为了将时间戳（K线结束时点）转换为VeighNa时间戳（K线开始时点）
        adjustment: timedelta = INTERVAL_ADJUSTMENT_MAP[interval]

        if df is not None:
            # 填充NaN为0
            df.fillna(0, inplace=True)

            for row in df.itertuples():
                if interval==Interval.MINUTE:
                    dt: datetime = string_to_datetime(row.datetime)
                elif interval==Interval.DAILY:
                    dt: datetime = string_to_date(row.datetime)
                elif interval==Interval.HOUR:
                    dt: datetime = string_to_datetime(row.datetime)
                dt: datetime = dt - adjustment
                dt: datetime = CHINA_TZ.localize(dt)

                bar: BarData = BarData(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=interval,
                    datetime=dt,
                    open_price=round_to(row.open, 0.000001),
                    high_price=round_to(row.high, 0.000001),
                    low_price=round_to(row.low, 0.000001),
                    close_price=round_to(row.close, 0.000001),
                    volume=row.volume,
                    turnover=row.turnover,
                    open_interest=getattr(row, "open_interest", 0),
                    gateway_name="AK"
                )

                data.append(bar)

        return data

    def convert_df_to_tick(self, df: DataFrame) -> Optional[List[TickData]]:
        return df

    def query_bar_history(self, req: HistoryRequest,output: Callable = print) -> Optional[List[BarData]]:
        """查询K线数据"""
        if not self.inited:
            n: bool = self.init()
            if not n:
                return []

        exchange: Exchange = req.exchange
        if exchange not in FEEDS:
            return []

        clazz = FEEDS[exchange]
        df = clazz().query_bar_history(req)

        return self.convert_df_to_bar(req, df)

    def query_tick_history(self, req: HistoryRequest) -> Optional[List[TickData]]:
        exchange: Exchange = req.exchange
        if exchange not in FEEDS:
            return []

        clazz = FEEDS[exchange]

        df = clazz().query_tick_history(req)
        return self.convert_df_to_tick(df)
