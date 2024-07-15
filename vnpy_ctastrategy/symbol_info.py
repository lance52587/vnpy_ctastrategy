import vnpy
from vnpy.trader.constant import Exchange
from vnpy.trader.utility import extract_vt_symbol

version = vnpy.__version__
if version >= '3.0.0':
    from vnpy_ctastrategy import CtaTemplate
else:
    from vnpy.app.cta_strategy import CtaTemplate

all_sizes = {'a': 10, 'ag': 15, 'al': 5, 'ao': 20, 'AP': 10, 'au': 1000, 'b': 10, 'bb': 500, 'bc': 5, 'br': 5, 'bu': 10,
             'c': 10, 'CF': 5, 'CJ': 5, 'cs': 10, 'cu': 5, 'CY': 5, 'eb': 5, 'ec': 50, 'eg': 10, 'fb': 10, 'FG': 20,
             'fu': 10, 'hc': 10, 'i': 100, 'j': 100, 'jd': 10, 'jm': 60, 'JR': 20, 'l': 5, 'lc': 1, 'lh': 16, 'LR': 20,
             'lu': 10, 'm': 10, 'MA': 10, 'ni': 1, 'nr': 10, 'OI': 10, 'p': 10, 'pb': 5, 'PF': 5, 'pg': 20, 'PK': 5,
             'PM': 50, 'pp': 5, 'PX': 5, 'rb': 10, 'RI': 20, 'RM': 10, 'rr': 10, 'RS': 10, 'ru': 10, 'SA': 20,
             'sc': 1000, 'sctas': 1000, 'SF': 5, 'SH': 30, 'si': 5, 'SM': 5, 'sn': 1, 'sp': 10, 'SR': 10, 'ss': 5,
             'TA': 5, 'UR': 20, 'v': 5, 'WH': 20, 'wr': 10, 'y': 10, 'ZC': 100, 'zn': 5, 'IC': 200, 'IF': 300,
             'IH': 300, 'IM': 200, 'T': 10000, 'TF': 10000, 'TS': 20000, 'TL': 10000, 'BTCUSDT': 1, 'ETHUSDT': 1}

all_priceticks = {'a': 1.0, 'ag': 1.0, 'al': 5.0, 'ao': 1.0, 'AP': 1.0, 'au': 0.02, 'b': 1.0, 'bb': 0.05, 'bc': 10.0,
                  'br': 5.0, 'bu': 1.0, 'c': 1.0, 'CF': 5.0, 'CJ': 5.0, 'cs': 1.0, 'cu': 10.0, 'CY': 5.0, 'eb': 1.0,
                  'ec': 0.1, 'eg': 1.0, 'fb': 0.5, 'FG': 1.0, 'fu': 1.0, 'hc': 1.0, 'i': 0.5, 'j': 0.5, 'jd': 1.0,
                  'jm': 0.5, 'JR': 1.0, 'l': 1.0, 'lc': 50.0, 'lh': 5.0, 'LR': 1.0, 'lu': 1.0, 'm': 1.0, 'MA': 1.0,
                  'ni': 10.0, 'nr': 5.0, 'OI': 1.0, 'p': 2.0, 'pb': 5.0, 'PF': 2.0, 'pg': 1.0, 'PK': 2.0, 'PM': 1.0,
                  'pp': 1.0, 'PX': 2.0, 'rb': 1.0, 'RI': 1.0, 'RM': 1.0, 'rr': 1.0, 'RS': 1.0, 'ru': 5.0, 'SA': 1.0,
                  'sc': 0.1, 'sctas': 0.1, 'SF': 2.0, 'SH': 1.0, 'si': 5.0, 'SM': 2.0, 'sn': 10.0, 'sp': 2.0, 'SR': 1.0,
                  'ss': 5.0, 'TA': 2.0, 'UR': 1.0, 'v': 1.0, 'WH': 1.0, 'wr': 1.0, 'y': 2.0, 'ZC': 0.2, 'zn': 5.0,
                  'IC': 0.2, 'IF': 0.2, 'IH': 0.2, 'IM': 0.2, 'T': 0.005, 'TF': 0.005, 'TS': 0.002, 'TL': 0.01,
                  'BTCUSDT': 0.1, 'ETHUSDT': 0.01}

all_symbol_pres = {
    'DCE': ['a', 'b', 'bb', 'c', 'cs', 'eb', 'eg', 'fb', 'i', 'j', 'jd', 'jm', 'l', 'lh', 'm', 'p', 'pg', 'pp', 'rr',
            'v', 'y'],
    'SHFE': ['ag', 'al', 'ao', 'au', 'br', 'bu', 'cu', 'fu', 'hc', 'ni', 'pb', 'rb', 'ru', 'sn', 'sp', 'ss', 'wr',
             'zn'],
    'CZCE': ['AP', 'CF', 'CJ', 'CY', 'FG', 'JR', 'LR', 'MA', 'OI', 'PF', 'PK', 'PM', 'PX', 'RI', 'RM', 'RS', 'SA', 'SF',
             'SH', 'SM', 'SR', 'TA', 'UR', 'WH', 'ZC'], 'INE': ['bc', 'ec', 'lu', 'nr', 'sc', 'sctas'],
    'GFEX': ['lc', 'si'], 'CFFEX': ['IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TL', 'TS'], 'BINANCE': ['BTCUSDT', 'ETHUSDT'],}

all_symbols = [symbol for exchange in all_symbol_pres.values() for symbol in exchange]

illiquid_symbol = ['JR', 'LR', 'PM', 'RI', 'RS', 'WH', 'WT', 'T', 'TF', 'TS', 'bb', 'fb', 'wr', 'CY', 'NR', 'rr',
                   'TL', 'ZC']

dbsymbols = ['a', 'ag', 'al', 'AP', 'au', 'b', 'bb', 'bc', 'bu', 'c', 'CF', 'CJ', 'cs', 'cu', 'CY', 'eb', 'eg', 'fb',
             'FG', 'fu', 'hc', 'i', 'IC', 'IF', 'IH', 'IM', 'j', 'jd', 'jm', 'JR', 'l', 'lc', 'lh', 'LR', 'lu', 'm',
             'MA', 'ni', 'nr', 'OI', 'p', 'pb', 'PF', 'pg', 'PK', 'PM', 'pp', 'rb', 'RI', 'RM', 'rr', 'RS', 'ru', 'SA',
             'sc', 'SF', 'si', 'SM', 'sn', 'sp', 'SR', 'ss', 'T', 'TA', 'TF', 'TL', 'TS', 'UR', 'v', 'WH', 'wr', 'y', 'ZC',
             'zn', 'ao', 'br', 'ec', 'PX', 'SH', 'BTCUSDT', 'ETHUSDT']

ind_symbol = {'black': ['ZC', 'jm', 'j', 'i', 'rb', 'hc', 'SM', 'SF', 'wr'],
              'agrictural': ['a', 'b', 'PK', 'm', 'y', 'p', 'RM', 'OI', 'SR', 'CF', 'CY', 'jd', 'lh', 'c', 'cs', 'AP',
                             'CJ', 'rr', 'RI', 'JR', 'LR', 'WH', 'PM', 'RS'],
              'energy': ['eg', 'sc', 'bu', 'sp', 'SA', 'FG', 'l', 'MA', 'pp', 'eb', 'nr', 'UR', 'ru', 'fu', 'bb', 'fb',
                         'v', 'PF', 'TA', 'lu', 'pg', 'br', 'PX', 'SH', 'ec'],
              'colored': ['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'bc', 'au', 'ag', 'ao', 'si', 'lc'],
              'stkind': ['IC', 'IF', 'IH', 'IM'], 'debt': ['TS', 'TF', 'T', 'TL'], }

ind_symbol_reverse = {symbol: k for k, v in ind_symbol.items() for symbol in v}

liquid_ind_symbol = {i: [j for j in ind_symbol[i] if j in dbsymbols and j not in illiquid_symbol] for i in
                     ind_symbol.keys()}

trading_hours = {
    0: ['AP', 'CJ', 'JR', 'LR', 'PK', 'PM', 'RI', 'RS', 'SF', 'SM', 'UR', 'WH', 'bb', 'ec', 'fb', 'jd', 'lc', 'lh',
        'si', 'wr'],  # 0：白盘品种
    1: ['CF', 'CY', 'FG', 'MA', 'OI', 'PF', 'PX', 'RM', 'SA', 'SH', 'SR', 'TA', 'ZC', 'a', 'b', 'br', 'bu', 'c', 'cs',
        'eb', 'eg', 'fu', 'hc', 'i', 'j', 'jm', 'l', 'lu', 'm', 'nr', 'p', 'pg', 'pp', 'rb', 'rr', 'ru', 'sp', 'v',
        'y'],  # 1：夜盘到23点品种
    2: ['al', 'ao', 'bc', 'cu', 'ni', 'pb', 'sn', 'sn', 'ss', 'zn'],  # 2：夜盘到凌晨1点品种
    3: ['ag', 'au', 'sc'],  # 3：夜盘到凌晨2点30分品种
    4: ['T', 'TF', 'TL', 'TS'],  # 4：9:30-11:30,13:00-15:15
    5: ['IC', 'IF', 'IH', 'IM'],  # 5：9:30-11:30,13:00-15:00
    6: ['sctas'],  # 6：每周一至周五的开市集合竞价阶段, 21:00 - 02:30（+1）,9:00 - 10:15
}

all_symbol_pres_rev = {symbol: k for k, v in all_symbol_pres.items() for symbol in v}


def extract_symbol_pre(vt_symbol, with_exchange=True, case_sensitive=True):
    '''
    :param vt_symbol: 'rb2010' or 'rb2010.SHFE'
    :return: rb，2010, SHFE or rb, 2010
    '''
    if '.' in vt_symbol:
        symbol, exchange = extract_vt_symbol(vt_symbol)
    else:
        symbol = vt_symbol
        exchange = None

    for i, s in enumerate(symbol):
        if s.isdigit():
            pre = symbol[:i]
            if not case_sensitive:
                pre = [s for s in all_symbols if s.lower() == pre.lower()][0]
            num = symbol[i:]

            if with_exchange:
                if exchange:
                    return pre, num, exchange
                else:
                    return pre, num, Exchange(all_symbol_pres_rev.get(pre, 'LOCAL'))
            else:
                return pre, num


# trading_hours反向映射
trading_hours_reverse = {symbol: k for k, v in trading_hours.items() for symbol in v}

# 给定symbol（例如AP, rb）, 返回是在交易时间中(True,False)【只根据交易时间和周六日过滤，不考虑假日TDAYS情况】
trading_periods = [trading_hours[0] + trading_hours[1] + trading_hours[2] + trading_hours[3],
                   trading_hours[1] + trading_hours[2] + trading_hours[3], trading_hours[2] + trading_hours[3]]


def is_trading(symbol, symbol_datetime):
    ret = False
    time_str = symbol_datetime.strftime("%H%M%S")
    time_weekday = symbol_datetime.weekday()

    exchange = all_symbol_pres_rev.get(symbol, None)

    # 根据品种过滤非交易时间的tick
    if exchange in ('CFFEX',) and time_weekday in (0, 1, 2, 3, 4):
        # 如果tick时间在9点半到11点半，13点到15点之间
        if "093000" <= time_str < "113000" or "130000" <= time_str < "150000":
            ret = True
        elif "150000" <= time_str < "151500":
            if symbol in trading_hours[4]:
                ret = True
    elif exchange in ('SHFE', 'CZCE', 'DCE', 'INE', 'GFEX'):
        if "090000" <= time_str < "101500" or "103000" <= time_str < "113000" or "133000" <= time_str < "150000":
            if time_weekday in (0, 1, 2, 3, 4):
                # if symbol in trading_hours[0] + trading_hours[1] + trading_hours[2] + trading_hours[3]:
                if symbol in trading_periods[0]:
                    ret = True
        elif "210000" <= time_str < "230000":
            if time_weekday in (0, 1, 2, 3, 4):
                # if symbol in trading_hours[1] + trading_hours[2] + trading_hours[3]:
                if symbol in trading_periods[1]:
                    ret = True
        elif "230000" <= time_str <= "235959":
            if time_weekday in (0, 1, 2, 3, 4):
                # if symbol in trading_hours[2] + trading_hours[3]:
                if symbol in trading_periods[2]:
                    ret = True
        elif "000000" <= time_str < "010000":
            if time_weekday in (1, 2, 3, 4, 5):
                # if symbol in trading_hours[2] + trading_hours[3]:
                if symbol in trading_periods[2]:
                    ret = True
        elif "010000" <= time_str < "023000":
            if time_weekday in (1, 2, 3, 4, 5):
                if symbol in trading_hours[3]:
                    ret = True
    return ret


symbol_min_order = {'PM': 10, 'ZC': 4}
contract_min_order = {'UR2401': 4, 'UR401': 4, 'UR2402': 4, 'UR402': 4, 'UR2403': 4, 'UR403': 4, 'UR2404': 4,
                      'UR404': 4, 'UR2405': 4, 'UR405': 4, 'SA2401': 4, 'SA401': 4, 'SA2402': 4, 'SA402': 4,
                      'SA2403': 4, 'SA403': 4, 'SA2404': 4, 'SA404': 4, 'SA2405': 4, 'SA405': 4, 'SA2406': 4,
                      'SA406': 4, 'SA2407': 4, 'SA407': 4, 'SA2408': 4, 'SA408': 4}


# 合约交易指令每次最小开仓下单量，如：RM405 -> 1，UR2401 -> 4，UR401 -> 4
def get_min_order(vt_symbol):
    '''
    :param vt_symbol: 'rb2010.SHFE'
    :return: minimum order quantity, int
    '''
    symbol, exchange = extract_vt_symbol(vt_symbol)
    symbol_pre = extract_symbol_pre(symbol, with_exchange=False)[0]
    return symbol_min_order.get(symbol_pre, symbol_min_order.get(symbol_pre.lower(), contract_min_order.get(symbol, 1)))


def get_size(strategy: CtaTemplate) -> int:
    """
    Return contract size data.
    用于覆写cta_engine.get_size()方法，否则行情服务器未启动时，无法获取合约倍数
    """
    return all_sizes.get(extract_symbol_pre(strategy.vt_symbol)[0], 0)
