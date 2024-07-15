# encoding: UTF-8
import json, os
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import MethodType
import plotly.io as pio
from pathos.multiprocessing import ProcessPool
from glob import glob
import dill

dill.settings['recurse'] = True
import pandas as pd
from vnpy.trader.utility import load_json, get_file_path
from vnpy.utils.symbol_info import all_priceticks, illiquid_symbol, all_sizes, dbsymbols, all_symbol_pres_rev

from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.engine import CtaEngine

class BatchBackTest:
    """
    提供批量CTA策略回测，输出结果到excel或pdf，和CTA策略批量优化，输出结果到excel或pdf，
    """

    def __init__(self, para_file="vt_symbol.json", export="result"):
        """
        加载配置路径
        """
        # 先用默认参数，后续再从配置文件读取进行更新
        self.paras = self.default_parameters()
        if os.path.exists(get_file_path(para_file)):
            para = load_json(para_file)
            for k, v in para.items():
                if k in self.paras:
                    print(f"symbol {k} para update from\n{self.paras[k]}\nto\n{v}\n")
                    self.paras[k].update(v)
                else:
                    print(f"symbol {k} para added\n{v}\n")
                    self.paras[k] = v

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.export = os.path.join(export, timestamp) + os.sep
        os.makedirs(self.export, exist_ok=True)

        self.classes = {}
        self.load_strategy_class = MethodType(CtaEngine.load_strategy_class, self)
        self.load_strategy_class_from_folder = MethodType(CtaEngine.load_strategy_class_from_folder, self)
        self.load_strategy_class_from_module = MethodType(CtaEngine.load_strategy_class_from_module, self)
        self.load_strategy_class()

        self.stats = {}
        self.daily_dfs = {}
        self.show_chart = MethodType(BacktestingEngine.show_chart, self)

        self.capital = 1_000_000
        self.agg_by = 'all'
        self.port_pnl = {}
        self.port_stats = {}

    def write_log(self, msg):
        print(msg)

    def default_parameters(self, symbols=dbsymbols, standard=1):
        para = {}
        # 期货品种
        for symbol in symbols:
            if symbol not in illiquid_symbol:
                vt_symbol = f"{symbol}.{all_symbol_pres_rev.get(symbol, 'LOCAL')}"
                para[vt_symbol] = {
                    "rate": 2.5e-5 if standard == 1 else 0.,
                    "slippage": 2 * all_priceticks[symbol] if standard == 1 else 0.,
                    "size": all_sizes[symbol],
                    "pricetick": all_priceticks[symbol]
                }

        # todo: 默认币的配置价格跳动和合约大小，而非要求用户配置在vt_symbol.json中

        # 股票品种
        for exchange in ('SSE', 'SZSE', 'BSE'):
            para[exchange] = {
                "rate": 8.5e-5 if standard == 1 else 0.,
                "slippage": 0.02 if standard == 1 else 0.,
                "size": 100,
                "pricetick": 0.02
            }
        return para

    def add_parameters(self, engine, vt_symbol: str, start_date, end_date, interval="1m", capital=None):
        """
        从vtSymbol.json文档读取品种的交易属性，比如费率，交易每跳，比率，滑点
        """
        symbol, exchange = vt_symbol.rsplit(".", 1)
        if exchange in ('SHFE', 'CFFEX', 'DCE', 'CZCE'):
            default_vt_symbol = f"{symbol.strip('0123456789')}.{exchange}"
        elif exchange in ('SSE', 'SZSE', 'BSE'):
            default_vt_symbol = f"{exchange}"
        else:
            default_vt_symbol = vt_symbol
        
        if vt_symbol in self.paras or default_vt_symbol in self.paras:
            _vt_symbol = vt_symbol if vt_symbol in self.paras else default_vt_symbol
            engine.set_parameters(
                vt_symbol=vt_symbol,
                interval=interval,
                start=start_date,
                end=end_date,
                rate=self.paras[_vt_symbol]["rate"],
                slippage=self.paras[_vt_symbol]["slippage"],
                size=self.paras[_vt_symbol]["size"],
                pricetick=self.paras[_vt_symbol]["pricetick"],
                capital=self.capital if capital is None else capital
            )
        else:
            print(f"{vt_symbol} parameter hasn't be maintained in para file")
        return engine


    def run_batch_test(self, stra_setting, start_date, end_date):
        """
        进行回测
        """
        # 单品种只加载一次历史数据：每个品种只实例化一个engine，回测完后clear_data，再回测该品种下一个策略配置
        stra_setting_vtsymbol = defaultdict(list)
        for _name, _config in stra_setting.items():
            vt_symbol = _config["vt_symbol"]
            stra_setting_vtsymbol[vt_symbol].append((_name, _config))

        for vt_symbol, stra_setting_vt in stra_setting_vtsymbol.items():
            engine = BacktestingEngine()
            engine = self.add_parameters(engine, vt_symbol, start_date, end_date)
            engine.load_data()
            for _name, _config in stra_setting_vt:
                _setting = json.loads(_config["setting"]) if type(_config["setting"]) is str else _config["setting"]
                engine.add_strategy(self.classes.get(_config["class_name"]), _setting)
                engine.run_backtesting()
                engine.calculate_result()
                stat = engine.calculate_statistics(output=False)
                df = engine.daily_df
                stat["class_name"] = _config["class_name"]
                stat["setting"] = str(_setting)
                stat["vt_symbol"] = vt_symbol
                stat['strategy_name'] = _name

                self.stats[_name] = stat
                self.daily_dfs[_name] = df
                engine.clear_data()

    def run_single_test(self, vt_symbol, stra_setting_vt, start_date, end_date):
        """
        单个品种的回测任务
        """
        engine = BacktestingEngine()
        engine = self.add_parameters(engine, vt_symbol, start_date, end_date)
        engine.load_data()
        results = []
        for _name, _config in stra_setting_vt:
            _setting = json.loads(_config["setting"]) if isinstance(_config["setting"], str) else _config["setting"]
            engine.add_strategy(self.classes.get(_config["class_name"]), _setting)
            engine.run_backtesting()
            engine.calculate_result()
            stat = engine.calculate_statistics(output=False)
            df = engine.daily_df
            stat["class_name"] = _config["class_name"]
            stat["setting"] = str(_setting)
            stat["vt_symbol"] = vt_symbol
            stat['strategy_name'] = _name
            results.append((_name, stat, df))
            engine.clear_data()
        return results

    def run_concurrent_test(self, stra_setting, start_date, end_date):
        """
        进行回测
        """
        # 单品种只加载一次历史数据：每个品种只实例化一个engine，回测完后clear_data，再回测该品种下一个策略配置
        stra_setting_vtsymbol = defaultdict(list)
        for _name, _config in stra_setting.items():
            vt_symbol = _config["vt_symbol"]
            stra_setting_vtsymbol[vt_symbol].append((_name, _config))

        # 并行回测
        with ProcessPool() as pool:
            results = pool.map(lambda x: self.run_single_test(*x),
                               [(vt_symbol, stra_setting_vt, start_date, end_date)
                                for vt_symbol, stra_setting_vt in stra_setting_vtsymbol.items()])

        # 整理结果
        for result in results:
            for _name, stat, df in result:
                self.stats[_name] = stat
                self.daily_dfs[_name] = df

    def run_batch_test_file(self, para_dict="cta_strategy.xlsx", start_date=datetime(2024, 5, 1),
                         end_date=datetime(2024, 12, 1), agg_by='all'):
        """
        从ctaStrategy.json去读交易策略和参数，进行回测
        """
        if para_dict.endswith('.json'):
            stra_setting = load_json(para_dict)
        elif para_dict.endswith('.xlsx'):
            filepath: Path = get_file_path(para_dict)
            df = pd.read_excel(filepath)
            stra_setting = df.to_dict(orient='index')
        else:
            print('para_dict format not supported')
            return
        print(f'stra_setting:')
        for k, v in stra_setting.items():
            print(f'{k}: {v}')

        self.agg_by = agg_by
        if agg_by not in ('class_name', 'vt_symbol', 'setting', 'all'):
            print('not supported agg_by')
            return

        # self.run_batch_test(stra_setting, start_date, end_date)
        self.run_concurrent_test(stra_setting, start_date, end_date)
        self.daily_view()
        self.save_result()


    def summary_result_from_folder(self, folder=None):
        """
        从指定文件夹读取各单品种回测结果.xlsx到daily_dfs，并进行daily_view和save_result
        """
        if folder is None:
            folder = os.getcwd()

        self.daily_dfs = {}
        for file in glob(f'{folder}/*.xlsx'):
            df = pd.read_excel(file)
            self.daily_dfs[os.path.basename(file).rsplit('.', 1)[0]] = df
        self.daily_view()
        self.save_result()


    def save_result(self):
        """
        输出交易结果到excel
        """
        try:
            path = self.export + "daily_stats.xlsx"
            excel_writer = pd.ExcelWriter(path)

            # 资产组合回测结果
            port_result = pd.DataFrame.from_dict(self.port_stats, orient='index')
            port_result.reset_index(drop=True).to_excel(excel_writer, sheet_name='port_stats')
            for n, (k, v) in enumerate(self.port_pnl.items()):
                if self.agg_by != 'setting':
                    sheet_name = f'port_{k}'
                else:
                    sheet_name = f'port_{n}'
                v.to_excel(excel_writer, sheet_name=sheet_name)
                fig = self.show_chart(v)
                pio.write_html(fig, file=f'{self.export}{sheet_name}.html', auto_open=False)

            # 单品种回测结果
            result = pd.DataFrame.from_dict(self.stats, orient='index')
            result.to_excel(excel_writer, sheet_name='stats', index=False)
            for k, v in self.daily_dfs.items():
                # 如果没有balance列，将net_pnl累加得到balance列
                if 'balance' not in v.columns:
                    v['balance'] = v['net_pnl'].cumsum() + self.capital
                    v["highlevel"] = v["balance"].expanding().max()
                    v["drawdown"] = v["balance"] - v["highlevel"]
                v.to_excel(excel_writer, sheet_name=str(k))
                fig = self.show_chart(v)
                pio.write_html(fig, file=f'{self.export}{k}.html', auto_open=False)

            excel_writer.close()
            print(f'CTA Batch result is export to {path}')
        except:
            print(traceback.format_exc())

    def daily_view(self):
        """
        按所选视图方式，分类daily_df
        """
        agg_by = self.agg_by
        self.daily_df_d = {}
        if agg_by == 'all':
            self.daily_df_d['all'] = self.daily_dfs
        else:
            # 拿到unique的by值，将同一类的策略dail_df汇总
            unique_by = list(set([v[agg_by] for v in self.stats.values()]))
            for ub in unique_by:
                self.daily_df_d[ub] = {k: v for k, v in self.daily_dfs.items() if self.stats[k][agg_by] == ub}

        # 汇总daily_df
        self.agg_daily_view()

    def agg_daily_view(self, daily_df_d=None):
        """
        将daily_df_d每个k下的v逐一merge，生成各k下的port_pnl，并engine.calculate_statistics(port_pnl)
        """
        if daily_df_d is None:
            daily_df_d = self.daily_df_d

        columns = ['date', 'net_pnl', 'commission', 'slippage', 'turnover', 'trade_count']
        for k, v in daily_df_d.items():
            port_pnl = pd.DataFrame()
            for _k, _v in v.items():
                if _v is None:
                    continue
                if port_pnl.empty:
                    port_pnl = _v.reset_index()[columns]
                else:
                    port_pnl = port_pnl.merge(_v.reset_index()[columns], on='date', how='outer').sort_values(by='date')
                    port_pnl.fillna(0, inplace=True)

                    port_pnl['net_pnl'] = port_pnl['net_pnl_x'] + port_pnl['net_pnl_y']
                    port_pnl['commission'] = port_pnl['commission_x'] + port_pnl['commission_y']
                    port_pnl['slippage'] = port_pnl['slippage_x'] + port_pnl['slippage_y']
                    port_pnl['turnover'] = port_pnl['turnover_x'] + port_pnl['turnover_y']
                    port_pnl['trade_count'] = port_pnl['trade_count_x'] + port_pnl['trade_count_y']
                    port_pnl = port_pnl[columns]

            # 统计资产组合pnl
            port_pnl.set_index('date', inplace=True)  # 将排序后的date列设置为索引

            engine = BacktestingEngine()
            engine.capital = self.capital * len(v)
            engine.daily_df = port_pnl
            port_stats = engine.calculate_statistics(output=False)
            port_stats[self.agg_by] = k
            self.port_stats[k] = port_stats
            self.port_pnl[k] = engine.daily_df


if __name__ == '__main__':
    bts = BatchBackTest()
    # bts.run_batch_test_file(agg_by='class_name')# agg_by='class_name' or 'vt_symbol' or 'setting' or 'all'

    bts.summary_result_from_folder('outer_result')