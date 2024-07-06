# encoding: UTF-8
import json, os
import traceback
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from types import MethodType
import plotly.io as pio

import pandas as pd
from vnpy.trader.utility import load_json, get_file_path
from vnpy.utils.symbol_info import all_priceticks, illiquid_symbol, all_sizes, dbsymbols, all_symbol_pres_rev

from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.engine import CtaEngine

class BatchBackTest:
    """
    提供批量CTA策略回测，输出结果到excel或pdf，和CTA策略批量优化，输出结果到excel或pdf，
    """

    def __init__(self, config_file="vt_symbol.json", export="result"):
        """
        加载配置路径
        """
        # 先用默认参数，后续再从配置文件读取进行更新
        self.setting = self.default_parameters()
        if os.path.exists(get_file_path(config_file)):
            _setting = load_json(config_file)
            for k, v in _setting.items():
                if k in self.setting:
                    print(f"symbol {k} para update from\n{self.setting[k]}\nto\n{v}\n")
                    self.setting[k].update(v)
                else:
                    print(f"symbol {k} para added\n{v}\n")
                    self.setting[k] = v

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.export = export + "\\" + timestamp + "\\"
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
        self.port_pnl = {}
        self.port_stats = {}

    def write_log(self, msg):
        print(msg)

    def default_parameters(self, symbols=dbsymbols, standard=1):
        setting = {}
        for symbol in symbols:
            if symbol not in illiquid_symbol:
                vt_symbol = f"{symbol}888.{all_symbol_pres_rev.get(symbol, 'LOCAL')}"
                setting[vt_symbol] = {
                    "rate": 2.5e-5 if standard == 1 else 0.,
                    "slippage": 2 * all_priceticks[symbol] if standard == 1 else 0.,
                    "size": all_sizes[symbol],
                    "pricetick": all_priceticks[symbol]
                }
        return setting

    def add_parameters(self, engine, vt_symbol: str, start_date, end_date, interval="1m", capital=None):
        """
        从vtSymbol.json文档读取品种的交易属性，比如费率，交易每跳，比率，滑点
        """
        if vt_symbol in self.setting:
            engine.set_parameters(
                vt_symbol=vt_symbol,
                interval=interval,
                start=start_date,
                end=end_date,
                rate=self.setting[vt_symbol]["rate"],
                slippage=self.setting[vt_symbol]["slippage"],
                size=self.setting[vt_symbol]["size"],
                pricetick=self.setting[vt_symbol]["pricetick"],
                capital=self.capital if capital is None else capital
            )
        else:
            print(f"symbol {vt_symbol} hasn't be maintained in config file")
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

        # todo: 并行回测
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


    def run_batch_test_file(self, para_dict="cta_strategy.xlsx", start_date=datetime(2024, 5, 1),
                         end_date=datetime(2024, 12, 1), agg_by='class_name'):
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
        # 逐一打印
        for k, v in stra_setting.items():
            print(f'{k}: {v}')
        self.agg_by = agg_by
        self.run_batch_test(stra_setting, start_date, end_date)
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
            port_result.to_excel(excel_writer, sheet_name='port_stats', index=False)
            for k, v in self.port_pnl.items():
                v.to_excel(excel_writer, sheet_name=f'port_{k}')
                fig = self.show_chart(v)
                pio.write_html(fig, file=f'{self.export}port_{k}.html', auto_open=False)

            # 单品种回测结果
            result = pd.DataFrame.from_dict(self.stats, orient='index')
            result.to_excel(excel_writer, sheet_name='stats', index=False)
            for k, v in self.daily_dfs.items():
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
        if agg_by not in ('class_name', 'vt_symbol', 'setting', 'all'):
            print('not supported agg_by')
            return

        # 拿到unique的by值，将同一类的策略dail_df汇总
        if agg_by == 'all':
            unique_by = ['all']
        else:
            unique_by = list(set([v[agg_by] for v in self.stats.values()]))

            
        self.daily_df_d = {}
        if agg_by == 'all':
            self.daily_df_d['all'] = self.daily_dfs
        else:
            for ub in unique_by:
                self.daily_df_d[ub] = {k: v for k, v in self.daily_dfs.items() if self.stats[k][agg_by] == ub}

        # 汇总daily_df
        self.agg_daily_view()

    def agg_daily_view(self):
        """
        将daily_df_d每个k下的v逐一merge，生成各k下的port_pnl，并engine.calculate_statistics(port_pnl)
        """
        columns = ['date', 'balance', 'net_pnl', 'commission', 'slippage', 'turnover', 'trade_count']
        for k, v in self.daily_df_d.items():
            port_pnl = pd.DataFrame()
            for _k, _v in v.items():
                if port_pnl.empty:
                    port_pnl = _v.reset_index()[columns]
                else:
                    port_pnl = port_pnl.merge(_v.reset_index()[columns], on='date', how='outer').sort_values(by='date')
                    port_pnl['balance_x'] = port_pnl['balance_x'].fillna(method='ffill').fillna(self.capital)
                    port_pnl['balance_y'] = port_pnl['balance_y'].fillna(method='ffill').fillna(self.capital)
                    port_pnl.fillna(0, inplace=True)

                    port_pnl['balance'] = port_pnl['balance_x'] + port_pnl['balance_y']
                    port_pnl['net_pnl'] = port_pnl['net_pnl_x'] + port_pnl['net_pnl_y']
                    port_pnl['commission'] = port_pnl['commission_x'] + port_pnl['commission_y']
                    port_pnl['slippage'] = port_pnl['slippage_x'] + port_pnl['slippage_y']
                    port_pnl['turnover'] = port_pnl['turnover_x'] + port_pnl['turnover_y']
                    port_pnl['trade_count'] = port_pnl['trade_count_x'] + port_pnl['trade_count_y']
                    port_pnl = port_pnl[columns]

            # 统计资产组合pnl
            port_pnl['drawdown'] = port_pnl['balance'] - port_pnl['balance'].expanding().max()
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
    bts.run_batch_test_file(agg_by='class_name')
