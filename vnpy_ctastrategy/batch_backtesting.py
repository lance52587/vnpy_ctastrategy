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
        self.setting = self.default_settings() if not os.path.exists(get_file_path(config_file)) else load_json(config_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.export = export + "\\" + timestamp + "\\"
        os.makedirs(self.export, exist_ok=True)

        self.classes = {}
        self.load_strategy_class = MethodType(CtaEngine.load_strategy_class, self)
        self.load_strategy_class_from_folder = MethodType(CtaEngine.load_strategy_class_from_folder, self)
        self.load_strategy_class_from_module = MethodType(CtaEngine.load_strategy_class_from_module, self)
        self.load_strategy_class()

        self.stats = []
        self.daily_dfs = {}
        self.show_chart = MethodType(BacktestingEngine.show_chart, self)

    def write_log(self, msg):
        print(msg)

    def default_settings(self, symbols=dbsymbols, standard=1):
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

    def add_parameters(self, engine, vt_symbol: str, start_date, end_date, interval="1m", capital=1_000_000):
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
                capital=capital
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

                self.stats.append(stat)
                self.daily_dfs[_name] = df
                engine.clear_data()


    def run_batch_test_file(self, para_dict="cta_strategy.json", start_date=datetime(2024, 5, 1),
                         end_date=datetime(2024, 12, 1)):
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
        print(f'stra_setting: {stra_setting}')
        self.run_batch_test(stra_setting, start_date, end_date)
        self.result_excel(self.stats)

    def result_excel(self, result):
        """
        输出交易结果到excel
        """
        try:
            path = self.export + "daily_stats.xlsx"
            result = pd.DataFrame(result)
            excel_writer = pd.ExcelWriter(path)
            result.to_excel(excel_writer, sheet_name='stats', index=False)
            for k, v in self.daily_dfs.items():
                v.to_excel(excel_writer, sheet_name=str(k), index=False)
                fig = self.show_chart(v)
                pio.write_html(fig, file=f'{self.export}{k}.html', auto_open=False)

            excel_writer.close()
            print(f'CTA Batch result is export to {path}')
        except:
            print(traceback.format_exc())

if __name__ == '__main__':
    bts = BatchBackTest()
    bts.run_batch_test_file()
