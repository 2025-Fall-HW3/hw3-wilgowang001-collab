"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust=False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation
"""

class MyPortfolio:
    # 策略修改：將 lookback 調整為 126 (約半年)，反應比 252 快，比 50 穩
    def __init__(self, price, exclude, lookback=126, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )
        self.portfolio_weights.fillna(0, inplace=True)

        """
        TODO: Complete Task 4 Below
        """
        # 策略：Sharpe Momentum (選夏普值最高的，而不只是漲最多的)
        # 這能同時兼顧 "Sharpe > 1" (低波動) 和 "Sharpe > SPY" (高報酬)
        
        top_n = 3
        
        # 1. 預先計算 rolling mean (報酬) 和 rolling std (風險)
        # 記得 shift(1)
        rolling_mean = self.returns[assets].rolling(window=self.lookback).mean().shift(1)
        rolling_std = self.returns[assets].rolling(window=self.lookback).std().shift(1)
        
        # 2. 每天進行選股
        for i in range(self.lookback, len(self.returns)):
            date = self.returns.index[i]
            
            # 取得當天的 Mean 和 Std
            mu = rolling_mean.loc[date]
            sigma = rolling_std.loc[date]
            
            if not mu.isnull().all() and not sigma.isnull().all():
                # 計算簡易夏普值 (Mean / Std)
                # 這裡加 1e-8 避免除以 0
                sharpe_score = mu / (sigma + 1e-8)
                
                # 選夏普值最高的 top_n
                top_assets = sharpe_score.nlargest(top_n).index
                
                # 平均分配
                weight = 1.0 / top_n
                self.portfolio_weights.loc[date, top_assets] = weight
            else:
                self.portfolio_weights.loc[date, assets] = 0.0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    from grader_2 import AssignmentJudge
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="append")
    parser.add_argument("--allocation", action="append")
    parser.add_argument("--performance", action="append")
    parser.add_argument("--report", action="append")
    parser.add_argument("--cumulative", action="append")
    args = parser.parse_args()
    judge = AssignmentJudge()
    judge.run_grading(args)
