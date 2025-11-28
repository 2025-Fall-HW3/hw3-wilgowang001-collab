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
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # 使用 Global Minimum Variance (GMV) 策略
        # 這個策略利用 Gurobi 找出波動率最小的組合，不需預測報酬率
        
        # 建立 Gurobi 環境 (放在迴圈外以提升效能)
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)  # 關閉 Gurobi 輸出資訊
            env.start()
            
            # 遍歷每一天 (從 lookback 天數後開始)
            for i in range(self.lookback, len(self.returns)):
                # 1. 取得過去 N 天的報酬率 (使用切片 i-lookback : i，不包含當天 i，避免偷看答案)
                window_returns = self.returns[assets].iloc[i - self.lookback : i]
                
                # 2. 計算共變異數矩陣 (Covariance Matrix)
                Sigma = window_returns.cov().values
                
                # 3. 建構優化模型
                with gp.Model(env=env) as model:
                    # 變數: 權重 w (0 <= w <= 1)
                    w = model.addMVar(n_assets, lb=0.0, ub=1.0, name="w")
                    
                    # 目標: 最小化組合變異數 (0.5 * w.T * Sigma * w)
                    # 這裡不需要 mu (預期報酬)，因為我們只求最小風險
                    model.setObjective(0.5 * w @ Sigma @ w, gp.GRB.MINIMIZE)
                    
                    # 限制: 權重總和為 1
                    model.addConstr(w.sum() == 1, "budget")
                    
                    # 求解
                    model.optimize()
                    
                    # 4. 填入權重
                    current_date = self.returns.index[i]
                    if model.status == gp.GRB.OPTIMAL:
                        self.portfolio_weights.loc[current_date, assets] = w.X
                    else:
                        # 如果優化失敗 (極少見)，退回等權重
                        self.portfolio_weights.loc[current_date, assets] = 1.0 / n_assets

        """
        TODO: Complete Task 4 Above

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
