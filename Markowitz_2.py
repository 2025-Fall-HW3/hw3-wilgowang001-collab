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
        n_assets = len(assets)

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # 策略：Mean-Variance Optimization with Momentum
        # 使用過去的平均報酬率作為預期報酬 (Momentum)，並減去風險懲罰
        
        # 設定風險厭惡係數 (可以調整，經驗上 1.0~2.0 在此作業表現不錯)
        risk_aversion = 1.0 

        # 建立 Gurobi 環境
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            
            # 遍歷每一天 (從 lookback 天數後開始)
            for i in range(self.lookback, len(self.returns)):
                # 1. 取得過去 N 天的報酬率 (嚴格切片，不含當天 i)
                window_returns = self.returns[assets].iloc[i - self.lookback : i]
                
                # 2. 計算預期報酬 mu (Momentum) 和 共變異數矩陣 Sigma
                # 這裡假設過去表現好的板塊，未來也會好 (動能效應)
                mu = window_returns.mean().values 
                Sigma = window_returns.cov().values
                
                # 3. 建構優化模型
                with gp.Model(env=env) as model:
                    # 變數: 權重 w (0 <= w <= 1)
                    w = model.addMVar(n_assets, lb=0.0, ub=1.0, name="w")
                    
                    # 目標: Maximize (Expected Return - Risk Penalty)
                    # Objective = w @ mu - 0.5 * gamma * (w @ Sigma @ w)
                    portfolio_return = w @ mu
                    portfolio_risk = w @ Sigma @ w
                    
                    model.setObjective(
                        portfolio_return - 0.5 * risk_aversion * portfolio_risk, 
                        gp.GRB.MAXIMIZE
                    )
                    
                    # 限制: 權重總和為 1
                    model.addConstr(w.sum() == 1, "budget")
                    
                    # 求解
                    model.optimize()
                    
                    # 4. 填入權重
                    current_date = self.returns.index[i]
                    if model.status == gp.GRB.OPTIMAL:
                        self.portfolio_weights.loc[current_date, assets] = w.X
                    else:
                        # 如果優化失敗，退回等權重
                        self.portfolio_weights.loc[current_date, assets] = 1.0 / n_assets

        """
        TODO: Complete Task 4 Above
        """

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
