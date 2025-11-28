"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
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

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust=False)
    df[asset] = raw['Adj Close']

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""


class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 1 Below
        """
        # Task 1: Equal Weight
        n_assets = len(assets)
        self.portfolio_weights[assets] = 1.0 / n_assets
        """
        TODO: Complete Task 1 Above
        """
        
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""


class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)
        
        # 先將所有權重初始化為 0
        self.portfolio_weights.fillna(0, inplace=True)

        """
        TODO: Complete Task 2 Below
        """
        # 改用迴圈 (Loop) 實作，確保與 Task 3 的 window 處理方式完全一致
        # 這能避免 pandas rolling 與 numpy std 之間的微小誤差
        
        for i in range(self.lookback, len(df)):
            # 1. 取得過去 lookback 天的報酬 (不包含今天 i)
            # 例如: i=50, window=[0...49]
            window_returns = df_returns[assets].iloc[i - self.lookback : i]
            
            # 2. 計算波動率 (std)
            # ddof=1 是預設值 (樣本標準差)，與 pandas 預設一致
            vols = window_returns.std()
            
            # 3. 計算倒數 (Inverse Volatility)
            # 這裡加上一個極小值 1e-8 避免除以 0 (雖然回測數據通常不會是 0)
            inv_vols = 1.0 / (vols + 1e-8)
            
            # 4. 歸一化
            weights = inv_vols / inv_vols.sum()
            
            # 5. 填入當天權重
            self.portfolio_weights.loc[df.index[i], assets] = weights.values

        """
        TODO: Complete Task 2 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns


"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]
            self.portfolio_weights.loc[df.index[i], assets] = self.mv_opt(
                R_n, self.gamma
            )

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                """
                TODO: Complete Task 3 Below
                """
                w = model.addMVar(n, name="w", lb=0.0, ub=1.0)
                model.addConstr(w.sum() == 1, name="budget")
                
                # Objective: Return - Gamma * Risk
                portfolio_return = mu @ w
                portfolio_risk = w @ Sigma @ w
                model.setObjective(portfolio_return - 0.5 * gamma * portfolio_risk, gp.GRB.MAXIMIZE)
                """
                TODO: Complete Task 3 Above
                """
                model.optimize()

                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    solution = []
                    for i in range(n):
                        var = model.getVarByName(f"w[{i}]")
                        solution.append(var.X)
                    return solution

        return [1/n]*n # Fallback

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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
    from grader import AssignmentJudge
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="append")
    parser.add_argument("--allocation", action="append")
    parser.add_argument("--performance", action="append")
    parser.add_argument("--report", action="append")
    args = parser.parse_args()
    judge = AssignmentJudge()
    judge.run_grading(args)
