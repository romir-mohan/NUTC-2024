# NUTC Trading

This repository contains a Python implementation of advanced trading strategies for a market-making algorithm. The strategies are designed to execute trades based on dynamically calculated fair prices, leveraging statistical modeling and predictive algorithms.

## Table of Contents
- [Overview](#overview)
- [Strategies Implemented](#strategies-implemented)
  - [Market Making Strategy](#market-making-strategy)
  - [Penny In, Penny Out with Levels Strategy](#penny-in-penny-out-with-levels-strategy)
- [Fair Price Calculation](#fair-price-calculation)
  - [Order Book Update](#order-book-update)
  - [Sliding Window Median Algorithm](#sliding-window-median-algorithm)
  - [Exponential Smoothing](#exponential-smoothing)
  - [Kernel Density Estimation (KDE)](#kernel-density-estimation-kde)
  - [Combining Predictions](#combining-predictions)
- [Class Structure](#class-structure)
  - [Strategy Class](#strategy-class)
  - [Prediction Class](#prediction-class)

## Overview
The code implements two primary trading strategies within a `Strategy` class:

- **Market Making Strategy**: Places limit buy and sell orders around the estimated fair price with dynamic spreads and quantities.
- **Penny In, Penny Out with Levels Strategy**: Places multiple orders at different spread levels to increase trading frequency and capture small price movements.

The fair price is calculated using a combination of real-time order book data and historical price information. The `Prediction` class is responsible for estimating the fair price by applying statistical methods, including the Sliding Window Median algorithm and Kernel Density Estimation (KDE).

## Strategies Implemented

### Market Making Strategy
**Objective**: To profit from the bid-ask spread by placing simultaneous limit buy and sell orders around the fair price.

**Key Features:**
- **Dynamic Spread Calculation**: The spread between the buy and sell orders is adjusted based on market volatility and current market spreads.
- **Dynamic Quantity Calculation**: The order quantity is determined based on available capital, position limits, and current fair price.
- **Risk Management**: Incorporates position limits to prevent overexposure to any single asset.

**Implementation Details:**
- **Fair Price Retrieval**: The strategy obtains the latest fair price calculated by the `Prediction` class.
- **Spread Calculation**:
  - **Base Spread**: Starts with a base spread percentage of the fair price.
  - **Volatility Adjustment**: Adjusts the spread based on recent price volatility.
  - **Market Spread Adjustment**: Ensures the spread is not narrower than the current market spread.
  - **Bounds Enforcement**: Limits the spread within predefined minimum and maximum values.
- **Quantity Calculation**:
  - **Capital Allocation**: Allocates a fixed percentage of total capital for each trade.
  - **Position Limits**: Ensures the total position does not exceed a set percentage of capital.
  - **Final Quantity**: Rounds down to the nearest whole number.
- **Order Placement**: Places limit buy and sell orders at prices calculated by adjusting the fair price with half the spread.

### Penny In, Penny Out with Levels Strategy
**Objective**: To increase trading frequency and capture small profits by placing multiple orders at various spread levels.

**Key Features:**
- **Spread Levels**: Utilizes different spreads (tight, slightly wider, even wider) to place orders.
- **Dynamic Order Allocation**: The number of orders at each spread level is determined based on predefined ratios.
- **Dynamic Quantity Calculation**: Order quantities are adjusted inversely proportional to the spread.
- **Order Management**: Handles unknown open order limits by attempting to place orders and canceling the oldest ones if necessary.

**Implementation Details:**
- **Fair Price Retrieval**: Obtains the latest fair price from the `Prediction` class.
- **Spread Levels Definition**: Defines spread levels with associated allocation ratios.
- **Order Allocation Calculation**:
  - **Total Desired Orders**: Sets a desired total number of open orders per ticker.
  - **Order Counts per Spread**: Calculates the number of orders for each spread level based on ratios.
  - **Adjustment**: Ensures the total number of orders matches the desired total.
- **Order Placement**:
  - Iterates over each spread level and attempts to place buy and sell order pairs.
  - **Quantity Calculation**: Adjusts quantities dynamically, increasing for tighter spreads.
  - **Order Failure Handling**: If order placement fails (e.g., due to open order limits), the strategy cancels the oldest orders and retries.

## Fair Price Calculation
This combines real-time order book data with historical price information to produce a price prediction.

### Order Book Update
- **Data Collection**: Continuously updates the local order book with the latest bid and ask prices and volumes.
- **Order Book Structure**: Maintains separate dictionaries for bids and asks, with prices as keys and quantities as values.

### Sliding Window Median Algorithm
**Purpose**: To estimate the fair price based on the current state of the order book.

**Implementation:**
- **Order List Creation**: Generates a sorted list of orders, assigning each a type (0 for bid, 1 for ask).
- **Parentheses String**: Constructs a string representing the order flow, using '(' for bids and ')' for asks.
- **Fair Price Index**: Finds the index where the difference between open and close parentheses is minimized.
- **Fair Price Determination**: Selects the price at the fair price index as the initial fair price estimate.

### Exponential Smoothing
**Purpose**: To smooth out the fair price estimate over time, reducing sensitivity to short-term fluctuations.

**Implementation:**
- **Decay Factor (alpha)**: Determines the weight of the most recent price in the average.
- **Soft Average**: Updates the average fair price using the formula:

  ```
  soft_average = (1 - α) × soft_average + α × current_price
  ```

### Kernel Density Estimation (KDE)
**Purpose**: To model the probability distribution of historical prices and identify the most probable price level.

**Implementation:**
- **Data Preparation**: Accumulates historical price data from trade updates.
- **Bandwidth Selection**: Uses Scott's Rule to determine the bandwidth for the KDE.
- **Density Estimation**: Fits a Gaussian KDE to the historical prices.
- **Most Probable Price**: Finds the price corresponding to the maximum density in the estimated distribution.

### Combining Predictions
**Final Fair Price**: Combines the soft average and the KDE-estimated price to produce the final fair price estimate.

**Formula:**

```
fair_price = 0.5 × soft_average + 0.5 × kde_price

```

## Class Structure

### Strategy Class
**Responsibilities:**
- Manages trading strategies and order placement.
- Maintains capital and position information.
- Handles order book and trade updates.

**Key Methods:**
- `on_orderbook_update`: Processes order book changes and triggers strategy execution.
- `on_trade_update`: Updates price history based on executed trades.
- `on_account_update`: Updates capital and positions after trades are filled.
- `execute_trading_strategy`: Calls individual trading strategies.
- `market_making_strategy`: Implements the market-making strategy with dynamic parameters.
- `penny_in_out_strategy`: Implements the Penny In, Penny Out strategy with levels.
- `calculate_dynamic_spread`: Calculates the spread for order placement based on volatility and market conditions.
- `calculate_dynamic_quantity`: Determines the order quantity based on capital allocation and position limits.
- `place_order`: Handles order placement and tracking.
- `try_place_order_with_retry`: Attempts to place an order and retries if necessary by canceling oldest orders.
- `cancel_oldest_order`: Cancels the oldest open order to free up capacity.

### Prediction Class
**Responsibilities:**
- Estimates the fair price using order book data and historical prices.
- Maintains internal state for price predictions.

**Key Methods:**
- `update`: Updates the predictor with the latest order book data.
- `predict_converge`: Calculates the initial fair price estimate using the Sliding Window Median algorithm.
- `update_price_history`: Adds new trade prices to the historical data.
- `update_kde`: Updates the KDE with the latest price history.
- `predict`: Combines the soft average and KDE estimate to produce the final fair price.
