import math
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import KernelDensity

# ===========================
# Configuration Parameters
# ===========================

STARTING_CAPITAL: float = 100000.0

# Risk Management
RISK_POSITION_LIMIT_PCT: float = 0.10  # Max 10% of capital in one asset

# Arbitrage Strategy
ARB_CAPITAL_ALLOCATION_PCT: float = 1  # Allocate 2% of capital to arbitrage

STOP_LOSS_PCT: float = 0.05  # 5% stop loss
TRAILING_STOP_LOSS_PCT: float = 0.07  # 7% trailing stop loss

# Market Making Strategy
MM_BASE_SPREAD_PCT: float = 0.001  # 0.1%
MM_MIN_SPREAD: float = 0.05
MM_MAX_SPREAD: float = 1.0
MM_WINDOW_SIZE: int = 50
MM_VOLATILITY_FACTOR: float = 100  # Adjust sensitivity as needed
MM_BASE_QUANTITY: int = 1
MM_CAPITAL_ALLOCATION_PCT: float = 0.01
MM_MAX_QUANTITY: int = 100

class Side(Enum):
    BUY = 0
    SELL = 1


class Ticker(Enum):
    ETH = 0
    BTC = 1
    LTC = 2


def place_market_order(side: Side, ticker: Ticker, quantity: float) -> bool:
    """Place a market order - DO NOT MODIFY"""
    return True


def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order - DO NOT MODIFY"""
    return 0


def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel a limit order - DO NOT MODIFY"""
    return True


class Strategy:
    """Trading strategy implementing advanced statistical modeling and predictive algorithms."""

    def __init__(self) -> None:
        self.order_ids: Dict[
            int, Dict] = {}  # {order_id: {'ticker': Ticker, 'side': Side, 'ioc': bool, 'strategy': str}}
        self.capital: float = STARTING_CAPITAL  # Starting capital
        self.positions: Dict[Ticker, float] = defaultdict(float)  # Stock positions {ticker: quantity}

        # Order book data
        self.order_book: Dict[Ticker, Dict[str, Dict[float, float]]] = defaultdict(
            lambda: {'buy': defaultdict(float), 'sell': defaultdict(float)}
        )

        # Price history for each ticker
        self.price_history: Dict[Ticker, List[float]] = defaultdict(list)  # {ticker: [prices]}

        # Predicted fair prices
        self.fair_prices: Dict[Ticker, float] = {}  # {ticker: fair_price}

        # Initialize predictors for each ticker
        self.predictors: Dict[Ticker, 'Prediction'] = {
            Ticker.ETH: Prediction(Ticker.ETH),
            Ticker.BTC: Prediction(Ticker.BTC),
            Ticker.LTC: Prediction(Ticker.LTC),
        }

        # Add these lines to initialize entry_prices and highest_prices
        self.entry_prices: Dict[Ticker, float] = {}
        self.highest_prices: Dict[Ticker, float] = {}

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        print(f"Orderbook update: {ticker.name} {side.name} {price} {quantity}")

        # Update local order book
        side_str = 'buy' if side == Side.BUY else 'sell'
        if quantity == 0:
            if price in self.order_book[ticker][side_str]:
                del self.order_book[ticker][side_str][price]
        else:
            self.order_book[ticker][side_str][price] = quantity

        # Update predictor with new order book data
        self.predictors[ticker].update(self.order_book[ticker])

        # Compute fair price using the predictor
        fair_price = self.predictors[ticker].predict()
        self.fair_prices[ticker] = fair_price
        print(f"Computed fair price for {ticker.name}: {fair_price}")

        # Execute trading strategies based on fair price
        self.execute_trading_strategy(ticker)

    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        print(f"Trade update: {ticker.name} {side.name} {price} {quantity}")

        # Update price history
        self.price_history[ticker].append(price)
        self.predictors[ticker].update_price_history(price)

    def on_account_update(
            self,
            ticker: Ticker,
            side: Side,
            price: float,
            quantity: float,
            capital_remaining: float,
    ) -> None:
        print(f"Account update: {ticker.name} {side.name} {price} {quantity} {capital_remaining}")
        self.capital = capital_remaining

        # Update positions
        if side == Side.BUY:
            self.positions[ticker] += quantity
        elif side == Side.SELL:
            self.positions[ticker] -= quantity

        # Remove filled order from order_ids
        filled_order_ids = [oid for oid, details in self.order_ids.items()
                            if details['ticker'] == ticker and details['side'] == side]
        for oid in filled_order_ids:
            del self.order_ids[oid]

    def execute_trading_strategy(self, ticker: Ticker) -> None:
        self.market_making_strategy(ticker)
        self.arbitrage_strategy(ticker)
        self.check_risk_management(ticker)

    def arbitrage_strategy(self, ticker: Ticker) -> None:
        """Execute Arbitrage Trading Strategy."""
        fair_price = self.fair_prices.get(ticker)
        if fair_price is None:
            print(f"Fair price for {ticker.name} is not available. Skipping arbitrage strategy.")
            return

        # Get the best bid and ask from the order book
        best_bid = max(self.order_book[ticker]['buy'].keys(), default=None)
        best_ask = min(self.order_book[ticker]['sell'].keys(), default=None)

        if best_bid is None or best_ask is None:
            print(f"Order book for {ticker.name} is empty. Skipping arbitrage strategy.")
            return

        # Check for arbitrage opportunity
        # If the best bid is higher than the fair price, sell
        if best_bid > fair_price:
            quantity = self.calculate_arbitrage_quantity(ticker, fair_price, Side.SELL)
            if quantity > 0:
                success = self.place_market_order(Side.SELL, ticker, quantity)
                if success:
                    print(f"Executed arbitrage SELL order for {quantity} units of {ticker.name} at price {best_bid}.")
        # If the best ask is lower than the fair price, buy
        elif best_ask < fair_price:
            quantity = self.calculate_arbitrage_quantity(ticker, fair_price, Side.BUY)
            if quantity > 0:
                success = self.place_market_order(Side.BUY, ticker, quantity)
                if success:
                    print(f"Executed arbitrage BUY order for {quantity} units of {ticker.name} at price {best_ask}.")

    def calculate_arbitrage_quantity(self, ticker: Ticker, fair_price: float, side: Side) -> float:
        """
        Calculate the quantity to trade for arbitrage based on capital and position limits.
        """
        max_trade_value = self.capital * ARB_CAPITAL_ALLOCATION_PCT

        if side == Side.BUY:
            quantity = max_trade_value / fair_price
            quantity = min(quantity, self.get_max_buy_quantity(ticker))
        else:
            quantity = min(max_trade_value / fair_price, self.get_max_sell_quantity(ticker))

        quantity = max(0, int(quantity))
        return quantity

    def get_max_buy_quantity(self, ticker: Ticker) -> float:
        """
        Calculate the maximum quantity that can be bought based on position limits.
        """
        fair_price = self.fair_prices.get(ticker, 1)  # Avoid division by zero
        max_position = (self.capital * RISK_POSITION_LIMIT_PCT) / fair_price
        current_position = self.positions.get(ticker, 0)
        available_quantity = max(0, max_position - current_position)
        return available_quantity

    def get_max_sell_quantity(self, ticker: Ticker) -> float:
        """
        Calculate the maximum quantity that can be sold based on current holdings.
        """
        current_position = self.positions.get(ticker, 0)
        return max(0, current_position)

    def check_risk_management(self, ticker: Ticker) -> None:
        """
        Implement risk management checks such as stop-loss and position limits.
        """
        position = self.positions.get(ticker, 0)

        current_price = self.fair_prices.get(ticker, 0)

        if current_price == 0 or position == 0:
            return

        max_position = (self.capital * RISK_POSITION_LIMIT_PCT) / current_price
        if abs(position) > max_position:
            print(f"Position limit exceeded for {ticker.name}. Executing position reduction.")
            self.reduce_position(ticker, position, max_position)

        if ticker not in self.entry_prices:
            self.entry_prices[ticker] = current_price
            self.highest_prices[ticker] = current_price

        entry_price = self.entry_prices[ticker]
        highest_price = self.highest_prices[ticker]

        if current_price > highest_price:
            self.highest_prices[ticker] = current_price

        if position > 0:
            if current_price < entry_price * (1 - STOP_LOSS_PCT):
                print(f"Stop-loss triggered for {ticker.name}. Closing position.")
                self.close_position(ticker)
            elif current_price < highest_price * (1 - TRAILING_STOP_LOSS_PCT):
                print(f"Trailing stop-loss triggered for {ticker.name}. Closing position.")
                self.close_position(ticker)
        elif position < 0:
            if current_price > entry_price * (1 + STOP_LOSS_PCT):
                print(f"Stop-loss triggered for {ticker.name}. Closing position.")
                self.close_position(ticker)
            elif current_price > highest_price * (1 + TRAILING_STOP_LOSS_PCT):
                print(f"Trailing stop-loss triggered for {ticker.name}. Closing position.")
                self.close_position(ticker)

    def reduce_position(self, ticker: Ticker, current_position: float, max_position: float) -> None:
        """Reduce position to comply with position limits."""
        reduction_amount = abs(current_position) - max_position
        side = Side.SELL if current_position > 0 else Side.BUY
        success = self.place_market_order(side, ticker, reduction_amount)
        if success:
            print(f"Reduced position for {ticker.name} by {reduction_amount}")

    def close_position(self, ticker: Ticker) -> None:
        """Close the entire position for a given ticker."""
        position = self.positions.get(ticker, 0)
        if position != 0:
            side = Side.SELL if position > 0 else Side.BUY
            success = self.place_market_order(side, ticker, abs(position))
            if success:
                print(f"Closed entire position for {ticker.name}")
                del self.entry_prices[ticker]
                del self.highest_prices[ticker]

    def market_making_strategy(self, ticker: Ticker) -> None:
        """Execute Market Making Trading Strategy with dynamic spread and quantity."""
        fair_price = self.fair_prices.get(ticker, 0.0)
        if fair_price == 0.0:
            print(f"Fair price for {ticker.name} is not available. Skipping market making strategy.")
            return

        spread = self.calculate_dynamic_spread(ticker, fair_price)
        quantity = self.calculate_dynamic_quantity(ticker, fair_price, 'market_making')

        if quantity <= 0:
            print(f"Calculated quantity is zero or negative for {ticker.name}. Skipping order placement.")
            return

        buy_price = fair_price - spread / 2
        sell_price = fair_price + spread / 2

        buy_order_id = self.place_order(
            side=Side.BUY,
            ticker=ticker,
            quantity=quantity,
            price=buy_price,
            ioc=False,
            strategy='market_making'
        )
        if buy_order_id:
            print(f"Placed MARKET MAKING BUY limit order at {buy_price} for {ticker.name} with order ID {buy_order_id}")

        sell_order_id = self.place_order(
            side=Side.SELL,
            ticker=ticker,
            quantity=quantity,
            price=sell_price,
            ioc=False,
            strategy='market_making'
        )
        if sell_order_id:
            print(
                f"Placed MARKET MAKING SELL limit order at {sell_price} for {ticker.name} with order ID {sell_order_id}")

    def calculate_dynamic_spread(self, ticker: Ticker, fair_price: float) -> float:
        """Calculate dynamic spread based on market volatility and current spread."""
        if len(self.price_history[ticker]) >= MM_WINDOW_SIZE:
            recent_prices = np.array(self.price_history[ticker][-MM_WINDOW_SIZE:])
            price_returns = np.diff(np.log(recent_prices))
            volatility = np.std(price_returns)
            spread = fair_price * MM_BASE_SPREAD_PCT * (1 + MM_VOLATILITY_FACTOR * volatility)
        else:
            spread = fair_price * MM_BASE_SPREAD_PCT

        # Adjust spread based on current market spread
        best_bid = max(self.order_book[ticker]['buy'].keys(), default=None)
        best_ask = min(self.order_book[ticker]['sell'].keys(), default=None)
        if best_bid is not None and best_ask is not None:
            current_market_spread = best_ask - best_bid
            spread = max(spread, current_market_spread)

        # Ensure spread is within bounds
        spread = min(max(spread, MM_MIN_SPREAD), MM_MAX_SPREAD)
        return spread

    def calculate_dynamic_quantity(self, ticker: Ticker, fair_price: float, strategy: str) -> float:
        """Calculate dynamic order quantity based on capital allocation and position limits."""
        if strategy == 'market_making':
            allocation_pct = MM_CAPITAL_ALLOCATION_PCT
        else:
            allocation_pct = 0.01  # Default to 1% if strategy is unknown

        max_trade_value = self.capital * allocation_pct

        quantity = max_trade_value / fair_price
        quantity = min(max(quantity, MM_BASE_QUANTITY), MM_MAX_QUANTITY)

        max_position = (self.capital * RISK_POSITION_LIMIT_PCT) / fair_price
        net_position = abs(self.positions.get(ticker, 0))

        if net_position + quantity > max_position:
            quantity = max(max_position - net_position, 0)

        quantity = int(quantity)
        return quantity

    def cancel_oldest_order(self) -> bool:
        """Cancel the oldest order."""
        if self.order_ids:
            oldest_order_id = next(iter(self.order_ids))
            details = self.order_ids[oldest_order_id]
            success = cancel_order(details['ticker'], oldest_order_id)
            if success:
                print(f"Canceled oldest order ID {oldest_order_id} for {details['ticker'].name}")
                del self.order_ids[oldest_order_id]
                return True
            else:
                print(f"Failed to cancel oldest order ID {oldest_order_id} for {details['ticker'].name}")
                return False
        else:
            print("No orders to cancel.")
            return False

    def place_order(
            self,
            side: Side,
            ticker: Ticker,
            quantity: float,
            price: float,
            ioc: bool,
            strategy: str,
            spread: Optional[int] = None
    ) -> Optional[int]:
        try:
            order_id = place_limit_order(side, ticker, quantity, price, ioc)
            if order_id != 0:
                self.order_ids[order_id] = {
                    'ticker': ticker,
                    'side': side,
                    'ioc': ioc,
                    'strategy': strategy
                }
                if spread is not None:
                    self.order_ids[order_id]['spread'] = spread
                return order_id
            else:
                print(f"Failed to place LIMIT order: {side.name} {ticker.name} {quantity} @ {price}")
                return None
        except Exception as e:
            print(f"Error placing LIMIT order: {e}")
            return None

    def place_market_order(self, side: Side, ticker: Ticker, quantity: float) -> bool:
        try:
            success = place_market_order(side, ticker, quantity)
            if success:
                print(f"Placed MARKET order: {side.name} {ticker.name} {quantity}")
                return True
            else:
                print(f"Failed to place MARKET order: {side.name} {ticker.name} {quantity}")
                return False
        except Exception as e:
            print(f"Error placing MARKET order: {e}")
            return False


class Prediction:
    """Class for price prediction using historical and current order book data."""

    def __init__(self, ticker: Ticker):
        self.ticker = ticker
        self.alpha: float = 0.2  # Decay Factor
        self.prices: List[float] = []
        self.soft_average: Optional[float] = None
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.book: List[float] = []
        self.price_history: List[float] = []
        self.kde: Optional[KernelDensity] = None  # Kernel Density Estimator

    def update(self, order_book: Dict[str, Dict[float, float]]) -> None:
        print("updating predictor")
        # Update bids and asks
        self.bids = order_book['buy']
        self.asks = order_book['sell']

        # Update book with all prices, weighted by volume
        self.book = []
        for price, volume in self.bids.items():
            self.book.extend([price] * int(volume))
        for price, volume in self.asks.items():
            self.book.extend([price] * int(volume))

        # Update soft average price
        current_price = self.predict_converge()
        self.prices.append(current_price)
        if self.soft_average is None:
            self.soft_average = current_price
        else:
            self.soft_average = (1 - self.alpha) * self.soft_average + self.alpha * current_price

        self.update_kde()

    def update_price_history(self, price: float) -> None:
        """Update the price history and KDE."""
        self.price_history.append(price)
        self.update_kde()

    def update_kde(self) -> None:
        """Update the Kernel Density Estimator with latest price history."""
        if len(self.price_history) >= 2:
            data = np.array(self.price_history).reshape(-1, 1)
            bandwidth = np.std(data) * (len(data) ** (-1 / 5))  # Scott's Rule
            self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)

    def predict_converge(self) -> float:
        """Predict the fair price using the Sliding Window Median algorithm."""
        if not self.book:
            return self.prices[-1] if self.prices else 0.0

        # Create a list of (price, type) where type is 0 for bid, 1 for ask
        orders = sorted(
            [(price, 0) for price in self.bids for _ in range(int(self.bids[price]))] +
            [(price, 1) for price in self.asks for _ in range(int(self.asks[price]))],
            key=lambda x: x[0]
        )

        # Build the parentheses string
        parentheses = ''.join('(' if order_type == 0 else ')' for _, order_type in orders)

        # Find the index that minimizes the difference between open and close parentheses
        min_diff = math.inf
        fair_price_index = -1
        open_count = 0
        close_count = parentheses.count(')')

        for i, char in enumerate(parentheses):
            if char == '(':
                open_count += 1
            else:
                close_count -= 1
            diff = abs(open_count - close_count)
            if diff < min_diff:
                min_diff = diff
                fair_price_index = i

        # Determine the fair price
        if fair_price_index >= 0 and fair_price_index < len(orders):
            fair_price = orders[fair_price_index][0]
        else:
            fair_price = self.prices[-1] if self.prices else 0.0

        return fair_price

    def predict(self) -> float:
        """Combine the current prediction with KDE estimation."""
        if self.kde is not None:
            # Use KDE to estimate most probable price
            grid = np.linspace(min(self.price_history), max(self.price_history), 1000).reshape(-1, 1)
            log_dens = self.kde.score_samples(grid)
            kde_price = grid[np.argmax(log_dens)][0]
            # Combine with soft average
            combined_price = 0.5 * self.soft_average + 0.5 * kde_price
            return combined_price
        else:
            return self.soft_average
