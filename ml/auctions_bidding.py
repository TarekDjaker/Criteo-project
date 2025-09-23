"""
Auctions & Bidding System - First-Price Focus
Critical for Criteo: Industry moved to first-price in 2019
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class AuctionType(Enum):
    """Auction types in programmatic advertising"""
    FIRST_PRICE = "first_price"
    SECOND_PRICE = "second_price"
    HYBRID = "hybrid"


@dataclass
class BidRequest:
    """Real-time bid request"""
    request_id: str
    user_id: str
    publisher_id: str
    ad_slot_id: str
    user_features: Dict
    context_features: Dict
    floor_price: float
    auction_type: AuctionType


@dataclass
class BidResponse:
    """Bid response with shading"""
    request_id: str
    bid_price: float
    shaded_bid: float  # After bid shading
    expected_value: float
    win_probability: float
    expected_profit: float


class FirstPriceBidder:
    """
    First-price auction bidding strategy
    Key innovation at Criteo post-2019
    """

    def __init__(self, margin_target: float = 0.3):
        self.margin_target = margin_target
        self.shading_model = None
        self.win_rate_history = []
        self.clearing_price_history = []

    def calculate_bid_value(self, pCTR: float, pCVR: float, conversion_value: float) -> float:
        """
        Calculate bid value based on expected value
        Real formula: bid = pCTR * pCVR * conversion_value * margin
        """
        expected_value = pCTR * pCVR * conversion_value
        bid_value = expected_value * (1 - self.margin_target)
        return bid_value

    def predict_clearing_price(self, features: Dict) -> Tuple[float, float]:
        """
        Predict the clearing price distribution
        Critical for first-price optimization
        """
        # Simplified model - real would use ML
        base_price = features.get('floor_price', 0.5)
        competition = features.get('competition_level', 1.0)

        # Predict mean and std of clearing price
        mean_price = base_price * (1 + 0.5 * competition)
        std_price = mean_price * 0.3

        return mean_price, std_price

    def optimal_shading_factor(self, win_prob_curve: callable, margin: float) -> float:
        """
        Calculate optimal bid shading factor
        Maximizes expected profit: P(win) * (value - bid)
        """
        # Grid search for optimal shading
        best_shading = 1.0
        best_profit = 0

        for shading in np.linspace(0.5, 1.0, 50):
            win_prob = win_prob_curve(shading)
            profit = win_prob * margin * shading
            if profit > best_profit:
                best_profit = profit
                best_shading = shading

        return best_shading

    def shade_bid(self, original_bid: float, features: Dict) -> float:
        """
        Apply bid shading for first-price auctions
        Key technique to avoid overpaying
        """
        # Predict clearing price
        mean_clear, std_clear = self.predict_clearing_price(features)

        # Calculate win probability curve
        def win_prob(bid_multiplier):
            bid = original_bid * bid_multiplier
            z_score = (bid - mean_clear) / (std_clear + 1e-10)
            # CDF of normal distribution
            return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

        # Find optimal shading
        shading_factor = self.optimal_shading_factor(win_prob, self.margin_target)

        # Apply shading
        shaded_bid = original_bid * shading_factor

        # Ensure above floor
        floor = features.get('floor_price', 0)
        shaded_bid = max(shaded_bid, floor * 1.01)

        return shaded_bid

    def bid(self, request: BidRequest, pCTR: float, pCVR: float,
           conversion_value: float) -> BidResponse:
        """
        Generate bid response with shading
        """
        # Calculate base bid value
        bid_value = self.calculate_bid_value(pCTR, pCVR, conversion_value)

        # Apply shading for first-price
        if request.auction_type == AuctionType.FIRST_PRICE:
            shaded_bid = self.shade_bid(bid_value, {
                'floor_price': request.floor_price,
                'competition_level': len(request.context_features.get('competitors', [])) / 10
            })
        else:
            # Second-price: bid true value
            shaded_bid = bid_value

        # Calculate win probability
        mean_clear, std_clear = self.predict_clearing_price({
            'floor_price': request.floor_price,
            'competition_level': len(request.context_features.get('competitors', [])) / 10
        })
        z_score = (shaded_bid - mean_clear) / (std_clear + 1e-10)
        win_prob = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

        # Expected profit
        expected_value = pCTR * pCVR * conversion_value
        expected_profit = win_prob * (expected_value - shaded_bid)

        return BidResponse(
            request_id=request.request_id,
            bid_price=bid_value,
            shaded_bid=shaded_bid,
            expected_value=expected_value,
            win_probability=win_prob,
            expected_profit=expected_profit
        )


class BudgetPacer:
    """
    Budget pacing to ensure smooth spending
    Critical for campaign performance
    """

    def __init__(self, daily_budget: float, campaign_duration_hours: int = 24):
        self.daily_budget = daily_budget
        self.campaign_duration = campaign_duration_hours
        self.spent = 0
        self.hours_elapsed = 0
        self.spending_history = []

    def get_pacing_multiplier(self) -> float:
        """
        Calculate pacing multiplier based on spend rate
        """
        if self.hours_elapsed == 0:
            return 1.0

        # Calculate ideal spend rate
        ideal_hourly_spend = self.daily_budget / self.campaign_duration
        ideal_total_spend = ideal_hourly_spend * self.hours_elapsed

        # Compare with actual spend
        if self.spent > ideal_total_spend * 1.1:  # Overspending
            return 0.8  # Slow down
        elif self.spent < ideal_total_spend * 0.9:  # Underspending
            return 1.2  # Speed up
        else:
            return 1.0  # On track

    def should_bid(self, bid_amount: float) -> bool:
        """
        Decide whether to participate in auction
        """
        if self.spent + bid_amount > self.daily_budget:
            return False

        # Probabilistic pacing
        pacing_mult = self.get_pacing_multiplier()
        participate_prob = min(1.0, pacing_mult)

        return np.random.random() < participate_prob

    def record_win(self, clearing_price: float):
        """Record auction win and update spend"""
        self.spent += clearing_price
        self.spending_history.append({
            'hour': self.hours_elapsed,
            'spent': self.spent,
            'price': clearing_price
        })

    def update_hour(self):
        """Update hour for pacing calculation"""
        self.hours_elapsed += 1


class HeaderBidding:
    """
    Header bidding implementation
    Criteo's approach to publisher yield optimization
    """

    def __init__(self):
        self.bidders = []
        self.auction_history = []

    def add_bidder(self, bidder_id: str, bidder: FirstPriceBidder):
        """Add bidder to header auction"""
        self.bidders.append({'id': bidder_id, 'bidder': bidder})

    def run_auction(self, request: BidRequest, bidder_contexts: Dict) -> Dict:
        """
        Run header bidding auction
        All bidders submit simultaneously
        """
        bids = []

        # Collect bids from all bidders
        for bidder_info in self.bidders:
            bidder_id = bidder_info['id']
            bidder = bidder_info['bidder']

            if bidder_id in bidder_contexts:
                context = bidder_contexts[bidder_id]
                bid_response = bidder.bid(
                    request,
                    context['pCTR'],
                    context['pCVR'],
                    context['conversion_value']
                )
                bids.append({
                    'bidder_id': bidder_id,
                    'bid': bid_response.shaded_bid,
                    'response': bid_response
                })

        # Determine winner (first-price)
        if not bids:
            return {'winner': None, 'clearing_price': 0, 'bids': []}

        bids.sort(key=lambda x: x['bid'], reverse=True)
        winner = bids[0]

        # First-price: pay what you bid
        clearing_price = winner['bid']

        result = {
            'winner': winner['bidder_id'],
            'clearing_price': clearing_price,
            'winning_bid': winner['bid'],
            'second_bid': bids[1]['bid'] if len(bids) > 1 else request.floor_price,
            'all_bids': bids
        }

        self.auction_history.append(result)
        return result


class BidLandscapeEstimator:
    """
    Estimate competitive landscape for better bidding
    """

    def __init__(self):
        self.observations = []

    def update(self, features: Dict, bid: float, won: bool, clearing_price: float = None):
        """Update model with auction outcome"""
        self.observations.append({
            'features': features,
            'bid': bid,
            'won': won,
            'clearing_price': clearing_price if won else None
        })

    def estimate_win_rate_curve(self, features: Dict) -> callable:
        """
        Estimate win rate as function of bid
        Real: Uses logistic regression or XGBoost
        """
        # Simplified: use historical data
        relevant_obs = [obs for obs in self.observations
                       if self._similar_context(obs['features'], features)]

        if len(relevant_obs) < 10:
            # Fallback to simple model
            def win_rate(bid):
                return 1 / (1 + np.exp(-(bid - 1.0)))
            return win_rate

        # Fit curve from observations
        bids = [obs['bid'] for obs in relevant_obs]
        wins = [1 if obs['won'] else 0 for obs in relevant_obs]

        # Simple logistic fit
        from scipy.optimize import curve_fit

        def logistic(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))

        try:
            popt, _ = curve_fit(logistic, bids, wins, p0=[1, 1, np.median(bids)])
            def win_rate(bid):
                return logistic(bid, *popt)
        except:
            def win_rate(bid):
                return 1 / (1 + np.exp(-(bid - 1.0)))

        return win_rate

    def _similar_context(self, context1: Dict, context2: Dict) -> bool:
        """Check if contexts are similar"""
        # Simplified: check key features
        key_features = ['publisher_id', 'ad_slot_id']
        for key in key_features:
            if context1.get(key) != context2.get(key):
                return False
        return True


def demonstrate_bidding_system():
    """
    Complete demonstration of bidding system
    """
    print("=== First-Price Bidding System Demo ===\n")

    # 1. Initialize bidders
    print("Setting up bidders...")
    criteo_bidder = FirstPriceBidder(margin_target=0.3)
    competitor_bidder = FirstPriceBidder(margin_target=0.25)

    # 2. Create header bidding system
    header_system = HeaderBidding()
    header_system.add_bidder('criteo', criteo_bidder)
    header_system.add_bidder('competitor', competitor_bidder)

    # 3. Simulate auctions
    print("\nRunning 100 auctions...")
    results = []

    for i in range(100):
        # Create bid request
        request = BidRequest(
            request_id=f"req_{i}",
            user_id=f"user_{i % 10}",
            publisher_id=f"pub_{i % 5}",
            ad_slot_id=f"slot_{i % 3}",
            user_features={},
            context_features={'competitors': list(range(np.random.randint(1, 10)))},
            floor_price=np.random.uniform(0.1, 1.0),
            auction_type=AuctionType.FIRST_PRICE
        )

        # Bidder contexts (predictions)
        contexts = {
            'criteo': {
                'pCTR': np.random.uniform(0.01, 0.1),
                'pCVR': np.random.uniform(0.01, 0.05),
                'conversion_value': np.random.uniform(5, 50)
            },
            'competitor': {
                'pCTR': np.random.uniform(0.01, 0.1),
                'pCVR': np.random.uniform(0.01, 0.05),
                'conversion_value': np.random.uniform(5, 50)
            }
        }

        # Run auction
        result = header_system.run_auction(request, contexts)
        results.append(result)

    # 4. Analyze results
    print("\n=== Auction Results Analysis ===")
    wins_by_bidder = pd.DataFrame(results)['winner'].value_counts()
    print(f"\nWins by bidder:")
    print(wins_by_bidder)

    criteo_wins = [r for r in results if r['winner'] == 'criteo']
    if criteo_wins:
        avg_clearing = np.mean([r['clearing_price'] for r in criteo_wins])
        avg_margin = np.mean([r['clearing_price'] - r['second_bid'] for r in criteo_wins])
        print(f"\nCriteo stats:")
        print(f"  Average clearing price: ${avg_clearing:.3f}")
        print(f"  Average margin over second: ${avg_margin:.3f}")

    # 5. Demonstrate bid shading impact
    print("\n=== Bid Shading Analysis ===")
    test_features = {'floor_price': 1.0, 'competition_level': 0.5}

    # Without shading
    original_bid = 5.0

    # With shading
    shaded_bid = criteo_bidder.shade_bid(original_bid, test_features)

    print(f"Original bid: ${original_bid:.2f}")
    print(f"Shaded bid: ${shaded_bid:.2f}")
    print(f"Savings: ${original_bid - shaded_bid:.2f} ({(1 - shaded_bid/original_bid)*100:.1f}%)")

    # 6. Budget pacing demo
    print("\n=== Budget Pacing Demo ===")
    pacer = BudgetPacer(daily_budget=1000, campaign_duration_hours=24)

    for hour in range(24):
        pacer.update_hour()
        hourly_wins = np.random.randint(10, 50)
        for _ in range(hourly_wins):
            bid_amount = np.random.uniform(1, 5)
            if pacer.should_bid(bid_amount):
                if np.random.random() < 0.3:  # 30% win rate
                    pacer.record_win(bid_amount)

    print(f"Final spend: ${pacer.spent:.2f} / ${pacer.daily_budget:.2f}")
    print(f"Utilization: {pacer.spent/pacer.daily_budget*100:.1f}%")

    # 7. Key insights
    print("\n=== Key Insights for Interview ===")
    print("1. First-price requires bid shading to avoid overpaying")
    print("2. Optimal shading depends on competition prediction")
    print("3. Header bidding allows simultaneous bidding")
    print("4. Budget pacing ensures smooth spending")
    print("5. Win rate estimation is critical for optimization")

    return header_system


if __name__ == "__main__":
    # Run demonstration
    system = demonstrate_bidding_system()

    print("\n✅ Bidding module ready!")
    print("\nRemember for interview:")
    print("- Industry moved to first-price in 2019")
    print("- Criteo was early adopter of bid shading")
    print("- Header bidding is Criteo innovation")
    print("- Real-time optimization at 1M+ QPS")