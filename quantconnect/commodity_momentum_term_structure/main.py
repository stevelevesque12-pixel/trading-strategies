# region imports
from AlgorithmImports import *

from selection import ContractQuote, annualized_roll_yield, pick_roll_pair, select_targets
# endregion
# https://quantpedia.com/Screener/Details/23
# Fuertes, Miffre & Rallis (2010), "Tactical allocation in commodity futures
# markets: Combining momentum and term structure signals".


class CommodityMomentumCombinedWithTermStructureAlgorithm(QCAlgorithm):

    # Momentum = mean daily return over this many days (21 ~ 1 month, as in the
    # QuantConnect library version; the paper also tests longer windows).
    MOMENTUM_DAYS = 21
    # Gross weight per leg (long leg sums to +5%, short leg to -5%).
    GROSS_PER_SIDE = 0.05
    # The traded (deferred) contract must outlive the next monthly rebalance.
    MIN_DAYS_TO_EXPIRY = 35
    # Wide enough that bi-monthly / quarterly contract cycles (cocoa, metals,
    # grains) still have a deferred contract past MIN_DAYS_TO_EXPIRY.
    MAX_DAYS_TO_EXPIRY = 180

    def initialize(self):
        self.set_start_date(self.end_date - timedelta(5*365))
        self.settings.seed_initial_prices = True
        self.set_cash(10_000_000)
        tickers = [
            Futures.Softs.COCOA,
            Futures.Softs.COFFEE,
            Futures.Grains.CORN,
            Futures.Softs.COTTON_2,
            Futures.Grains.OATS,
            Futures.Softs.ORANGE_JUICE,
            Futures.Grains.SOYBEAN_MEAL,
            Futures.Grains.SOYBEAN_OIL,
            Futures.Grains.SOYBEANS,
            Futures.Softs.SUGAR_11,
            Futures.Grains.WHEAT,
            Futures.Meats.FEEDER_CATTLE,
            Futures.Meats.LEAN_HOGS,
            Futures.Meats.LIVE_CATTLE,
            Futures.Energies.CRUDE_OIL_WTI,
            Futures.Energies.HEATING_OIL,
            Futures.Energies.NATURAL_GAS,
            Futures.Energies.GASOLINE,
            Futures.Metals.MICRO_GOLD,
            Futures.Metals.PALLADIUM,
            Futures.Metals.PLATINUM,
            Futures.Metals.SILVER
        ]
        for ticker in tickers:
            future = self.add_future(ticker)
            future.set_filter(timedelta(0), timedelta(self.MAX_DAYS_TO_EXPIRY))
        self.schedule.on(self.date_rules.month_start("SPY"), self.time_rules.after_market_open("SPY", 30), self._rebalance)

    def _rebalance(self):
        # One entry per commodity: roll yield from the nearest vs the next
        # contract, and momentum of the deferred contract we would trade.
        roll_yield = {}
        momentum = {}
        for _, chain in self.current_slice.future_chains.items():
            quotes = [
                ContractQuote(c.symbol, c.expiry, self.securities[c.symbol].price, c.open_interest or 0)
                for c in chain
            ]
            pair = pick_roll_pair(quotes, self.time, self.MIN_DAYS_TO_EXPIRY)
            if pair is None:
                continue
            nearest, deferred = pair
            indicator = getattr(self.securities[deferred.key], "mean_return", None)
            if indicator is None or not indicator.is_ready:
                continue
            roll_yield[deferred.key] = annualized_roll_yield(
                nearest.price, deferred.price, (deferred.expiry - nearest.expiry).days
            )
            momentum[deferred.key] = indicator.current.value

        targets = select_targets(roll_yield, momentum, self.GROSS_PER_SIDE)
        if not targets:
            self.liquidate()
            return
        self.set_holdings([PortfolioTarget(symbol, weight) for symbol, weight in targets.items()], True)

    def on_securities_changed(self, changes):
        # Add an indicator to each contract to track its mean daily return.
        for security in changes.added_securities:
            roc = self.roc(security, 1, Resolution.DAILY)
            security.mean_return = IndicatorExtensions.sma(roc, self.MOMENTUM_DAYS)
            for bar in self.history[TradeBar](security, self.MOMENTUM_DAYS + 1, Resolution.DAILY):
                roc.update(bar)
