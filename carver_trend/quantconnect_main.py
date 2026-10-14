# region imports
from AlgorithmImports import *
from universe import AdvancedFuturesUniverseSelectionModel
from alpha import AdjustedTrendAlphaModel
from portfolio import BufferedPortfolioConstructionModel
# endregion


class AdjustedTrendFuturesAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(self.end_date - timedelta(5*365))
        self.set_cash(1_000_000)

        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.seed_initial_prices = True
        self.settings.minimum_order_margin_portfolio_percentage = 0

        self.universe_settings.data_normalization_mode = DataNormalizationMode.BACKWARDS_PANAMA_CANAL
        self.universe_settings.data_mapping_mode = DataMappingMode.OPEN_INTEREST
        self.add_universe_selection(AdvancedFuturesUniverseSelectionModel())

        emac_filters = self.get_parameter("emac_filters", 6)
        blend_years = self.get_parameter("blend_years", 3)        # Number of years to use when blending sigma estimates
        self.add_alpha(AdjustedTrendAlphaModel(
            self,
            emac_filters, 
            self.get_parameter("abs_forecast_cap", 20),           # Hardcoded on p.173
            self.get_parameter("sigma_span", 32),                 # Hardcoded to 32 on p.604
            self.get_parameter("target_risk", 0.2),               # Recommend value is 0.2 on p.75
            blend_years
        ))

        self.settings.rebalance_portfolio_on_security_changes = False
        self.settings.rebalance_portfolio_on_insight_changes = False
        self._total_count = 0
        self._day = -1
        self.set_portfolio_construction(BufferedPortfolioConstructionModel(
            self._rebalance_func,
            self.get_parameter("buffer_scaler", 0.1)              # Hardcoded on p.167 & p.173
        ))

        self.add_risk_management(NullRiskManagementModel())

        self.set_execution(ImmediateExecutionModel())

        self.set_warm_up(timedelta(max(365*blend_years, 4*2**emac_filters) + 7))

    def _rebalance_func(self, time):
        if ((self._total_count != self.insights.total_count or self._day != self.time.day) and 
            not self.is_warming_up and 
            self.current_slice.quote_bars.count > 0):
            self._total_count = self.insights.total_count
            self._day = self.time.day
            return time
        return None
