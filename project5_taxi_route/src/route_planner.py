"""RoutePlanner: composition. A planner HAS-A Route and HAS-A Forecaster."""

import numpy as np

from src.forecasters import Forecaster
from src.planning import backtest, plan_fleet
from src.route import Route


class RoutePlanner:
    """Combine one Route with one Forecaster to answer planning questions.

    This is *composition* ("has-a"): the planner owns a Route object and a
    Forecaster object and delegates work to them, instead of inheriting from them.
    Because it only needs something with a .forecast() method, any Forecaster
    subclass can be plugged in.
    """

    def __init__(self, route: Route, forecaster: Forecaster, first_day: int = 4) -> None:
        self.route = route
        self.forecaster = forecaster
        self.first_day = first_day

    def backtest_forecasts(self) -> np.ndarray:
        """Walk-forward forecasts (passengers) for first_day ... last day."""
        forecasts, _ = backtest(self.forecaster, self.route.passengers, self.first_day)
        return forecasts

    def backtest_mae(self) -> float:
        """Walk-forward mean absolute error, in passengers."""
        _, mae = backtest(self.forecaster, self.route.passengers, self.first_day)
        return mae

    def next_day_passengers(self) -> float:
        """Forecast passengers for the day after the last observed day."""
        return self.forecaster.forecast(self.route.passengers)

    def next_day_revenue(self) -> float:
        """Forecast revenue (UGX) for the next day = passengers x fare."""
        return self.next_day_passengers() * self.route.fare

    def vehicles_needed(self, **fleet_options) -> int:
        """Vehicles to deploy tomorrow (see plan_fleet for the options)."""
        return plan_fleet(self.next_day_passengers(), **fleet_options)[2]

    def __repr__(self) -> str:
        return f"RoutePlanner({self.route.name!r}, {self.forecaster})"
