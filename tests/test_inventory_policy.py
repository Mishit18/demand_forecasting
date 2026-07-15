from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR / "src"))

from inventory_policy import reorder_point, safety_stock  # noqa: E402


def test_safety_stock_formula():
    assert round(safety_stock(error_sigma=100, service_level=0.95, lead_time_days=4), 2) == 329.0


def test_reorder_point_formula():
    assert round(reorder_point(average_daily_demand=500, error_sigma=100, service_level=0.95, lead_time_days=4), 2) == 2329.0
