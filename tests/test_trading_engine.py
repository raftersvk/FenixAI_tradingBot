"""
Tests para el Trading Engine.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestTradingConfig:
    """Tests para TradingConfig."""

    def test_default_config(self):
        """Verificar configuración por defecto."""
        from src.trading.engine import TradingConfig

        config = TradingConfig()

        assert config.symbol == "BTCUSDT"
        assert config.interval == "15m"
        assert config.testnet is True
        assert config.dry_run is False

    def test_custom_config(self):
        """Verificar configuración personalizada."""
        from src.trading.engine import TradingConfig

        config = TradingConfig(
            symbol="ETHUSDT",
            interval="5m",
            max_risk_per_trade=1.5,
        )

        assert config.symbol == "ETHUSDT"
        assert config.interval == "5m"
        assert config.max_risk_per_trade == 1.5


@pytest.fixture
def mock_unified_config():
    """Mock config for TradingEngine tests."""
    from src.config.unified_loader import (
        FenixConfig,
        TradingConfig,
        AgentsConfig,
        LLMConfig,
        BinanceConfig,
        LoggingConfig,
        ResilienceConfig,
    )

    return FenixConfig(
        trading=TradingConfig(
            symbol="BTCUSDT",
            timeframe="15m",
            min_klines_to_start=20,
            max_risk_per_trade=0.02,
        ),
        agents=AgentsConfig(enable_visual=True, enable_sentiment=True),
        llm=LLMConfig(),
        binance=BinanceConfig(testnet=True),
        logging=LoggingConfig(),
        resilience=ResilienceConfig(),
    )


class TestTradingEngine:
    """Tests para TradingEngine."""

    @pytest.fixture
    def trading_engine(self, mock_unified_config):
        """Crear instancia del trading engine."""
        from src.trading.engine import TradingEngine

        with patch("src.trading.engine.get_config", return_value=mock_unified_config):
            engine = TradingEngine(
                symbol="BTCUSDT",
                timeframe="15m",
                use_testnet=True,
                paper_trading=True,
            )
            engine._init_components = MagicMock()
            return engine

    def test_engine_initialization(self, trading_engine):
        """Verificar inicialización del engine."""
        assert trading_engine.symbol == "BTCUSDT"
        assert trading_engine.timeframe == "15m"
        assert trading_engine.paper_trading is True

    def test_engine_not_running_initially(self, trading_engine):
        """Verificar que el engine no está corriendo inicialmente."""
        assert trading_engine._running is False

    def test_min_klines_requirement(self, trading_engine, mock_unified_config):
        """Verificar requisito mínimo de klines."""
        assert (
            trading_engine._min_klines_to_start == mock_unified_config.trading.min_klines_to_start
        )
        assert trading_engine._kline_count == 0

    @pytest.mark.asyncio
    async def test_engine_initialize(self, trading_engine):
        """Verificar inicialización async del engine."""
        with patch.object(trading_engine, "_trading_graph", MagicMock()):
            result = await trading_engine.initialize()
            assert isinstance(result, bool)


class TestTradingEngineSignalLogging:
    """Tests para logging de señales."""

    def test_signal_log_path_exists(self, mock_unified_config):
        """Verificar que se crea el path de logs."""
        from src.trading.engine import TradingEngine

        with patch("src.trading.engine.get_config", return_value=mock_unified_config):
            engine = TradingEngine(
                symbol="BTCUSDT",
                timeframe="15m",
            )
            engine._init_components = MagicMock()

        assert engine.signal_log_path.parent.exists()


class TestTradingEngineSafety:
    """Tests de seguridad del trading engine."""

    def test_paper_trading_default(self, mock_unified_config):
        """Verificar que paper trading es el default."""
        from src.trading.engine import TradingEngine

        with patch("src.trading.engine.get_config", return_value=mock_unified_config):
            engine = TradingEngine(symbol="BTCUSDT", timeframe="15m")
            engine._init_components = MagicMock()

        assert engine.paper_trading is True

    def test_live_trading_requires_flag(self, mock_unified_config):
        """Verificar que live trading requiere flag explícito."""
        from src.trading.engine import TradingEngine

        with patch("src.trading.engine.get_config", return_value=mock_unified_config):
            engine = TradingEngine(
                symbol="BTCUSDT",
                timeframe="15m",
                paper_trading=False,
                allow_live_trading=False,
            )
            engine._init_components = MagicMock()

        assert engine.allow_live_trading is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
