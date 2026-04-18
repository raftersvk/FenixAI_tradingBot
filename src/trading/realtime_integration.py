# src/trading/realtime_integration.py
"""
Real-time data integration with LangGraph.

Connects MarketDataManager with FenixTradingGraph for
continuous analysis and live trading signals.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable

from src.trading.market_data import MarketDataManager, MicrostructureMetrics
from src.core.langgraph_orchestrator import FenixTradingGraph, FenixAgentState
from config.llm_provider_config import LLMProvidersConfig, AgentProviderConfig

logger = logging.getLogger(__name__)


class RealtimeTradingPipeline:
    """
    Real-time trading pipeline.

    Connects:
    1. MarketDataManager (WebSocket) → live data
    2. FenixTradingGraph (LangGraph) → multi-agent analysis
    3. Callbacks → signal execution

    Flow:
    - Receives closed klines via WebSocket
    - Accumulates calculated indicators
    - Executes agents graph
    - Emits trading signals
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        llm_config: LLMProvidersConfig | None = None,
        use_testnet: bool = True,
        min_interval_seconds: int = 60,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_interval = min_interval_seconds
        self._last_analysis_time: datetime | None = None

        # Initialize components
        self.market_data = MarketDataManager(
            symbol=symbol,
            timeframe=timeframe,
            use_testnet=use_testnet,
        )

        # If no llm_config is provided, try to load from LLM Provider Loader (project-level config)
        if llm_config is None:
            try:
                from src.config.llm_provider_loader import get_provider_loader

                loader = get_provider_loader()
                llm_config = loader.get_config()
                logger.info(
                    f"RealtimeTradingPipeline: Using LLM config from provider loader ({loader.active_profile})"
                )
            except Exception:
                logger.warning(
                    "RealtimeTradingPipeline: Provider loader not available, using Ollama defaults"
                )
                llm_config = self._create_default_ollama_config()

        self.trading_graph = FenixTradingGraph(
            llm_config=llm_config,
            enable_visual=False,  # Disabled in real-time due to latency
            enable_sentiment=True,
            enable_risk=True,
        )

        # Accumulated state
        self._indicators: dict[str, Any] = {}
        self._kline_history: list[dict] = []
        self._microstructure: MicrostructureMetrics | None = None

        # Callbacks
        self._signal_callbacks: list[Callable[[FenixAgentState], None]] = []

        # Control
        self._running = False
        self._analysis_task: asyncio.Task | None = None

        logger.info(f"RealtimeTradingPipeline initialized: {symbol}@{timeframe}")

    def _create_default_ollama_config(self) -> LLMProvidersConfig:
        """Creates default configuration using local Ollama."""
        ollama_fast = AgentProviderConfig(
            provider_type="ollama_local",
            model_name="gemma3:1b",  # Fast for real-time
            temperature=0.2,
            max_tokens=1000,
        )
        
        ollama_reasoning = AgentProviderConfig(
            provider_type="ollama_local",
            model_name="qwen2.5:7b",  # Better reasoning
            temperature=0.3,
            max_tokens=2000,
        )

        return LLMProvidersConfig(
            technical=ollama_fast,
            sentiment=ollama_fast,
            visual=ollama_fast,
            qabba=ollama_fast,
            decision=ollama_reasoning,
            risk=ollama_fast,
        )

    def on_signal(self, callback: Callable[[FenixAgentState], None]) -> None:
        """Registers callback for new trading signals."""
        self._signal_callbacks.append(callback)

    async def start(self) -> None:
        """Starts the complete pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return

        self._running = True

        # Register callbacks in MarketDataManager
        self.market_data.on_kline(self._on_kline_closed)
        self.market_data.on_microstructure_update(self._on_microstructure_update)
        
        # Start MarketDataManager
        await self.market_data.start()

        logger.info("RealtimeTradingPipeline started")

    async def stop(self) -> None:
        """Stops the pipeline."""
        self._running = False

        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass

        await self.market_data.stop()
        logger.info("RealtimeTradingPipeline stopped")

    async def _on_kline_closed(self, kline_data: dict) -> None:
        """Callback when a kline is closed."""
        # Save to history
        self._kline_history.append(kline_data)
        if len(self._kline_history) > 100:
            self._kline_history = self._kline_history[-100:]

        # Calculate basic indicators (TODO: use ta-lib if available)
        self._update_indicators(kline_data)

        # Check if we should analyze
        if self._should_analyze():
            await self._run_analysis()

    def _on_microstructure_update(self, metrics: MicrostructureMetrics) -> None:
        """Callback for microstructure updates."""
        self._microstructure = metrics
        self._indicators["obi"] = metrics.obi
        self._indicators["cvd"] = metrics.cvd
        self._indicators["spread"] = metrics.spread

    def _update_indicators(self, kline: dict) -> None:
        """Updates indicators with new kline."""
        # Current price and volume
        self._indicators["current_price"] = float(kline.get("close", 0))
        self._indicators["current_volume"] = float(kline.get("volume", 0))
        
        # Simple RSI (14 periods)
        if len(self._kline_history) >= 14:
            closes = [float(k.get("close", 0)) for k in self._kline_history[-15:]]
            gains = [max(0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
            losses = [max(0, closes[i - 1] - closes[i]) for i in range(1, len(closes))]

            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0.001

            rs = avg_gain / avg_loss
            self._indicators["rsi"] = 100 - (100 / (1 + rs))

        # EMAs simples
        if len(self._kline_history) >= 21:
            closes = [float(k.get("close", 0)) for k in self._kline_history[-21:]]
            self._indicators["ema_9"] = sum(closes[-9:]) / 9
            self._indicators["ema_21"] = sum(closes) / 21

    def _should_analyze(self) -> bool:
        """Determines if it's time to run analysis."""
        if not self._running:
            return False

        now = datetime.now()

        if self._last_analysis_time is None:
            return True

        elapsed = (now - self._last_analysis_time).total_seconds()
        return elapsed >= self.min_interval

    async def _run_analysis(self) -> None:
        """Executes the complete analysis pipeline."""
        self._last_analysis_time = datetime.now()
        
        try:
            logger.info(f"Running analysis for {self.symbol}...")
            
            # Execute agents graph
            result = await self.trading_graph.ainvoke(
                symbol=self.symbol,
                timeframe=self.timeframe,
                indicators=self._indicators.copy(),
                current_price=self._indicators.get("current_price", 0),
                current_volume=self._indicators.get("current_volume", 0),
                obi=self._indicators.get("obi", 1.0),
                cvd=self._indicators.get("cvd", 0.0),
                spread=self._indicators.get("spread", 0.01),
            )

            # Emit signal to all callbacks
            for callback in self._signal_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in signal callback: {e}")

            # Log del resultado
            decision = result.get("final_trade_decision", {})
            risk = result.get("risk_assessment", {})

            logger.info(
                f"Analysis complete: "
                f"Decision={decision.get('final_decision', 'N/A')} | "
                f"Risk={risk.get('verdict', 'N/A')} | "
                f"Price={self._indicators.get('current_price', 0):.2f}"
            )

        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")

    async def force_analysis(self) -> FenixAgentState:
        """Forces an immediate analysis."""
        await self._run_analysis()
        return self.get_last_state()
    
    def get_last_state(self) -> dict:
        """Returns the last known state."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "indicators": self._indicators.copy(),
            "microstructure": self._microstructure,
            "last_analysis": self._last_analysis_time,
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


async def example_usage():
    """Example usage of the real-time pipeline."""
    
    def on_signal(state: FenixAgentState):
        decision = state.get("final_trade_decision", {})
        action = decision.get("final_decision", "HOLD")
        confidence = decision.get("confidence_in_decision", "N/A")

        risk = state.get("risk_assessment", {})
        verdict = risk.get("verdict", "N/A")

        print(f"🚀 SIGNAL: {action} | Confidence: {confidence} | Risk: {verdict}")

    # Crear pipeline
    pipeline = RealtimeTradingPipeline(
        symbol="BTCUSDT",
        timeframe="15m",
        use_testnet=True,
        min_interval_seconds=60,
    )

    # Registrar callback
    pipeline.on_signal(on_signal)

    # Iniciar
    await pipeline.start()

    try:
        # Correr por 5 minutos
        await asyncio.sleep(300)
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
