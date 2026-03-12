"""
Options Trading Platform - Backend API Server

FastAPI-based backend server providing real-time options data,
pricing services, and market making functionality.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uvicorn
from pydantic import BaseModel
import numpy as np
import pandas as pd

# Import our trading system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python_api', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'market_data', 'src'))

from pricing_engine import PricingEngine, OptionContract, MarketData, OptionType, ExerciseType
from market_maker import MarketMaker, MarketMakerConfig, Quote
from data_provider import SimulatedDataProvider, RealTimeDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="SIG Options Trading Platform", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
connected_clients: List[WebSocket] = []
market_maker: Optional[MarketMaker] = None
data_manager: Optional[RealTimeDataManager] = None
pricing_engine = PricingEngine()

# Pydantic models for API
class OptionPriceRequest(BaseModel):
    symbol: str
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry_days: int
    spot_price: float
    volatility: float
    risk_free_rate: float = 0.02
    dividend_yield: float = 0.0

class QuoteRequest(BaseModel):
    symbol: str
    option_type: str
    strike: float
    expiry_days: int

class PortfolioPosition(BaseModel):
    symbol: str
    option_type: str
    strike: float
    expiry_days: int
    quantity: int
    entry_price: float

class RiskLimits(BaseModel):
    max_delta: float = 1000.0
    max_gamma: float = 500.0
    max_vega: float = 10000.0
    max_position_size: int = 100

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the trading system"""
    global market_maker, data_manager

    logger.info("Starting Options Trading Platform...")

    # Initialize market maker
    config = MarketMakerConfig(
        max_position_size=100,
        max_portfolio_delta=1000.0,
        max_portfolio_gamma=500.0,
        min_spread_bps=10.0,
        edge_target=0.005
    )
    market_maker = MarketMaker(config)
    market_maker.start()

    # Initialize data provider and manager
    data_provider = SimulatedDataProvider(volatility=0.25, update_interval=1.0)
    data_manager = RealTimeDataManager(data_provider)
    await data_manager.start()

    # Subscribe to data feeds
    await data_manager.subscribe_to_options_chain("SPY", handle_options_chain_update)
    await data_manager.subscribe_to_options_chain("QQQ", handle_options_chain_update)

    logger.info("Trading platform initialized successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if market_maker:
        market_maker.stop()
    if data_manager:
        await data_manager.stop()
    logger.info("Trading platform shutdown complete")

# API Routes

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard"""
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "market_maker_running": market_maker.is_running if market_maker else False,
        "connected_clients": len(manager.active_connections)
    }

@app.post("/api/price")
async def price_option(request: OptionPriceRequest):
    """Price an option using the pricing engine"""
    try:
        # Create option contract
        option = OptionContract(
            symbol=request.symbol,
            option_type=OptionType.CALL if request.option_type.upper() == 'CALL' else OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=request.strike,
            expiry=request.expiry_days / 365.0,
            underlying=request.symbol
        )

        # Create market data
        market_data = MarketData(
            spot_price=request.spot_price,
            risk_free_rate=request.risk_free_rate,
            dividend_yield=request.dividend_yield,
            volatility=request.volatility,
            time_to_expiry=request.expiry_days / 365.0
        )

        # Price the option
        result = pricing_engine.price_option(option, market_data)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        return {
            "symbol": request.symbol,
            "price": result.price,
            "greeks": result.greeks.to_dict(),
            "success": True
        }

    except Exception as e:
        logger.error(f"Error pricing option: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quote")
async def get_market_maker_quote(request: QuoteRequest):
    """Get market maker quote for an option"""
    try:
        if not market_maker:
            raise HTTPException(status_code=503, detail="Market maker not available")

        # Create option contract
        option = OptionContract(
            symbol=f"{request.symbol}_{request.expiry_days}D_{request.option_type}_{request.strike}",
            option_type=OptionType.CALL if request.option_type.upper() == 'CALL' else OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=request.strike,
            expiry=request.expiry_days / 365.0,
            underlying=request.symbol
        )

        # Use current market data (simplified)
        market_data = MarketData(
            spot_price=400.0,  # Mock spot price
            risk_free_rate=0.02,
            dividend_yield=0.0,
            volatility=0.25,
            time_to_expiry=request.expiry_days / 365.0
        )

        # Generate quote
        quote = market_maker.generate_quote(option, market_data)

        if not quote:
            raise HTTPException(status_code=400, detail="Unable to generate quote")

        return {
            "symbol": quote.symbol,
            "bid_price": quote.bid_price,
            "ask_price": quote.ask_price,
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "theoretical_value": quote.theoretical_value,
            "spread_bps": quote.spread_bps,
            "timestamp": quote.timestamp
        }

    except Exception as e:
        logger.error(f"Error generating quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio summary"""
    try:
        if not market_maker:
            return {"positions": [], "total_pnl": 0.0, "greeks": {}}

        summary = market_maker.get_portfolio_summary()
        return summary

    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/risk")
async def get_risk_metrics():
    """Get current risk metrics"""
    try:
        if not market_maker:
            return {"metrics": {}}

        performance = market_maker.get_performance_metrics()
        portfolio = market_maker.get_portfolio_summary()

        return {
            "performance_metrics": performance,
            "portfolio_greeks": portfolio.get("portfolio_greeks", {}),
            "risk_alerts": portfolio.get("risk_alerts", []),
            "var_95": calculate_portfolio_var(),
            "max_drawdown": performance.get("max_drawdown", 0),
            "sharpe_ratio": performance.get("sharpe_ratio", 0)
        }

    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/options-chain/{symbol}")
async def get_options_chain(symbol: str):
    """Get current options chain for symbol"""
    try:
        if not data_manager:
            return generate_mock_options_chain(symbol)

        chain = data_manager.get_latest_chain(symbol)
        if not chain:
            return generate_mock_options_chain(symbol)

        # Convert to API format
        result = {
            "symbol": symbol,
            "underlying_price": chain.underlying_quote.mid_price if chain.underlying_quote else 400.0,
            "timestamp": chain.timestamp.isoformat(),
            "options": []
        }

        for expiry in chain.expiries[:3]:  # Limit to first 3 expiries
            strikes = chain.get_strikes_for_expiry(expiry)
            for strike in strikes[:10]:  # Limit strikes
                call = chain.get_option(expiry, strike, 'C')
                put = chain.get_option(expiry, strike, 'P')

                option_data = {
                    "strike": strike,
                    "expiry": expiry.isoformat(),
                    "call": {
                        "bid": call.bid if call else 0,
                        "ask": call.ask if call else 0,
                        "last": call.last if call else 0,
                        "volume": call.volume if call else 0,
                        "implied_vol": call.implied_vol if call else 0,
                        "delta": call.delta if call else 0,
                        "gamma": call.gamma if call else 0,
                        "theta": call.theta if call else 0,
                        "vega": call.vega if call else 0
                    },
                    "put": {
                        "bid": put.bid if put else 0,
                        "ask": put.ask if put else 0,
                        "last": put.last if put else 0,
                        "volume": put.volume if put else 0,
                        "implied_vol": put.implied_vol if put else 0,
                        "delta": put.delta if put else 0,
                        "gamma": put.gamma if put else 0,
                        "theta": put.theta if put else 0,
                        "vega": put.vega if put else 0
                    }
                }
                result["options"].append(option_data)

        return result

    except Exception as e:
        logger.error(f"Error getting options chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trade")
async def execute_trade(symbol: str, quantity: int, price: float):
    """Execute a trade (fill simulation)"""
    try:
        if not market_maker:
            raise HTTPException(status_code=503, detail="Market maker not available")

        # Simplified trade execution
        # In practice, this would integrate with actual order management system

        # Create mock option contract
        option = OptionContract(
            symbol=symbol,
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=400.0,
            expiry=30/365.0,
            underlying="SPY"
        )

        market_data = MarketData(400.0, 0.02, 0.0, 0.25, 30/365.0)

        # Handle the fill
        market_maker.handle_fill(symbol, quantity, price, option, market_data)

        # Broadcast trade update
        await manager.broadcast(json.dumps({
            "type": "trade",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now().isoformat()
        }))

        return {"success": True, "message": f"Trade executed: {quantity} {symbol} @ {price}"}

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "subscribe":
                symbol = message.get("symbol", "SPY")
                logger.info(f"Client subscribed to {symbol}")

            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Background tasks

async def handle_options_chain_update(chain):
    """Handle options chain updates from data provider"""
    try:
        # Broadcast update to connected clients
        message = {
            "type": "options_chain_update",
            "symbol": chain.underlying,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "underlying_price": chain.underlying_quote.mid_price if chain.underlying_quote else 400.0,
                "option_count": len(chain.calls) + len(chain.puts)
            }
        }

        await manager.broadcast(json.dumps(message))

    except Exception as e:
        logger.error(f"Error handling chain update: {e}")

def generate_mock_options_chain(symbol: str):
    """Generate mock options chain data"""
    spot_price = 400.0
    strikes = [spot_price + i * 5 for i in range(-10, 11)]  # 21 strikes
    expiries = [7, 14, 30, 60, 90]  # Days to expiry

    options = []
    for days in expiries:
        for strike in strikes:
            time_to_expiry = days / 365.0

            # Simple Black-Scholes for mock data
            call_price = max(spot_price - strike, 0) * 0.6 + 2  # Intrinsic + time value
            put_price = max(strike - spot_price, 0) * 0.6 + 2

            options.append({
                "strike": strike,
                "expiry": (datetime.now() + timedelta(days=days)).isoformat(),
                "call": {
                    "bid": call_price * 0.98,
                    "ask": call_price * 1.02,
                    "last": call_price,
                    "volume": np.random.randint(10, 1000),
                    "implied_vol": 0.25 + np.random.normal(0, 0.02),
                    "delta": 0.5 if strike == spot_price else (0.8 if strike < spot_price else 0.2),
                    "gamma": 0.05,
                    "theta": -0.02,
                    "vega": 0.1
                },
                "put": {
                    "bid": put_price * 0.98,
                    "ask": put_price * 1.02,
                    "last": put_price,
                    "volume": np.random.randint(10, 1000),
                    "implied_vol": 0.25 + np.random.normal(0, 0.02),
                    "delta": -0.5 if strike == spot_price else (-0.8 if strike > spot_price else -0.2),
                    "gamma": 0.05,
                    "theta": -0.02,
                    "vega": 0.1
                }
            })

    return {
        "symbol": symbol,
        "underlying_price": spot_price,
        "timestamp": datetime.now().isoformat(),
        "options": options
    }

def calculate_portfolio_var(confidence_level: float = 0.95) -> float:
    """Calculate portfolio Value at Risk"""
    # Simplified VaR calculation
    # In practice, would use historical simulation or Monte Carlo
    if not market_maker:
        return 0.0

    portfolio_summary = market_maker.get_portfolio_summary()
    total_value = portfolio_summary.get("total_pnl", 0.0)

    # Simple VaR estimation (1% of portfolio value)
    var_95 = abs(total_value) * 0.01
    return var_95

# Background task to send periodic updates
async def send_periodic_updates():
    """Send periodic updates to connected clients"""
    while True:
        try:
            if manager.active_connections and market_maker:
                # Get current portfolio state
                portfolio = market_maker.get_portfolio_summary()
                performance = market_maker.get_performance_metrics()

                message = {
                    "type": "portfolio_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "total_pnl": portfolio.get("total_pnl", 0.0),
                        "portfolio_greeks": portfolio.get("portfolio_greeks", {}),
                        "active_quotes": portfolio.get("active_quotes", 0),
                        "performance": performance
                    }
                }

                await manager.broadcast(json.dumps(message))

            await asyncio.sleep(5)  # Update every 5 seconds

        except Exception as e:
            logger.error(f"Error in periodic updates: {e}")
            await asyncio.sleep(5)

# Start periodic updates task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(send_periodic_updates())

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=True
    )