#pragma once

#include "option_types.h"
#include "math_utils.h"

namespace OptionsEngine {

class BlackScholesModel {
public:
    static PricingResult price_option(
        const OptionContract& option,
        const MarketData& market_data
    );

    static Greeks calculate_greeks(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_implied_volatility(
        const OptionContract& option,
        const MarketData& market_data,
        double market_price,
        double tolerance = 1e-6
    );

    static double calculate_delta(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_gamma(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_theta(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_vega(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_rho(
        const OptionContract& option,
        const MarketData& market_data
    );

    // Second-order Greeks
    static double calculate_vomma(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_vanna(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_charm(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_color(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_speed(
        const OptionContract& option,
        const MarketData& market_data
    );

    static double calculate_ultima(
        const OptionContract& option,
        const MarketData& market_data
    );

private:
    static double d1(const OptionContract& option, const MarketData& market_data);
    static double d2(const OptionContract& option, const MarketData& market_data);

    static double call_price(const OptionContract& option, const MarketData& market_data);
    static double put_price(const OptionContract& option, const MarketData& market_data);

    // Helper functions for numerical derivatives
    static double numerical_delta(
        const OptionContract& option,
        const MarketData& market_data,
        double bump = 0.01
    );

    static double numerical_gamma(
        const OptionContract& option,
        const MarketData& market_data,
        double bump = 0.01
    );

    static double numerical_theta(
        const OptionContract& option,
        const MarketData& market_data,
        double bump = 1.0/365.0  // 1 day
    );

    static double numerical_vega(
        const OptionContract& option,
        const MarketData& market_data,
        double bump = 0.01
    );

    static double numerical_rho(
        const OptionContract& option,
        const MarketData& market_data,
        double bump = 0.01
    );
};

class BlackScholesMerton : public BlackScholesModel {
public:
    // Extended Black-Scholes-Merton model with dividends
    static PricingResult price_option_with_dividends(
        const OptionContract& option,
        const MarketData& market_data,
        const std::vector<double>& dividend_times,
        const std::vector<double>& dividend_amounts
    );

    // Analytical Greeks for dividend-paying stocks
    static Greeks calculate_greeks_with_dividends(
        const OptionContract& option,
        const MarketData& market_data,
        const std::vector<double>& dividend_times,
        const std::vector<double>& dividend_amounts
    );
};

}  // namespace OptionsEngine