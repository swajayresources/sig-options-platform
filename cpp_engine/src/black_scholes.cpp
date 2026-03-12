#include "black_scholes.h"
#include <cmath>
#include <algorithm>
#include <functional>

namespace OptionsEngine {

// Helper functions
double BlackScholesModel::d1(const OptionContract& option, const MarketData& market_data) {
    double S = market_data.spot_price;
    double K = option.strike;
    double r = market_data.risk_free_rate;
    double q = market_data.dividend_yield;
    double vol = market_data.volatility;
    double T = market_data.time_to_expiry;

    if (T <= 0.0 || vol <= 0.0) {
        return 0.0;
    }

    return (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
}

double BlackScholesModel::d2(const OptionContract& option, const MarketData& market_data) {
    double vol = market_data.volatility;
    double T = market_data.time_to_expiry;

    if (T <= 0.0) {
        return d1(option, market_data);
    }

    return d1(option, market_data) - vol * std::sqrt(T);
}

double BlackScholesModel::call_price(const OptionContract& option, const MarketData& market_data) {
    double S = market_data.spot_price;
    double K = option.strike;
    double r = market_data.risk_free_rate;
    double q = market_data.dividend_yield;
    double T = market_data.time_to_expiry;

    if (T <= 0.0) {
        return std::max(S - K, 0.0);
    }

    double d1_val = d1(option, market_data);
    double d2_val = d2(option, market_data);

    double call_value = S * std::exp(-q * T) * MathUtils::NormalDistribution::cdf(d1_val) -
                       K * std::exp(-r * T) * MathUtils::NormalDistribution::cdf(d2_val);

    return std::max(call_value, 0.0);
}

double BlackScholesModel::put_price(const OptionContract& option, const MarketData& market_data) {
    double S = market_data.spot_price;
    double K = option.strike;
    double r = market_data.risk_free_rate;
    double q = market_data.dividend_yield;
    double T = market_data.time_to_expiry;

    if (T <= 0.0) {
        return std::max(K - S, 0.0);
    }

    double d1_val = d1(option, market_data);
    double d2_val = d2(option, market_data);

    double put_value = K * std::exp(-r * T) * MathUtils::NormalDistribution::cdf(-d2_val) -
                      S * std::exp(-q * T) * MathUtils::NormalDistribution::cdf(-d1_val);

    return std::max(put_value, 0.0);
}

// Main pricing function
PricingResult BlackScholesModel::price_option(
    const OptionContract& option,
    const MarketData& market_data) {

    PricingResult result;

    try {
        if (market_data.time_to_expiry <= 0.0) {
            // Option has expired
            double intrinsic = 0.0;
            if (option.type == OptionType::CALL) {
                intrinsic = std::max(market_data.spot_price - option.strike, 0.0);
            } else {
                intrinsic = std::max(option.strike - market_data.spot_price, 0.0);
            }
            result.price = intrinsic;
            result.success = true;
            return result;
        }

        if (market_data.volatility <= 0.0) {
            result.success = false;
            result.error_message = "Volatility must be positive";
            return result;
        }

        if (option.type == OptionType::CALL) {
            result.price = call_price(option, market_data);
        } else {
            result.price = put_price(option, market_data);
        }

        result.greeks = calculate_greeks(option, market_data);
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

// Greeks calculations
double BlackScholesModel::calculate_delta(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        if (option.type == OptionType::CALL) {
            return (market_data.spot_price > option.strike) ? 1.0 : 0.0;
        } else {
            return (market_data.spot_price < option.strike) ? -1.0 : 0.0;
        }
    }

    double d1_val = d1(option, market_data);
    double q = market_data.dividend_yield;
    double T = market_data.time_to_expiry;

    if (option.type == OptionType::CALL) {
        return std::exp(-q * T) * MathUtils::NormalDistribution::cdf(d1_val);
    } else {
        return -std::exp(-q * T) * MathUtils::NormalDistribution::cdf(-d1_val);
    }
}

double BlackScholesModel::calculate_gamma(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double S = market_data.spot_price;
    double q = market_data.dividend_yield;
    double vol = market_data.volatility;
    double T = market_data.time_to_expiry;

    double d1_val = d1(option, market_data);

    return std::exp(-q * T) * MathUtils::NormalDistribution::pdf(d1_val) /
           (S * vol * std::sqrt(T));
}

double BlackScholesModel::calculate_theta(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double S = market_data.spot_price;
    double K = option.strike;
    double r = market_data.risk_free_rate;
    double q = market_data.dividend_yield;
    double vol = market_data.volatility;
    double T = market_data.time_to_expiry;

    double d1_val = d1(option, market_data);
    double d2_val = d2(option, market_data);

    double term1 = -S * std::exp(-q * T) * MathUtils::NormalDistribution::pdf(d1_val) * vol /
                   (2.0 * std::sqrt(T));

    if (option.type == OptionType::CALL) {
        double term2 = -r * K * std::exp(-r * T) * MathUtils::NormalDistribution::cdf(d2_val);
        double term3 = q * S * std::exp(-q * T) * MathUtils::NormalDistribution::cdf(d1_val);
        return term1 + term2 + term3;
    } else {
        double term2 = r * K * std::exp(-r * T) * MathUtils::NormalDistribution::cdf(-d2_val);
        double term3 = -q * S * std::exp(-q * T) * MathUtils::NormalDistribution::cdf(-d1_val);
        return term1 + term2 + term3;
    }
}

double BlackScholesModel::calculate_vega(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double S = market_data.spot_price;
    double q = market_data.dividend_yield;
    double T = market_data.time_to_expiry;

    double d1_val = d1(option, market_data);

    return S * std::exp(-q * T) * MathUtils::NormalDistribution::pdf(d1_val) * std::sqrt(T);
}

double BlackScholesModel::calculate_rho(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double K = option.strike;
    double r = market_data.risk_free_rate;
    double T = market_data.time_to_expiry;

    double d2_val = d2(option, market_data);

    if (option.type == OptionType::CALL) {
        return K * T * std::exp(-r * T) * MathUtils::NormalDistribution::cdf(d2_val);
    } else {
        return -K * T * std::exp(-r * T) * MathUtils::NormalDistribution::cdf(-d2_val);
    }
}

// Second-order Greeks
double BlackScholesModel::calculate_vomma(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double S = market_data.spot_price;
    double q = market_data.dividend_yield;
    double vol = market_data.volatility;
    double T = market_data.time_to_expiry;

    double d1_val = d1(option, market_data);
    double d2_val = d2(option, market_data);

    double vega = calculate_vega(option, market_data);

    return vega * d1_val * d2_val / vol;
}

double BlackScholesModel::calculate_vanna(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double q = market_data.dividend_yield;
    double vol = market_data.volatility;

    double d1_val = d1(option, market_data);
    double d2_val = d2(option, market_data);

    double vega = calculate_vega(option, market_data);

    return -std::exp(-q * market_data.time_to_expiry) * d2_val / vol *
           MathUtils::NormalDistribution::pdf(d1_val) * std::sqrt(market_data.time_to_expiry);
}

double BlackScholesModel::calculate_charm(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double S = market_data.spot_price;
    double r = market_data.risk_free_rate;
    double q = market_data.dividend_yield;
    double vol = market_data.volatility;
    double T = market_data.time_to_expiry;

    double d1_val = d1(option, market_data);
    double d2_val = d2(option, market_data);

    double term1 = q * std::exp(-q * T) * MathUtils::NormalDistribution::cdf(d1_val);
    double term2 = std::exp(-q * T) * MathUtils::NormalDistribution::pdf(d1_val) *
                   (2.0 * (r - q) * T - d2_val * vol * std::sqrt(T)) / (2.0 * T * vol * std::sqrt(T));

    if (option.type == OptionType::CALL) {
        return -term1 - term2;
    } else {
        return term1 - term2;
    }
}

double BlackScholesModel::calculate_color(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double S = market_data.spot_price;
    double r = market_data.risk_free_rate;
    double q = market_data.dividend_yield;
    double vol = market_data.volatility;
    double T = market_data.time_to_expiry;

    double d1_val = d1(option, market_data);
    double d2_val = d2(option, market_data);

    double gamma = calculate_gamma(option, market_data);

    double term1 = -gamma / S;
    double term2 = 2.0 * q * T + 1.0;
    double term3 = (2.0 * (r - q) * T - d2_val * vol * std::sqrt(T)) / (vol * std::sqrt(T));

    return gamma * (term1 * (term2 + term3));
}

double BlackScholesModel::calculate_speed(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double S = market_data.spot_price;
    double vol = market_data.volatility;
    double T = market_data.time_to_expiry;

    double d1_val = d1(option, market_data);
    double gamma = calculate_gamma(option, market_data);

    return -gamma / S * (d1_val / (vol * std::sqrt(T)) + 1.0);
}

double BlackScholesModel::calculate_ultima(
    const OptionContract& option,
    const MarketData& market_data) {

    if (market_data.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double vol = market_data.volatility;

    double d1_val = d1(option, market_data);
    double d2_val = d2(option, market_data);
    double vomma = calculate_vomma(option, market_data);

    return -vomma / vol * (d1_val * d2_val * (1.0 - d1_val * d2_val) + d1_val * d1_val + d2_val * d2_val);
}

Greeks BlackScholesModel::calculate_greeks(
    const OptionContract& option,
    const MarketData& market_data) {

    Greeks greeks;

    greeks.delta = calculate_delta(option, market_data);
    greeks.gamma = calculate_gamma(option, market_data);
    greeks.theta = calculate_theta(option, market_data);
    greeks.vega = calculate_vega(option, market_data);
    greeks.rho = calculate_rho(option, market_data);

    // Second-order Greeks
    greeks.vomma = calculate_vomma(option, market_data);
    greeks.vanna = calculate_vanna(option, market_data);
    greeks.charm = calculate_charm(option, market_data);
    greeks.color = calculate_color(option, market_data);
    greeks.speed = calculate_speed(option, market_data);
    greeks.ultima = calculate_ultima(option, market_data);

    return greeks;
}

double BlackScholesModel::calculate_implied_volatility(
    const OptionContract& option,
    const MarketData& market_data,
    double market_price,
    double tolerance) {

    if (market_price <= 0.0) {
        return 0.0;
    }

    // Use Brent's method to find implied volatility
    auto pricing_function = [&](double vol) -> double {
        MarketData temp_data = market_data;
        temp_data.volatility = vol;
        PricingResult result = price_option(option, temp_data);
        return result.price - market_price;
    };

    try {
        return MathUtils::NumericalMethods::brent_method(
            pricing_function, 0.001, 5.0, tolerance, 100
        );
    } catch (const std::exception&) {
        // If Brent's method fails, try Newton-Raphson with vega
        double vol_guess = 0.2;  // 20% initial guess

        for (int i = 0; i < 50; ++i) {
            MarketData temp_data = market_data;
            temp_data.volatility = vol_guess;

            PricingResult result = price_option(option, temp_data);
            double price_diff = result.price - market_price;

            if (std::abs(price_diff) < tolerance) {
                return vol_guess;
            }

            double vega = calculate_vega(option, temp_data);
            if (std::abs(vega) < MathUtils::EPSILON) {
                break;
            }

            vol_guess = vol_guess - price_diff / vega;

            if (vol_guess <= 0.0) {
                vol_guess = 0.001;
            }
        }
    }

    return 0.0;  // Failed to find implied volatility
}

}  // namespace OptionsEngine