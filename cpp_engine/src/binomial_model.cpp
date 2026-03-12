#include "binomial_model.h"
#include "math_utils.h"
#include <algorithm>
#include <cmath>

namespace OptionsEngine {

BinomialModel::BinomialModel(int steps) : steps_(steps) {}

BinomialModel::BinomialParameters BinomialModel::calculate_parameters(
    const MarketData& market_data) const {

    BinomialParameters params;

    double dt = market_data.time_to_expiry / steps_;
    double r = market_data.risk_free_rate;
    double q = market_data.dividend_yield;
    double vol = market_data.volatility;

    // Cox-Ross-Rubinstein parameters
    params.up_factor = std::exp(vol * std::sqrt(dt));
    params.down_factor = 1.0 / params.up_factor;
    params.discount_factor = std::exp(-r * dt);

    // Risk-neutral probability
    double forward_factor = std::exp((r - q) * dt);
    params.risk_neutral_prob = (forward_factor - params.down_factor) /
                              (params.up_factor - params.down_factor);

    return params;
}

std::vector<double> BinomialModel::build_stock_tree(
    double spot_price,
    const BinomialParameters& params) const {

    // Only need final stock prices for European options
    std::vector<double> final_prices(steps_ + 1);

    for (int i = 0; i <= steps_; ++i) {
        int num_ups = i;
        int num_downs = steps_ - i;
        final_prices[i] = spot_price *
                         std::pow(params.up_factor, num_ups) *
                         std::pow(params.down_factor, num_downs);
    }

    return final_prices;
}

std::vector<double> BinomialModel::build_option_tree_european(
    const OptionContract& option,
    const std::vector<double>& stock_prices,
    const BinomialParameters& params) const {

    std::vector<double> option_values(steps_ + 1);

    // Calculate option values at expiry
    for (int i = 0; i <= steps_; ++i) {
        option_values[i] = intrinsic_value(option, stock_prices[i]);
    }

    // Work backwards through the tree
    for (int step = steps_ - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            double continuation_value = params.discount_factor *
                (params.risk_neutral_prob * option_values[i + 1] +
                 (1.0 - params.risk_neutral_prob) * option_values[i]);
            option_values[i] = continuation_value;
        }
    }

    return option_values;
}

std::vector<double> BinomialModel::build_option_tree_american(
    const OptionContract& option,
    const std::vector<double>& stock_prices,
    const BinomialParameters& params) const {

    // Build full stock price tree for American options
    std::vector<std::vector<double>> stock_tree(steps_ + 1);
    for (int step = 0; step <= steps_; ++step) {
        stock_tree[step].resize(step + 1);
    }

    // Initialize stock tree
    stock_tree[0][0] = stock_prices[0];  // Assuming spot price is first element

    for (int step = 1; step <= steps_; ++step) {
        for (int i = 0; i <= step; ++i) {
            if (i == 0) {
                stock_tree[step][i] = stock_tree[step - 1][0] * params.down_factor;
            } else if (i == step) {
                stock_tree[step][i] = stock_tree[step - 1][step - 1] * params.up_factor;
            } else {
                // This node can be reached by up or down move
                stock_tree[step][i] = stock_tree[step - 1][i - 1] * params.up_factor;
            }
        }
    }

    // Build option value tree
    std::vector<std::vector<double>> option_tree(steps_ + 1);
    for (int step = 0; step <= steps_; ++step) {
        option_tree[step].resize(step + 1);
    }

    // Calculate option values at expiry
    for (int i = 0; i <= steps_; ++i) {
        option_tree[steps_][i] = intrinsic_value(option, stock_tree[steps_][i]);
    }

    // Work backwards through the tree with early exercise check
    for (int step = steps_ - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            double continuation_value = params.discount_factor *
                (params.risk_neutral_prob * option_tree[step + 1][i + 1] +
                 (1.0 - params.risk_neutral_prob) * option_tree[step + 1][i]);

            double exercise_value = intrinsic_value(option, stock_tree[step][i]);

            option_tree[step][i] = std::max(continuation_value, exercise_value);
        }
    }

    // Return just the root values for compatibility
    std::vector<double> result(1);
    result[0] = option_tree[0][0];
    return result;
}

double BinomialModel::intrinsic_value(const OptionContract& option, double stock_price) const {
    if (option.type == OptionType::CALL) {
        return std::max(stock_price - option.strike, 0.0);
    } else {
        return std::max(option.strike - stock_price, 0.0);
    }
}

PricingResult BinomialModel::price_option(
    const OptionContract& option,
    const MarketData& market_data) {

    PricingResult result;

    try {
        if (market_data.time_to_expiry <= 0.0) {
            result.price = intrinsic_value(option, market_data.spot_price);
            result.success = true;
            return result;
        }

        BinomialParameters params = calculate_parameters(market_data);
        std::vector<double> stock_prices = build_stock_tree(market_data.spot_price, params);

        std::vector<double> option_values;
        if (option.exercise_type == ExerciseType::EUROPEAN) {
            option_values = build_option_tree_european(option, stock_prices, params);
        } else {
            option_values = build_option_tree_american(option, stock_prices, params);
        }

        result.price = option_values[0];
        result.greeks = calculate_greeks(option, market_data);
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

PricingResult BinomialModel::price_american_option(
    const OptionContract& option,
    const MarketData& market_data) {

    OptionContract american_option = option;
    american_option.exercise_type = ExerciseType::AMERICAN;

    return price_option(american_option, market_data);
}

double BinomialModel::calculate_delta_from_tree(
    const OptionContract& option,
    const MarketData& market_data) {

    // Calculate option values for S+ΔS and S-ΔS
    double bump = market_data.spot_price * 0.01;  // 1% bump

    MarketData market_up = market_data;
    market_up.spot_price += bump;

    MarketData market_down = market_data;
    market_down.spot_price -= bump;

    PricingResult result_up = price_option(option, market_up);
    PricingResult result_down = price_option(option, market_down);

    if (!result_up.success || !result_down.success) {
        return 0.0;
    }

    return (result_up.price - result_down.price) / (2.0 * bump);
}

double BinomialModel::calculate_gamma_from_tree(
    const OptionContract& option,
    const MarketData& market_data) {

    double bump = market_data.spot_price * 0.01;

    MarketData market_up = market_data;
    market_up.spot_price += bump;

    MarketData market_down = market_data;
    market_down.spot_price -= bump;

    PricingResult result_center = price_option(option, market_data);
    PricingResult result_up = price_option(option, market_up);
    PricingResult result_down = price_option(option, market_down);

    if (!result_center.success || !result_up.success || !result_down.success) {
        return 0.0;
    }

    return (result_up.price - 2.0 * result_center.price + result_down.price) / (bump * bump);
}

double BinomialModel::calculate_theta_from_tree(
    const OptionContract& option,
    const MarketData& market_data) {

    double time_bump = 1.0 / 365.0;  // 1 day

    if (market_data.time_to_expiry <= time_bump) {
        return 0.0;
    }

    MarketData market_shorter = market_data;
    market_shorter.time_to_expiry -= time_bump;

    PricingResult result_now = price_option(option, market_data);
    PricingResult result_later = price_option(option, market_shorter);

    if (!result_now.success || !result_later.success) {
        return 0.0;
    }

    return (result_later.price - result_now.price) / time_bump;
}

Greeks BinomialModel::calculate_greeks(
    const OptionContract& option,
    const MarketData& market_data) {

    Greeks greeks;

    greeks.delta = calculate_delta_from_tree(option, market_data);
    greeks.gamma = calculate_gamma_from_tree(option, market_data);
    greeks.theta = calculate_theta_from_tree(option, market_data);

    // For vega and rho, use finite differences
    double vol_bump = 0.01;  // 1% volatility bump
    MarketData market_vol_up = market_data;
    market_vol_up.volatility += vol_bump;

    PricingResult result_base = price_option(option, market_data);
    PricingResult result_vol_up = price_option(option, market_vol_up);

    if (result_base.success && result_vol_up.success) {
        greeks.vega = (result_vol_up.price - result_base.price) / vol_bump;
    }

    double rate_bump = 0.01;  // 1% rate bump
    MarketData market_rate_up = market_data;
    market_rate_up.risk_free_rate += rate_bump;

    PricingResult result_rate_up = price_option(option, market_rate_up);

    if (result_base.success && result_rate_up.success) {
        greeks.rho = (result_rate_up.price - result_base.price) / rate_bump;
    }

    return greeks;
}

// Trinomial Model Implementation
TrinomialModel::TrinomialModel(int steps) : steps_(steps) {}

TrinomialModel::TrinomialParameters TrinomialModel::calculate_parameters(
    const MarketData& market_data) const {

    TrinomialParameters params;

    double dt = market_data.time_to_expiry / steps_;
    double r = market_data.risk_free_rate;
    double q = market_data.dividend_yield;
    double vol = market_data.volatility;

    // Boyle trinomial parameters
    params.up_factor = std::exp(vol * std::sqrt(2.0 * dt));
    params.down_factor = 1.0 / params.up_factor;
    params.discount_factor = std::exp(-r * dt);

    double pu_temp = (std::exp((r - q) * dt / 2.0) - std::exp(-vol * std::sqrt(dt / 2.0))) /
                     (std::exp(vol * std::sqrt(dt / 2.0)) - std::exp(-vol * std::sqrt(dt / 2.0)));

    double pd_temp = (std::exp(vol * std::sqrt(dt / 2.0)) - std::exp((r - q) * dt / 2.0)) /
                     (std::exp(vol * std::sqrt(dt / 2.0)) - std::exp(-vol * std::sqrt(dt / 2.0)));

    params.up_prob = pu_temp * pu_temp;
    params.down_prob = pd_temp * pd_temp;
    params.middle_prob = 1.0 - params.up_prob - params.down_prob;

    return params;
}

// Adaptive Mesh Model Implementation
AdaptiveMeshModel::AdaptiveMeshModel(int initial_steps, double convergence_tolerance)
    : initial_steps_(initial_steps), convergence_tolerance_(convergence_tolerance) {}

bool AdaptiveMeshModel::has_converged(double price1, double price2) const {
    return std::abs(price1 - price2) / std::max(std::abs(price1), 1e-10) < convergence_tolerance_;
}

PricingResult AdaptiveMeshModel::price_option(
    const OptionContract& option,
    const MarketData& market_data) {

    return price_with_adaptive_refinement(option, market_data);
}

PricingResult AdaptiveMeshModel::price_with_adaptive_refinement(
    const OptionContract& option,
    const MarketData& market_data,
    int max_steps) {

    BinomialModel binomial_coarse(initial_steps_);
    BinomialModel binomial_fine(initial_steps_ * 2);

    PricingResult result_coarse = binomial_coarse.price_option(option, market_data);
    PricingResult result_fine = binomial_fine.price_option(option, market_data);

    if (!result_coarse.success || !result_fine.success) {
        return result_coarse.success ? result_coarse : result_fine;
    }

    int current_steps = initial_steps_ * 2;

    while (!has_converged(result_coarse.price, result_fine.price) && current_steps < max_steps) {
        result_coarse = result_fine;

        current_steps *= 2;
        BinomialModel binomial_finer(current_steps);
        result_fine = binomial_finer.price_option(option, market_data);

        if (!result_fine.success) {
            return result_coarse;
        }
    }

    return result_fine;
}

}  // namespace OptionsEngine