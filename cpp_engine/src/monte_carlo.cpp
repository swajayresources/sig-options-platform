#include "monte_carlo.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace OptionsEngine {

MonteCarloEngine::MonteCarloEngine(size_t num_simulations, unsigned int seed)
    : num_simulations_(num_simulations), time_steps_(252), rng_(seed) {}

std::vector<double> MonteCarloEngine::generate_price_path(
    const MarketData& market_data,
    int steps) {

    return simulate_gbm_path(
        market_data.spot_price,
        market_data.risk_free_rate,
        market_data.dividend_yield,
        market_data.volatility,
        market_data.time_to_expiry,
        steps
    );
}

std::vector<std::vector<double>> MonteCarloEngine::generate_multiple_paths(
    const MarketData& market_data,
    size_t num_paths,
    int steps) {

    std::vector<std::vector<double>> paths(num_paths);
    for (size_t i = 0; i < num_paths; ++i) {
        paths[i] = generate_price_path(market_data, steps);
    }
    return paths;
}

std::vector<double> MonteCarloEngine::simulate_gbm_path(
    double S0, double r, double q, double vol, double T, int steps) {

    std::vector<double> path(steps + 1);
    path[0] = S0;

    double dt = T / steps;
    double drift = (r - q - 0.5 * vol * vol) * dt;
    double vol_sqrt_dt = vol * std::sqrt(dt);

    for (int i = 1; i <= steps; ++i) {
        double z = rng_.normal();
        path[i] = path[i - 1] * std::exp(drift + vol_sqrt_dt * z);
    }

    return path;
}

std::vector<double> MonteCarloEngine::simulate_jump_diffusion_path(
    double S0, double r, double q, double vol, double T, int steps,
    double jump_intensity, double jump_mean, double jump_std) {

    std::vector<double> path(steps + 1);
    path[0] = S0;

    double dt = T / steps;
    double drift = (r - q - 0.5 * vol * vol - jump_intensity * (std::exp(jump_mean + 0.5 * jump_std * jump_std) - 1.0)) * dt;
    double vol_sqrt_dt = vol * std::sqrt(dt);

    for (int i = 1; i <= steps; ++i) {
        double z = rng_.normal();
        double jump_component = 0.0;

        // Poisson process for jumps
        double poisson_prob = jump_intensity * dt;
        if (rng_.uniform() < poisson_prob) {
            double jump_size = jump_mean + jump_std * rng_.normal();
            jump_component = jump_size;
        }

        path[i] = path[i - 1] * std::exp(drift + vol_sqrt_dt * z + jump_component);
    }

    return path;
}

double MonteCarloEngine::european_payoff(const OptionContract& option, double final_price) {
    if (option.type == OptionType::CALL) {
        return std::max(final_price - option.strike, 0.0);
    } else {
        return std::max(option.strike - final_price, 0.0);
    }
}

double MonteCarloEngine::asian_payoff(
    const OptionContract& option,
    const std::vector<double>& price_path,
    const std::vector<double>& averaging_times) {

    // Calculate average price over specified times
    double sum = 0.0;
    int count = 0;

    double T = option.expiry;
    double dt = T / (price_path.size() - 1);

    for (double avg_time : averaging_times) {
        int index = static_cast<int>(avg_time / dt);
        if (index < static_cast<int>(price_path.size())) {
            sum += price_path[index];
            count++;
        }
    }

    if (count == 0) return 0.0;

    double average_price = sum / count;

    if (option.type == OptionType::CALL) {
        return std::max(average_price - option.strike, 0.0);
    } else {
        return std::max(option.strike - average_price, 0.0);
    }
}

double MonteCarloEngine::barrier_payoff(
    const OptionContract& option,
    const std::vector<double>& price_path,
    double barrier_level,
    bool is_knock_out,
    bool is_up_and_out) {

    bool barrier_hit = false;

    // Check if barrier was hit during the path
    for (double price : price_path) {
        if (is_up_and_out && price >= barrier_level) {
            barrier_hit = true;
            break;
        } else if (!is_up_and_out && price <= barrier_level) {
            barrier_hit = true;
            break;
        }
    }

    double final_price = price_path.back();
    double vanilla_payoff = european_payoff(option, final_price);

    if (is_knock_out) {
        return barrier_hit ? 0.0 : vanilla_payoff;
    } else {
        return barrier_hit ? vanilla_payoff : 0.0;
    }
}

double MonteCarloEngine::lookback_payoff(
    const OptionContract& option,
    const std::vector<double>& price_path) {

    double max_price = *std::max_element(price_path.begin(), price_path.end());
    double min_price = *std::min_element(price_path.begin(), price_path.end());

    if (option.type == OptionType::CALL) {
        // Lookback call pays max(S_max - K, 0)
        return std::max(max_price - option.strike, 0.0);
    } else {
        // Lookback put pays max(K - S_min, 0)
        return std::max(option.strike - min_price, 0.0);
    }
}

PricingResult MonteCarloEngine::price_european_option(
    const OptionContract& option,
    const MarketData& market_data) {

    PricingResult result;

    try {
        if (market_data.time_to_expiry <= 0.0) {
            result.price = european_payoff(option, market_data.spot_price);
            result.success = true;
            return result;
        }

        std::vector<double> payoffs(num_simulations_);

        for (size_t i = 0; i < num_simulations_; ++i) {
            std::vector<double> path = generate_price_path(market_data, 1);
            payoffs[i] = european_payoff(option, path.back());
        }

        // Calculate mean and discount to present value
        double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / num_simulations_;
        result.price = mean_payoff * std::exp(-market_data.risk_free_rate * market_data.time_to_expiry);

        // Calculate standard error
        double std_error = calculate_standard_error(payoffs, mean_payoff);

        result.greeks = calculate_greeks_finite_difference(option, market_data);
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

PricingResult MonteCarloEngine::price_asian_option(
    const OptionContract& option,
    const MarketData& market_data,
    const std::vector<double>& averaging_times) {

    PricingResult result;

    try {
        std::vector<double> payoffs(num_simulations_);

        for (size_t i = 0; i < num_simulations_; ++i) {
            std::vector<double> path = generate_price_path(market_data, time_steps_);
            payoffs[i] = asian_payoff(option, path, averaging_times);
        }

        double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / num_simulations_;
        result.price = mean_payoff * std::exp(-market_data.risk_free_rate * market_data.time_to_expiry);

        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

PricingResult MonteCarloEngine::price_barrier_option(
    const OptionContract& option,
    const MarketData& market_data,
    double barrier_level,
    bool is_knock_out,
    bool is_up_and_out) {

    PricingResult result;

    try {
        std::vector<double> payoffs(num_simulations_);

        for (size_t i = 0; i < num_simulations_; ++i) {
            std::vector<double> path = generate_price_path(market_data, time_steps_);
            payoffs[i] = barrier_payoff(option, path, barrier_level, is_knock_out, is_up_and_out);
        }

        double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / num_simulations_;
        result.price = mean_payoff * std::exp(-market_data.risk_free_rate * market_data.time_to_expiry);

        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

PricingResult MonteCarloEngine::price_lookback_option(
    const OptionContract& option,
    const MarketData& market_data,
    int time_steps) {

    PricingResult result;

    try {
        std::vector<double> payoffs(num_simulations_);

        for (size_t i = 0; i < num_simulations_; ++i) {
            std::vector<double> path = generate_price_path(market_data, time_steps);
            payoffs[i] = lookback_payoff(option, path);
        }

        double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / num_simulations_;
        result.price = mean_payoff * std::exp(-market_data.risk_free_rate * market_data.time_to_expiry);

        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

PricingResult MonteCarloEngine::price_custom_payoff(
    const OptionContract& option,
    const MarketData& market_data,
    std::function<double(const std::vector<double>&)> payoff_function,
    int time_steps) {

    PricingResult result;

    try {
        std::vector<double> payoffs(num_simulations_);

        for (size_t i = 0; i < num_simulations_; ++i) {
            std::vector<double> path = generate_price_path(market_data, time_steps);
            payoffs[i] = payoff_function(path);
        }

        double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / num_simulations_;
        result.price = mean_payoff * std::exp(-market_data.risk_free_rate * market_data.time_to_expiry);

        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

PricingResult MonteCarloEngine::price_with_antithetic_variates(
    const OptionContract& option,
    const MarketData& market_data) {

    PricingResult result;

    try {
        if (num_simulations_ % 2 != 0) {
            // Ensure even number of simulations for antithetic pairs
            const_cast<MonteCarloEngine*>(this)->num_simulations_++;
        }

        std::vector<double> payoffs(num_simulations_);

        for (size_t i = 0; i < num_simulations_; i += 2) {
            // Generate normal random numbers
            std::vector<double> normals = rng_.normal_vector(1);

            // Original path
            double z1 = normals[0];
            double dt = market_data.time_to_expiry;
            double drift = (market_data.risk_free_rate - market_data.dividend_yield -
                           0.5 * market_data.volatility * market_data.volatility) * dt;
            double vol_sqrt_dt = market_data.volatility * std::sqrt(dt);

            double S1 = market_data.spot_price * std::exp(drift + vol_sqrt_dt * z1);
            payoffs[i] = european_payoff(option, S1);

            // Antithetic path
            double z2 = -z1;
            double S2 = market_data.spot_price * std::exp(drift + vol_sqrt_dt * z2);
            payoffs[i + 1] = european_payoff(option, S2);
        }

        double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / num_simulations_;
        result.price = mean_payoff * std::exp(-market_data.risk_free_rate * market_data.time_to_expiry);

        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

PricingResult MonteCarloEngine::price_with_control_variates(
    const OptionContract& option,
    const MarketData& market_data,
    const OptionContract& control_option) {

    PricingResult result;

    try {
        std::vector<double> payoffs(num_simulations_);
        std::vector<double> control_payoffs(num_simulations_);

        for (size_t i = 0; i < num_simulations_; ++i) {
            std::vector<double> path = generate_price_path(market_data, 1);
            double final_price = path.back();

            payoffs[i] = european_payoff(option, final_price);
            control_payoffs[i] = european_payoff(control_option, final_price);
        }

        // Calculate control variate coefficient
        double cov = 0.0, var_control = 0.0;
        double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / num_simulations_;
        double mean_control = std::accumulate(control_payoffs.begin(), control_payoffs.end(), 0.0) / num_simulations_;

        for (size_t i = 0; i < num_simulations_; ++i) {
            cov += (payoffs[i] - mean_payoff) * (control_payoffs[i] - mean_control);
            var_control += (control_payoffs[i] - mean_control) * (control_payoffs[i] - mean_control);
        }

        double beta = (var_control > 0) ? cov / var_control : 0.0;

        // Get theoretical price of control option (assume Black-Scholes)
        // This would require implementing Black-Scholes pricing here or passing it in
        double control_theoretical = 0.0;  // Placeholder

        // Apply control variate adjustment
        double adjusted_mean = mean_payoff - beta * (mean_control - control_theoretical);
        result.price = adjusted_mean * std::exp(-market_data.risk_free_rate * market_data.time_to_expiry);

        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

Greeks MonteCarloEngine::calculate_greeks_finite_difference(
    const OptionContract& option,
    const MarketData& market_data) {

    Greeks greeks;

    // Delta calculation
    double spot_bump = market_data.spot_price * 0.01;
    MarketData market_up = market_data;
    market_up.spot_price += spot_bump;
    MarketData market_down = market_data;
    market_down.spot_price -= spot_bump;

    PricingResult result_up = price_european_option(option, market_up);
    PricingResult result_down = price_european_option(option, market_down);

    if (result_up.success && result_down.success) {
        greeks.delta = (result_up.price - result_down.price) / (2.0 * spot_bump);
    }

    // Gamma calculation
    PricingResult result_center = price_european_option(option, market_data);
    if (result_center.success && result_up.success && result_down.success) {
        greeks.gamma = (result_up.price - 2.0 * result_center.price + result_down.price) / (spot_bump * spot_bump);
    }

    // Theta calculation
    double time_bump = 1.0 / 365.0;
    if (market_data.time_to_expiry > time_bump) {
        MarketData market_time = market_data;
        market_time.time_to_expiry -= time_bump;
        PricingResult result_time = price_european_option(option, market_time);

        if (result_center.success && result_time.success) {
            greeks.theta = (result_time.price - result_center.price) / time_bump;
        }
    }

    // Vega calculation
    double vol_bump = 0.01;
    MarketData market_vol = market_data;
    market_vol.volatility += vol_bump;
    PricingResult result_vol = price_european_option(option, market_vol);

    if (result_center.success && result_vol.success) {
        greeks.vega = (result_vol.price - result_center.price) / vol_bump;
    }

    // Rho calculation
    double rate_bump = 0.01;
    MarketData market_rate = market_data;
    market_rate.risk_free_rate += rate_bump;
    PricingResult result_rate = price_european_option(option, market_rate);

    if (result_center.success && result_rate.success) {
        greeks.rho = (result_rate.price - result_center.price) / rate_bump;
    }

    return greeks;
}

Greeks MonteCarloEngine::calculate_greeks(
    const OptionContract& option,
    const MarketData& market_data) {

    return calculate_greeks_finite_difference(option, market_data);
}

double MonteCarloEngine::calculate_standard_error(
    const std::vector<double>& payoffs,
    double mean_payoff) {

    double variance = 0.0;
    for (double payoff : payoffs) {
        variance += (payoff - mean_payoff) * (payoff - mean_payoff);
    }
    variance /= (payoffs.size() - 1);

    return std::sqrt(variance / payoffs.size());
}

// Longstaff-Schwartz Engine Implementation
LongstaffSchwartzEngine::LongstaffSchwartzEngine(
    size_t num_simulations,
    int regression_basis_functions)
    : num_simulations_(num_simulations),
      basis_functions_(regression_basis_functions),
      rng_() {}

std::vector<double> LongstaffSchwartzEngine::basis_function_values(
    double stock_price, int num_functions) {

    std::vector<double> basis(num_functions);

    // Polynomial basis functions: 1, S, S^2, ...
    double power = 1.0;
    for (int i = 0; i < num_functions; ++i) {
        basis[i] = power;
        power *= stock_price;
    }

    return basis;
}

PricingResult LongstaffSchwartzEngine::price_american_option(
    const OptionContract& option,
    const MarketData& market_data,
    int time_steps) {

    PricingResult result;

    try {
        // Generate all price paths
        std::vector<std::vector<double>> paths(num_simulations_);
        for (size_t i = 0; i < num_simulations_; ++i) {
            double dt = market_data.time_to_expiry / time_steps;
            double drift = (market_data.risk_free_rate - market_data.dividend_yield -
                           0.5 * market_data.volatility * market_data.volatility) * dt;
            double vol_sqrt_dt = market_data.volatility * std::sqrt(dt);

            paths[i].resize(time_steps + 1);
            paths[i][0] = market_data.spot_price;

            for (int t = 1; t <= time_steps; ++t) {
                double z = rng_.normal();
                paths[i][t] = paths[i][t - 1] * std::exp(drift + vol_sqrt_dt * z);
            }
        }

        // Initialize option values at expiry
        std::vector<double> option_values(num_simulations_);
        for (size_t i = 0; i < num_simulations_; ++i) {
            double final_price = paths[i][time_steps];
            if (option.type == OptionType::CALL) {
                option_values[i] = std::max(final_price - option.strike, 0.0);
            } else {
                option_values[i] = std::max(option.strike - final_price, 0.0);
            }
        }

        double dt = market_data.time_to_expiry / time_steps;
        double discount_factor = std::exp(-market_data.risk_free_rate * dt);

        // Work backwards through time
        for (int t = time_steps - 1; t >= 1; --t) {
            std::vector<std::vector<double>> X;
            std::vector<double> y;

            // Find in-the-money paths
            for (size_t i = 0; i < num_simulations_; ++i) {
                double stock_price = paths[i][t];
                double intrinsic = 0.0;

                if (option.type == OptionType::CALL) {
                    intrinsic = std::max(stock_price - option.strike, 0.0);
                } else {
                    intrinsic = std::max(option.strike - stock_price, 0.0);
                }

                if (intrinsic > 0.0) {
                    X.push_back(basis_function_values(stock_price, basis_functions_));
                    y.push_back(option_values[i] * discount_factor);
                }
            }

            // Perform regression if we have enough data points
            std::vector<double> coefficients;
            if (X.size() >= static_cast<size_t>(basis_functions_)) {
                coefficients = least_squares_regression(X, y);
            }

            // Update option values
            for (size_t i = 0; i < num_simulations_; ++i) {
                double stock_price = paths[i][t];
                double intrinsic = 0.0;

                if (option.type == OptionType::CALL) {
                    intrinsic = std::max(stock_price - option.strike, 0.0);
                } else {
                    intrinsic = std::max(option.strike - stock_price, 0.0);
                }

                if (intrinsic > 0.0 && !coefficients.empty()) {
                    std::vector<double> basis = basis_function_values(stock_price, basis_functions_);
                    double continuation_value = 0.0;

                    for (int j = 0; j < basis_functions_ && j < static_cast<int>(coefficients.size()); ++j) {
                        continuation_value += coefficients[j] * basis[j];
                    }

                    if (intrinsic > continuation_value) {
                        option_values[i] = intrinsic;
                    } else {
                        option_values[i] *= discount_factor;
                    }
                } else {
                    option_values[i] *= discount_factor;
                }
            }
        }

        // Calculate final option price
        double sum = std::accumulate(option_values.begin(), option_values.end(), 0.0);
        result.price = (sum / num_simulations_) * discount_factor;
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

std::vector<double> LongstaffSchwartzEngine::least_squares_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y) {

    size_t n = X.size();
    size_t p = X[0].size();

    // Normal equations: (X'X)β = X'y
    std::vector<std::vector<double>> XtX(p, std::vector<double>(p, 0.0));
    std::vector<double> Xty(p, 0.0);

    // Calculate X'X and X'y
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            Xty[j] += X[i][j] * y[i];
            for (size_t k = 0; k < p; ++k) {
                XtX[j][k] += X[i][j] * X[i][k];
            }
        }
    }

    // Solve using Gaussian elimination (simplified)
    std::vector<double> coefficients(p, 0.0);

    // This is a simplified implementation - in practice, use a robust linear algebra library
    for (size_t i = 0; i < p; ++i) {
        if (std::abs(XtX[i][i]) > 1e-10) {
            coefficients[i] = Xty[i] / XtX[i][i];
        }
    }

    return coefficients;
}

}  // namespace OptionsEngine