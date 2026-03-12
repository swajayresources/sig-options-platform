#pragma once

#include "option_types.h"
#include "math_utils.h"
#include <vector>
#include <functional>

namespace OptionsEngine {

class MonteCarloEngine {
public:
    MonteCarloEngine(
        size_t num_simulations = 100000,
        unsigned int seed = std::random_device{}()
    );

    PricingResult price_european_option(
        const OptionContract& option,
        const MarketData& market_data
    );

    PricingResult price_asian_option(
        const OptionContract& option,
        const MarketData& market_data,
        const std::vector<double>& averaging_times
    );

    PricingResult price_barrier_option(
        const OptionContract& option,
        const MarketData& market_data,
        double barrier_level,
        bool is_knock_out = true,
        bool is_up_and_out = true
    );

    PricingResult price_lookback_option(
        const OptionContract& option,
        const MarketData& market_data,
        int time_steps = 252
    );

    // Path-dependent options with custom payoff functions
    PricingResult price_custom_payoff(
        const OptionContract& option,
        const MarketData& market_data,
        std::function<double(const std::vector<double>&)> payoff_function,
        int time_steps = 252
    );

    Greeks calculate_greeks(
        const OptionContract& option,
        const MarketData& market_data
    );

    // Variance reduction techniques
    PricingResult price_with_antithetic_variates(
        const OptionContract& option,
        const MarketData& market_data
    );

    PricingResult price_with_control_variates(
        const OptionContract& option,
        const MarketData& market_data,
        const OptionContract& control_option
    );

    PricingResult price_with_importance_sampling(
        const OptionContract& option,
        const MarketData& market_data,
        double drift_adjustment = 0.0
    );

    void set_num_simulations(size_t num_sims) { num_simulations_ = num_sims; }
    size_t get_num_simulations() const { return num_simulations_; }

    void set_time_steps(int steps) { time_steps_ = steps; }
    int get_time_steps() const { return time_steps_; }

private:
    size_t num_simulations_;
    int time_steps_;
    MathUtils::RandomNumberGenerator rng_;

    // Generate stock price paths
    std::vector<double> generate_price_path(
        const MarketData& market_data,
        int steps = 1
    );

    std::vector<std::vector<double>> generate_multiple_paths(
        const MarketData& market_data,
        size_t num_paths,
        int steps = 1
    );

    // Geometric Brownian Motion simulation
    std::vector<double> simulate_gbm_path(
        double S0, double r, double q, double vol, double T, int steps
    );

    // Jump diffusion simulation (Merton model)
    std::vector<double> simulate_jump_diffusion_path(
        double S0, double r, double q, double vol, double T, int steps,
        double jump_intensity, double jump_mean, double jump_std
    );

    // Calculate payoff for different option types
    double european_payoff(const OptionContract& option, double final_price);
    double asian_payoff(
        const OptionContract& option,
        const std::vector<double>& price_path,
        const std::vector<double>& averaging_times
    );

    double barrier_payoff(
        const OptionContract& option,
        const std::vector<double>& price_path,
        double barrier_level,
        bool is_knock_out,
        bool is_up_and_out
    );

    double lookback_payoff(
        const OptionContract& option,
        const std::vector<double>& price_path
    );

    // Greeks calculation using finite differences
    Greeks calculate_greeks_finite_difference(
        const OptionContract& option,
        const MarketData& market_data
    );

    // Standard error calculation
    double calculate_standard_error(
        const std::vector<double>& payoffs,
        double mean_payoff
    );
};

class QuasiMonteCarloEngine {
public:
    QuasiMonteCarloEngine(size_t num_simulations = 100000);

    PricingResult price_option(
        const OptionContract& option,
        const MarketData& market_data
    );

private:
    size_t num_simulations_;

    // Sobol sequence generator
    class SobolGenerator {
    public:
        SobolGenerator(int dimensions);
        std::vector<double> next();
        void reset();

    private:
        int dimensions_;
        std::vector<unsigned int> state_;
        std::vector<std::vector<unsigned int>> direction_numbers_;
    };

    SobolGenerator sobol_gen_;

    // Convert uniform to normal using Box-Muller
    std::vector<double> uniform_to_normal(const std::vector<double>& uniform);
};

class LongstaffSchwartzEngine {
public:
    LongstaffSchwartzEngine(
        size_t num_simulations = 100000,
        int regression_basis_functions = 3
    );

    // American option pricing using Longstaff-Schwartz method
    PricingResult price_american_option(
        const OptionContract& option,
        const MarketData& market_data,
        int time_steps = 50
    );

private:
    size_t num_simulations_;
    int basis_functions_;
    MathUtils::RandomNumberGenerator rng_;

    // Regression basis functions
    std::vector<double> basis_function_values(double stock_price, int num_functions);

    // Least squares regression
    std::vector<double> least_squares_regression(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y
    );
};

}  // namespace OptionsEngine