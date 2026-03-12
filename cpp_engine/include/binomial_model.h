#pragma once

#include "option_types.h"
#include <vector>

namespace OptionsEngine {

class BinomialModel {
public:
    BinomialModel(int steps = 100);

    PricingResult price_option(
        const OptionContract& option,
        const MarketData& market_data
    );

    Greeks calculate_greeks(
        const OptionContract& option,
        const MarketData& market_data
    );

    // For American options with early exercise
    PricingResult price_american_option(
        const OptionContract& option,
        const MarketData& market_data
    );

    void set_steps(int steps) { steps_ = steps; }
    int get_steps() const { return steps_; }

private:
    int steps_;

    struct BinomialParameters {
        double up_factor;
        double down_factor;
        double risk_neutral_prob;
        double discount_factor;
    };

    BinomialParameters calculate_parameters(const MarketData& market_data) const;

    std::vector<double> build_stock_tree(
        double spot_price,
        const BinomialParameters& params
    ) const;

    std::vector<double> build_option_tree_european(
        const OptionContract& option,
        const std::vector<double>& stock_prices,
        const BinomialParameters& params
    ) const;

    std::vector<double> build_option_tree_american(
        const OptionContract& option,
        const std::vector<double>& stock_prices,
        const BinomialParameters& params
    ) const;

    double intrinsic_value(const OptionContract& option, double stock_price) const;

    // Greeks calculation using finite differences on the tree
    double calculate_delta_from_tree(
        const OptionContract& option,
        const MarketData& market_data
    );

    double calculate_gamma_from_tree(
        const OptionContract& option,
        const MarketData& market_data
    );

    double calculate_theta_from_tree(
        const OptionContract& option,
        const MarketData& market_data
    );
};

class TrinomialModel {
public:
    TrinomialModel(int steps = 100);

    PricingResult price_option(
        const OptionContract& option,
        const MarketData& market_data
    );

    Greeks calculate_greeks(
        const OptionContract& option,
        const MarketData& market_data
    );

    PricingResult price_american_option(
        const OptionContract& option,
        const MarketData& market_data
    );

private:
    int steps_;

    struct TrinomialParameters {
        double up_factor;
        double down_factor;
        double up_prob;
        double middle_prob;
        double down_prob;
        double discount_factor;
    };

    TrinomialParameters calculate_parameters(const MarketData& market_data) const;

    std::vector<std::vector<double>> build_stock_tree(
        double spot_price,
        const TrinomialParameters& params
    ) const;

    std::vector<std::vector<double>> build_option_tree_european(
        const OptionContract& option,
        const std::vector<std::vector<double>>& stock_tree,
        const TrinomialParameters& params
    ) const;

    std::vector<std::vector<double>> build_option_tree_american(
        const OptionContract& option,
        const std::vector<std::vector<double>>& stock_tree,
        const TrinomialParameters& params
    ) const;
};

class AdaptiveMeshModel {
public:
    AdaptiveMeshModel(int initial_steps = 50, double convergence_tolerance = 1e-4);

    PricingResult price_option(
        const OptionContract& option,
        const MarketData& market_data
    );

    // Automatically refines mesh until convergence
    PricingResult price_with_adaptive_refinement(
        const OptionContract& option,
        const MarketData& market_data,
        int max_steps = 1000
    );

private:
    int initial_steps_;
    double convergence_tolerance_;

    bool has_converged(double price1, double price2) const;
};

}  // namespace OptionsEngine