#pragma once

#include <vector>
#include <memory>
#include <string>

namespace OptionsEngine {

enum class OptionType {
    CALL,
    PUT
};

enum class ExerciseType {
    EUROPEAN,
    AMERICAN,
    BERMUDAN
};

enum class PricingModel {
    BLACK_SCHOLES,
    BINOMIAL,
    TRINOMIAL,
    MONTE_CARLO,
    HESTON,
    SABR,
    JUMP_DIFFUSION
};

struct MarketData {
    double spot_price;
    double risk_free_rate;
    double dividend_yield;
    double volatility;
    double time_to_expiry;

    MarketData(double S, double r, double q, double vol, double T)
        : spot_price(S), risk_free_rate(r), dividend_yield(q),
          volatility(vol), time_to_expiry(T) {}
};

struct OptionContract {
    OptionType type;
    ExerciseType exercise_type;
    double strike;
    double expiry;
    std::string underlying;

    OptionContract(OptionType opt_type, ExerciseType ex_type,
                  double K, double T, const std::string& symbol)
        : type(opt_type), exercise_type(ex_type), strike(K),
          expiry(T), underlying(symbol) {}
};

struct Greeks {
    double delta;      // Price sensitivity to underlying
    double gamma;      // Delta sensitivity to underlying
    double theta;      // Price sensitivity to time
    double vega;       // Price sensitivity to volatility
    double rho;        // Price sensitivity to interest rate

    // Second-order Greeks
    double vomma;      // Vega sensitivity to volatility (volga)
    double vanna;      // Vega sensitivity to underlying
    double charm;      // Delta sensitivity to time
    double color;      // Gamma sensitivity to time
    double speed;      // Gamma sensitivity to underlying
    double ultima;     // Vomma sensitivity to volatility

    Greeks() : delta(0), gamma(0), theta(0), vega(0), rho(0),
               vomma(0), vanna(0), charm(0), color(0), speed(0), ultima(0) {}
};

struct PricingResult {
    double price;
    Greeks greeks;
    double implied_volatility;
    bool success;
    std::string error_message;

    PricingResult() : price(0), implied_volatility(0), success(false) {}
};

struct VolatilitySurface {
    std::vector<double> strikes;
    std::vector<double> expiries;
    std::vector<std::vector<double>> volatilities;
    std::vector<std::vector<double>> bid_vol;
    std::vector<std::vector<double>> ask_vol;

    double getVolatility(double strike, double expiry) const;
    void updatePoint(double strike, double expiry, double vol);
};

}  // namespace OptionsEngine