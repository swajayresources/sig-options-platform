#pragma once

#include "option_types.h"
#include "math_utils.h"
#include <vector>
#include <map>
#include <functional>

namespace OptionsEngine {

struct VolatilityPoint {
    double strike;
    double expiry;
    double volatility;
    double bid_vol;
    double ask_vol;
    double mid_vol;
    double bid_price;
    double ask_price;
    double confidence;

    VolatilityPoint(double K, double T, double vol, double bid = 0, double ask = 0)
        : strike(K), expiry(T), volatility(vol), bid_vol(bid), ask_vol(ask),
          mid_vol((bid + ask) / 2.0), bid_price(0), ask_price(0), confidence(1.0) {}
};

class VolatilitySurfaceBuilder {
public:
    VolatilitySurfaceBuilder();

    void add_market_data_point(const VolatilityPoint& point);
    void add_option_quote(
        const OptionContract& option,
        double bid_price, double ask_price,
        const MarketData& market_data
    );

    VolatilitySurface build_surface();
    VolatilitySurface build_surface_with_interpolation();
    VolatilitySurface build_arbitrage_free_surface();

    // Surface fitting methods
    void fit_svi_surface();
    void fit_sabr_surface();
    void fit_heston_surface();

    // Calibration and smoothing
    void smooth_surface(double smoothing_factor = 1.0);
    void enforce_arbitrage_bounds();
    void extrapolate_wings();

    void clear_data();
    size_t get_data_points_count() const { return data_points_.size(); }

private:
    std::vector<VolatilityPoint> data_points_;
    std::vector<double> unique_strikes_;
    std::vector<double> unique_expiries_;

    // Surface parameters
    struct SVIParameters {
        double a, b, rho, m, sigma;  // SVI parameterization
    };

    struct SABRParameters {
        double alpha, beta, rho, nu;  // SABR parameters
    };

    std::map<double, SVIParameters> svi_slices_;
    std::map<double, SABRParameters> sabr_slices_;

    // Internal methods
    void extract_unique_values();
    double interpolate_volatility(double strike, double expiry) const;
    double extrapolate_volatility(double strike, double expiry) const;

    // Calibration helpers
    SVIParameters calibrate_svi_slice(double expiry, const std::vector<VolatilityPoint>& slice_data);
    SABRParameters calibrate_sabr_slice(double expiry, double forward, const std::vector<VolatilityPoint>& slice_data);

    // Arbitrage checking
    bool check_calendar_arbitrage(double strike) const;
    bool check_butterfly_arbitrage(double expiry) const;

    // Smoothing functions
    void apply_kernel_smoothing(double bandwidth);
    void apply_tension_spline_smoothing();
};

class VolatilitySurfaceInterpolator {
public:
    enum class InterpolationMethod {
        BILINEAR,
        BICUBIC,
        KRIGING,
        RADIAL_BASIS,
        THIN_PLATE_SPLINE
    };

    VolatilitySurfaceInterpolator(const VolatilitySurface& surface);

    double interpolate(double strike, double expiry, InterpolationMethod method = InterpolationMethod::BICUBIC);

    // Specialized interpolation methods
    double bilinear_interpolation(double strike, double expiry);
    double bicubic_interpolation(double strike, double expiry);
    double kriging_interpolation(double strike, double expiry);
    double radial_basis_interpolation(double strike, double expiry);
    double thin_plate_spline_interpolation(double strike, double expiry);

    // Surface derivatives
    double delta_surface(double strike, double expiry);  // d(vol)/d(strike)
    double gamma_surface(double strike, double expiry);  // d²(vol)/d(strike)²
    double theta_surface(double strike, double expiry);  // d(vol)/d(expiry)

private:
    const VolatilitySurface& surface_;

    // Grid helpers
    std::pair<int, int> find_grid_cell(double strike, double expiry) const;
    std::vector<double> get_neighboring_values(double strike, double expiry, int radius = 1) const;

    // Interpolation kernels
    double gaussian_kernel(double distance, double bandwidth);
    double cubic_kernel(double t);
    double wendland_kernel(double r, double support);
};

class VolatilitySurfaceAnalyzer {
public:
    VolatilitySurfaceAnalyzer(const VolatilitySurface& surface);

    // Surface quality metrics
    double calculate_smoothness_metric();
    double calculate_arbitrage_penalty();
    double calculate_fitting_error(const std::vector<VolatilityPoint>& market_data);

    // Risk analysis
    std::vector<double> calculate_vega_bucket_sensitivities(
        const std::vector<OptionContract>& portfolio,
        const MarketData& market_data
    );

    double calculate_volga_exposure(
        const std::vector<OptionContract>& portfolio,
        const MarketData& market_data
    );

    double calculate_vanna_exposure(
        const std::vector<OptionContract>& portfolio,
        const MarketData& market_data
    );

    // Surface diagnostics
    std::vector<std::pair<double, double>> find_arbitrage_violations();
    std::vector<std::pair<double, double>> find_outlier_points(double threshold = 3.0);

    // Scenario analysis
    VolatilitySurface shock_surface(double parallel_shift);
    VolatilitySurface twist_surface(double short_end_shift, double long_end_shift);
    VolatilitySurface skew_shock(double atm_shift, double wing_shift);

private:
    const VolatilitySurface& surface_;

    // Helper methods
    double calculate_local_curvature(double strike, double expiry);
    bool is_arbitrage_free_locally(double strike, double expiry);
    double robust_standard_deviation(const std::vector<double>& values);
};

class SabrModel {
public:
    struct Parameters {
        double alpha;  // ATM volatility
        double beta;   // CEV parameter
        double rho;    // Correlation
        double nu;     // Vol of vol

        Parameters(double a = 0.2, double b = 0.5, double r = 0.0, double n = 0.3)
            : alpha(a), beta(b), rho(r), nu(n) {}
    };

    static double calculate_volatility(
        double forward, double strike, double time_to_expiry,
        const Parameters& params
    );

    static Parameters calibrate_to_slice(
        double forward, double time_to_expiry,
        const std::vector<double>& strikes,
        const std::vector<double>& market_vols
    );

    static double calculate_alpha_from_atm_vol(
        double forward, double time_to_expiry, double atm_vol,
        double beta, double rho, double nu
    );

private:
    static double hagan_volatility_formula(
        double forward, double strike, double time_to_expiry,
        const Parameters& params
    );

    static double asymptotic_expansion_correction(
        double forward, double strike, double time_to_expiry,
        const Parameters& params
    );
};

class SviModel {
public:
    struct Parameters {
        double a, b, rho, m, sigma;

        Parameters(double a_val = 0.04, double b_val = 0.4, double rho_val = -0.4,
                  double m_val = 0.0, double sigma_val = 0.2)
            : a(a_val), b(b_val), rho(rho_val), m(m_val), sigma(sigma_val) {}
    };

    static double calculate_total_variance(double log_moneyness, const Parameters& params);
    static double calculate_volatility(double log_moneyness, double time_to_expiry, const Parameters& params);

    static Parameters calibrate_to_slice(
        const std::vector<double>& log_moneyness,
        const std::vector<double>& total_variances
    );

    // Arbitrage-free constraints
    static bool check_arbitrage_free(const Parameters& params);
    static Parameters project_to_arbitrage_free(const Parameters& params);

private:
    static double objective_function(
        const std::vector<double>& params_vec,
        const std::vector<double>& log_moneyness,
        const std::vector<double>& market_variances
    );
};

class VolatilityRiskManager {
public:
    VolatilityRiskManager(const VolatilitySurface& base_surface);

    // Risk scenario generation
    std::vector<VolatilitySurface> generate_monte_carlo_scenarios(int num_scenarios);
    std::vector<VolatilitySurface> generate_historical_scenarios(
        const std::vector<VolatilitySurface>& historical_surfaces,
        int lookback_days
    );

    // VaR calculations
    double calculate_volatility_var(
        const std::vector<OptionContract>& portfolio,
        const MarketData& market_data,
        double confidence_level = 0.95
    );

    double calculate_expected_shortfall(
        const std::vector<OptionContract>& portfolio,
        const MarketData& market_data,
        double confidence_level = 0.95
    );

    // Stress testing
    double stress_test_parallel_shift(
        const std::vector<OptionContract>& portfolio,
        const MarketData& market_data,
        double shift_amount
    );

    double stress_test_skew_rotation(
        const std::vector<OptionContract>& portfolio,
        const MarketData& market_data,
        double rotation_amount
    );

private:
    const VolatilitySurface& base_surface_;
    MathUtils::RandomNumberGenerator rng_;

    VolatilitySurface apply_shock(const VolatilitySurface& surface,
                                 const std::vector<std::vector<double>>& shock_matrix);

    double calculate_portfolio_pnl(
        const std::vector<OptionContract>& portfolio,
        const MarketData& base_market,
        const VolatilitySurface& base_surface,
        const VolatilitySurface& shocked_surface
    );
};

}  // namespace OptionsEngine