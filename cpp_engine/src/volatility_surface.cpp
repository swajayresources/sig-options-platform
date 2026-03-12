#include "volatility_surface.h"
#include "black_scholes.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <numeric>

namespace OptionsEngine {

// VolatilitySurface implementation
double VolatilitySurface::getVolatility(double strike, double expiry) const {
    if (strikes.empty() || expiries.empty()) {
        return 0.2;  // Default 20% volatility
    }

    // Find surrounding grid points
    auto strike_it = std::lower_bound(strikes.begin(), strikes.end(), strike);
    auto expiry_it = std::lower_bound(expiries.begin(), expiries.end(), expiry);

    size_t strike_idx = std::min(static_cast<size_t>(strike_it - strikes.begin()), strikes.size() - 1);
    size_t expiry_idx = std::min(static_cast<size_t>(expiry_it - expiries.begin()), expiries.size() - 1);

    if (strike_idx > 0 && strike < strikes[strike_idx]) strike_idx--;
    if (expiry_idx > 0 && expiry < expiries[expiry_idx]) expiry_idx--;

    // Boundary cases
    if (strike_idx >= volatilities.size() || expiry_idx >= volatilities[strike_idx].size()) {
        return volatilities.empty() ? 0.2 : volatilities.back().back();
    }

    // Simple bilinear interpolation
    if (strike_idx == strikes.size() - 1 || expiry_idx == volatilities[strike_idx].size() - 1) {
        return volatilities[strike_idx][expiry_idx];
    }

    double k1 = strikes[strike_idx], k2 = strikes[strike_idx + 1];
    double t1 = expiries[expiry_idx], t2 = expiries[expiry_idx + 1];

    double v11 = volatilities[strike_idx][expiry_idx];
    double v12 = volatilities[strike_idx][expiry_idx + 1];
    double v21 = volatilities[strike_idx + 1][expiry_idx];
    double v22 = volatilities[strike_idx + 1][expiry_idx + 1];

    return MathUtils::Interpolation::bilinear_interpolation(
        strike, expiry, k1, k2, t1, t2, v11, v12, v21, v22
    );
}

void VolatilitySurface::updatePoint(double strike, double expiry, double vol) {
    // Find or create the point in the surface
    auto strike_it = std::find(strikes.begin(), strikes.end(), strike);
    auto expiry_it = std::find(expiries.begin(), expiries.end(), expiry);

    size_t strike_idx, expiry_idx;

    if (strike_it == strikes.end()) {
        strikes.push_back(strike);
        std::sort(strikes.begin(), strikes.end());
        strike_idx = std::find(strikes.begin(), strikes.end(), strike) - strikes.begin();
    } else {
        strike_idx = strike_it - strikes.begin();
    }

    if (expiry_it == expiries.end()) {
        expiries.push_back(expiry);
        std::sort(expiries.begin(), expiries.end());
        expiry_idx = std::find(expiries.begin(), expiries.end(), expiry) - expiries.begin();
    } else {
        expiry_idx = expiry_it - expiries.begin();
    }

    // Resize volatilities matrix if needed
    if (volatilities.size() <= strike_idx) {
        volatilities.resize(strike_idx + 1);
    }
    if (volatilities[strike_idx].size() <= expiry_idx) {
        volatilities[strike_idx].resize(expiry_idx + 1);
    }

    volatilities[strike_idx][expiry_idx] = vol;
}

// VolatilitySurfaceBuilder implementation
VolatilitySurfaceBuilder::VolatilitySurfaceBuilder() {}

void VolatilitySurfaceBuilder::add_market_data_point(const VolatilityPoint& point) {
    data_points_.push_back(point);
}

void VolatilitySurfaceBuilder::add_option_quote(
    const OptionContract& option,
    double bid_price, double ask_price,
    const MarketData& market_data) {

    if (bid_price <= 0 || ask_price <= 0 || ask_price < bid_price) {
        return;  // Invalid quote
    }

    double mid_price = (bid_price + ask_price) / 2.0;

    // Calculate implied volatility for bid, ask, and mid prices
    double bid_vol = BlackScholesModel::calculate_implied_volatility(
        option, market_data, bid_price
    );
    double ask_vol = BlackScholesModel::calculate_implied_volatility(
        option, market_data, ask_price
    );
    double mid_vol = BlackScholesModel::calculate_implied_volatility(
        option, market_data, mid_price
    );

    if (bid_vol > 0 && ask_vol > 0 && mid_vol > 0) {
        VolatilityPoint point(option.strike, option.expiry, mid_vol, bid_vol, ask_vol);
        point.bid_price = bid_price;
        point.ask_price = ask_price;
        point.confidence = 1.0 / (ask_price - bid_price);  // Tighter spreads = higher confidence

        add_market_data_point(point);
    }
}

VolatilitySurface VolatilitySurfaceBuilder::build_surface() {
    extract_unique_values();

    VolatilitySurface surface;
    surface.strikes = unique_strikes_;
    surface.expiries = unique_expiries_;

    // Initialize surface matrices
    surface.volatilities.resize(unique_strikes_.size());
    surface.bid_vol.resize(unique_strikes_.size());
    surface.ask_vol.resize(unique_strikes_.size());

    for (size_t i = 0; i < unique_strikes_.size(); ++i) {
        surface.volatilities[i].resize(unique_expiries_.size(), 0.2);  // Default 20%
        surface.bid_vol[i].resize(unique_expiries_.size(), 0.18);
        surface.ask_vol[i].resize(unique_expiries_.size(), 0.22);
    }

    // Fill in known data points
    for (const auto& point : data_points_) {
        auto strike_it = std::find(unique_strikes_.begin(), unique_strikes_.end(), point.strike);
        auto expiry_it = std::find(unique_expiries_.begin(), unique_expiries_.end(), point.expiry);

        if (strike_it != unique_strikes_.end() && expiry_it != unique_expiries_.end()) {
            size_t strike_idx = strike_it - unique_strikes_.begin();
            size_t expiry_idx = expiry_it - unique_expiries_.begin();

            surface.volatilities[strike_idx][expiry_idx] = point.volatility;
            surface.bid_vol[strike_idx][expiry_idx] = point.bid_vol;
            surface.ask_vol[strike_idx][expiry_idx] = point.ask_vol;
        }
    }

    return surface;
}

VolatilitySurface VolatilitySurfaceBuilder::build_surface_with_interpolation() {
    VolatilitySurface surface = build_surface();

    // Interpolate missing points using radial basis functions
    for (size_t i = 0; i < unique_strikes_.size(); ++i) {
        for (size_t j = 0; j < unique_expiries_.size(); ++j) {
            if (surface.volatilities[i][j] == 0.2) {  // Default value indicates missing data
                double interpolated_vol = interpolate_volatility(unique_strikes_[i], unique_expiries_[j]);
                surface.volatilities[i][j] = interpolated_vol;
            }
        }
    }

    return surface;
}

double VolatilitySurfaceBuilder::interpolate_volatility(double strike, double expiry) const {
    if (data_points_.empty()) {
        return 0.2;
    }

    // Weighted average based on distance
    double weighted_sum = 0.0;
    double weight_sum = 0.0;

    for (const auto& point : data_points_) {
        double strike_dist = std::abs(strike - point.strike) / strike;
        double expiry_dist = std::abs(expiry - point.expiry) / expiry;
        double distance = std::sqrt(strike_dist * strike_dist + expiry_dist * expiry_dist);

        double weight = 1.0 / (1.0 + distance * distance);
        weighted_sum += weight * point.volatility;
        weight_sum += weight;
    }

    return weight_sum > 0 ? weighted_sum / weight_sum : 0.2;
}

void VolatilitySurfaceBuilder::extract_unique_values() {
    std::set<double> strike_set, expiry_set;

    for (const auto& point : data_points_) {
        strike_set.insert(point.strike);
        expiry_set.insert(point.expiry);
    }

    unique_strikes_.assign(strike_set.begin(), strike_set.end());
    unique_expiries_.assign(expiry_set.begin(), expiry_set.end());
}

void VolatilitySurfaceBuilder::fit_svi_surface() {
    extract_unique_values();

    for (double expiry : unique_expiries_) {
        std::vector<VolatilityPoint> slice_data;
        for (const auto& point : data_points_) {
            if (std::abs(point.expiry - expiry) < 1e-6) {
                slice_data.push_back(point);
            }
        }

        if (slice_data.size() >= 5) {  // Need at least 5 points for SVI
            SVIParameters params = calibrate_svi_slice(expiry, slice_data);
            svi_slices_[expiry] = params;
        }
    }
}

VolatilitySurfaceBuilder::SVIParameters VolatilitySurfaceBuilder::calibrate_svi_slice(
    double expiry, const std::vector<VolatilityPoint>& slice_data) {

    // Simplified SVI calibration - in practice, use non-linear optimization
    SVIParameters params;

    if (slice_data.empty()) {
        return params;
    }

    // Find ATM point
    double spot = 100.0;  // Assuming normalized
    double min_dist = std::numeric_limits<double>::max();
    double atm_vol = 0.2;

    for (const auto& point : slice_data) {
        double dist = std::abs(point.strike - spot);
        if (dist < min_dist) {
            min_dist = dist;
            atm_vol = point.volatility;
        }
    }

    // Simple parameter estimation
    params.a = atm_vol * atm_vol * expiry * 0.5;
    params.b = 0.5;
    params.rho = -0.3;
    params.m = 0.0;
    params.sigma = 0.3;

    return params;
}

void VolatilitySurfaceBuilder::smooth_surface(double smoothing_factor) {
    // Apply kernel smoothing to reduce noise
    apply_kernel_smoothing(smoothing_factor);
}

void VolatilitySurfaceBuilder::apply_kernel_smoothing(double bandwidth) {
    std::vector<VolatilityPoint> smoothed_points;

    for (const auto& center_point : data_points_) {
        double weighted_sum = 0.0;
        double weight_sum = 0.0;

        for (const auto& point : data_points_) {
            double strike_dist = (point.strike - center_point.strike) / center_point.strike;
            double expiry_dist = (point.expiry - center_point.expiry) / center_point.expiry;
            double distance = std::sqrt(strike_dist * strike_dist + expiry_dist * expiry_dist);

            double weight = std::exp(-0.5 * (distance / bandwidth) * (distance / bandwidth));
            weighted_sum += weight * point.volatility;
            weight_sum += weight;
        }

        VolatilityPoint smoothed_point = center_point;
        smoothed_point.volatility = weight_sum > 0 ? weighted_sum / weight_sum : center_point.volatility;
        smoothed_points.push_back(smoothed_point);
    }

    data_points_ = smoothed_points;
}

void VolatilitySurfaceBuilder::clear_data() {
    data_points_.clear();
    unique_strikes_.clear();
    unique_expiries_.clear();
    svi_slices_.clear();
    sabr_slices_.clear();
}

// SABR Model implementation
double SabrModel::calculate_volatility(
    double forward, double strike, double time_to_expiry,
    const Parameters& params) {

    return hagan_volatility_formula(forward, strike, time_to_expiry, params);
}

double SabrModel::hagan_volatility_formula(
    double forward, double strike, double time_to_expiry,
    const Parameters& params) {

    if (time_to_expiry <= 0 || params.alpha <= 0) {
        return 0.01;
    }

    double beta = params.beta;
    double alpha = params.alpha;
    double rho = params.rho;
    double nu = params.nu;

    if (std::abs(forward - strike) < 1e-8) {
        // ATM case
        double term1 = alpha / std::pow(forward, 1.0 - beta);
        double term2 = 1.0 + time_to_expiry * (
            (1.0 - beta) * (1.0 - beta) * alpha * alpha / (24.0 * std::pow(forward, 2.0 - 2.0 * beta)) +
            0.25 * rho * beta * nu * alpha / std::pow(forward, 1.0 - beta) +
            (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        );

        return term1 * term2;
    }

    // General case
    double log_fk = std::log(forward / strike);
    double fk_beta = std::pow(forward * strike, (1.0 - beta) / 2.0);

    double z = nu / alpha * fk_beta * log_fk;
    double x_z = z;

    if (std::abs(z) > 1e-8) {
        double sqrt_term = std::sqrt(1.0 - 2.0 * rho * z + z * z);
        x_z = std::log((sqrt_term + z - rho) / (1.0 - rho));
    }

    double numerator = alpha * (1.0 + time_to_expiry * (
        (1.0 - beta) * (1.0 - beta) * alpha * alpha / (24.0 * fk_beta * fk_beta) +
        0.25 * rho * beta * nu * alpha / fk_beta +
        (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    ));

    double denominator = fk_beta * (1.0 + (1.0 - beta) * (1.0 - beta) * log_fk * log_fk / 24.0 +
                                   std::pow((1.0 - beta) * log_fk, 4) / 1920.0) * (z / x_z);

    return numerator / denominator;
}

SabrModel::Parameters SabrModel::calibrate_to_slice(
    double forward, double time_to_expiry,
    const std::vector<double>& strikes,
    const std::vector<double>& market_vols) {

    // Simplified calibration - in practice, use optimization libraries
    Parameters params;

    if (strikes.empty() || market_vols.empty()) {
        return params;
    }

    // Find ATM volatility
    double atm_vol = market_vols[0];
    for (size_t i = 0; i < strikes.size(); ++i) {
        if (std::abs(strikes[i] - forward) < std::abs(strikes[0] - forward)) {
            atm_vol = market_vols[i];
        }
    }

    params.beta = 0.5;  // Common choice
    params.alpha = calculate_alpha_from_atm_vol(forward, time_to_expiry, atm_vol,
                                               params.beta, 0.0, 0.3);
    params.rho = -0.3;   // Typical negative correlation
    params.nu = 0.3;     // Moderate vol-of-vol

    return params;
}

double SabrModel::calculate_alpha_from_atm_vol(
    double forward, double time_to_expiry, double atm_vol,
    double beta, double rho, double nu) {

    double factor = 1.0 + time_to_expiry * (
        0.25 * rho * beta * nu / std::pow(forward, 1.0 - beta) +
        (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    );

    return atm_vol * std::pow(forward, 1.0 - beta) / factor;
}

// SVI Model implementation
double SviModel::calculate_total_variance(double log_moneyness, const Parameters& params) {
    double k = log_moneyness;
    return params.a + params.b * (params.rho * (k - params.m) +
                                 std::sqrt((k - params.m) * (k - params.m) + params.sigma * params.sigma));
}

double SviModel::calculate_volatility(double log_moneyness, double time_to_expiry, const Parameters& params) {
    if (time_to_expiry <= 0) {
        return 0.01;
    }

    double total_variance = calculate_total_variance(log_moneyness, params);
    return std::sqrt(std::max(total_variance / time_to_expiry, 1e-8));
}

bool SviModel::check_arbitrage_free(const Parameters& params) {
    // Check SVI arbitrage-free conditions
    if (params.a < 0) return false;
    if (params.b < 0) return false;
    if (std::abs(params.rho) >= 1.0) return false;
    if (params.sigma <= 0) return false;

    // Additional Gatheral conditions
    double discriminant = params.sigma * params.sigma + (params.rho * params.rho - 1.0) * params.b * params.b;
    if (discriminant < 0) return false;

    return true;
}

// VolatilitySurfaceAnalyzer implementation
VolatilitySurfaceAnalyzer::VolatilitySurfaceAnalyzer(const VolatilitySurface& surface)
    : surface_(surface) {}

double VolatilitySurfaceAnalyzer::calculate_smoothness_metric() {
    if (surface_.volatilities.empty()) {
        return 0.0;
    }

    double total_curvature = 0.0;
    int count = 0;

    for (size_t i = 1; i < surface_.strikes.size() - 1; ++i) {
        for (size_t j = 1; j < surface_.expiries.size() - 1; ++j) {
            double curvature = calculate_local_curvature(surface_.strikes[i], surface_.expiries[j]);
            total_curvature += curvature * curvature;
            count++;
        }
    }

    return count > 0 ? total_curvature / count : 0.0;
}

double VolatilitySurfaceAnalyzer::calculate_local_curvature(double strike, double expiry) {
    // Calculate discrete Laplacian as a measure of local curvature
    VolatilitySurfaceInterpolator interpolator(surface_);

    double h = 0.01;  // Small step size
    double center = interpolator.interpolate(strike, expiry);
    double up = interpolator.interpolate(strike, expiry + h);
    double down = interpolator.interpolate(strike, expiry - h);
    double left = interpolator.interpolate(strike - h, expiry);
    double right = interpolator.interpolate(strike + h, expiry);

    return std::abs(up + down + left + right - 4.0 * center) / (h * h);
}

std::vector<std::pair<double, double>> VolatilitySurfaceAnalyzer::find_arbitrage_violations() {
    std::vector<std::pair<double, double>> violations;

    // Check calendar arbitrage: total variance should be increasing in time
    for (size_t i = 0; i < surface_.strikes.size(); ++i) {
        for (size_t j = 0; j < surface_.expiries.size() - 1; ++j) {
            double vol1 = surface_.volatilities[i][j];
            double vol2 = surface_.volatilities[i][j + 1];
            double t1 = surface_.expiries[j];
            double t2 = surface_.expiries[j + 1];

            if (vol1 * vol1 * t1 > vol2 * vol2 * t2) {
                violations.push_back({surface_.strikes[i], surface_.expiries[j]});
            }
        }
    }

    return violations;
}

VolatilitySurface VolatilitySurfaceAnalyzer::shock_surface(double parallel_shift) {
    VolatilitySurface shocked_surface = surface_;

    for (size_t i = 0; i < shocked_surface.volatilities.size(); ++i) {
        for (size_t j = 0; j < shocked_surface.volatilities[i].size(); ++j) {
            shocked_surface.volatilities[i][j] += parallel_shift;
            shocked_surface.volatilities[i][j] = std::max(shocked_surface.volatilities[i][j], 0.001);
        }
    }

    return shocked_surface;
}

}  // namespace OptionsEngine