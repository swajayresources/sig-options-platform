#include "math_utils.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace OptionsEngine {
namespace MathUtils {

// Normal Distribution Implementation
double NormalDistribution::pdf(double x) {
    return std::exp(-0.5 * x * x) / SQRT_2PI;
}

double NormalDistribution::cdf(double x) {
    // Abramowitz and Stegun approximation
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    int sign = (x >= 0) ? 1 : -1;
    x = std::abs(x) / std::sqrt(2.0);

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

double NormalDistribution::inverse_cdf(double p) {
    if (p <= 0.0 || p >= 1.0) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }

    // Beasley-Springer-Moro algorithm
    const double a0 = -3.969683028665376e+01;
    const double a1 =  2.209460984245205e+02;
    const double a2 = -2.759285104469687e+02;
    const double a3 =  1.383577518672690e+02;
    const double a4 = -3.066479806614716e+01;
    const double a5 =  2.506628277459239e+00;

    const double b1 = -5.447609879822406e+01;
    const double b2 =  1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 =  6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;

    const double c0 = -7.784894002430293e-03;
    const double c1 = -3.223964580411365e-01;
    const double c2 = -2.400758277161838e+00;
    const double c3 = -2.549732539343734e+00;
    const double c4 =  4.374664141464968e+00;
    const double c5 =  2.938163982698783e+00;

    const double d1 =  7.784695709041462e-03;
    const double d2 =  3.224671290700398e-01;
    const double d3 =  2.445134137142996e+00;
    const double d4 =  3.754408661907416e+00;

    double x, r;

    if (p > 0.5) {
        r = std::sqrt(-std::log(1.0 - p));
        if (r <= 5.0) {
            r -= 1.6;
            x = (((((c5 * r + c4) * r + c3) * r + c2) * r + c1) * r + c0) /
                ((((d4 * r + d3) * r + d2) * r + d1) * r + 1.0);
        } else {
            r -= 5.0;
            x = (((((c5 * r + c4) * r + c3) * r + c2) * r + c1) * r + c0) /
                ((((d4 * r + d3) * r + d2) * r + d1) * r + 1.0);
        }
    } else {
        r = std::sqrt(-std::log(p));
        if (r <= 5.0) {
            r -= 1.6;
            x = -(((((c5 * r + c4) * r + c3) * r + c2) * r + c1) * r + c0) /
                 ((((d4 * r + d3) * r + d2) * r + d1) * r + 1.0);
        } else {
            r -= 5.0;
            x = -(((((c5 * r + c4) * r + c3) * r + c2) * r + c1) * r + c0) /
                 ((((d4 * r + d3) * r + d2) * r + d1) * r + 1.0);
        }
    }

    return x;
}

// Random Number Generator Implementation
RandomNumberGenerator::RandomNumberGenerator(unsigned int seed)
    : generator_(seed), uniform_dist_(0.0, 1.0), normal_dist_(0.0, 1.0) {}

double RandomNumberGenerator::uniform() {
    return uniform_dist_(generator_);
}

double RandomNumberGenerator::normal() {
    return normal_dist_(generator_);
}

std::vector<double> RandomNumberGenerator::normal_vector(size_t n) {
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = normal();
    }
    return result;
}

std::vector<double> RandomNumberGenerator::antithetic_normal_vector(size_t n) {
    std::vector<double> result(n);
    for (size_t i = 0; i < n; i += 2) {
        double z = normal();
        result[i] = z;
        if (i + 1 < n) {
            result[i + 1] = -z;  // Antithetic variate
        }
    }
    return result;
}

// Numerical Methods Implementation
double NumericalMethods::newton_raphson(
    std::function<double(double)> f,
    std::function<double(double)> df,
    double initial_guess,
    double tolerance,
    int max_iterations) {

    double x = initial_guess;
    for (int i = 0; i < max_iterations; ++i) {
        double fx = f(x);
        double dfx = df(x);

        if (std::abs(dfx) < EPSILON) {
            throw std::runtime_error("Derivative too small in Newton-Raphson");
        }

        double x_new = x - fx / dfx;

        if (std::abs(x_new - x) < tolerance) {
            return x_new;
        }

        x = x_new;
    }

    throw std::runtime_error("Newton-Raphson failed to converge");
}

double NumericalMethods::bisection(
    std::function<double(double)> f,
    double a, double b,
    double tolerance,
    int max_iterations) {

    if (f(a) * f(b) > 0) {
        throw std::invalid_argument("Function must have opposite signs at endpoints");
    }

    for (int i = 0; i < max_iterations; ++i) {
        double c = (a + b) / 2.0;
        double fc = f(c);

        if (std::abs(fc) < tolerance || (b - a) / 2.0 < tolerance) {
            return c;
        }

        if (f(a) * fc < 0) {
            b = c;
        } else {
            a = c;
        }
    }

    throw std::runtime_error("Bisection method failed to converge");
}

double NumericalMethods::brent_method(
    std::function<double(double)> f,
    double a, double b,
    double tolerance,
    int max_iterations) {

    double fa = f(a);
    double fb = f(b);

    if (fa * fb > 0) {
        throw std::invalid_argument("Function must have opposite signs at endpoints");
    }

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    bool mflag = true;
    double d = 0;

    for (int i = 0; i < max_iterations; ++i) {
        double s;

        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) +
                b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Check conditions for using bisection instead
        if (((s < (3 * a + b) / 4 || s > b)) ||
            (mflag && std::abs(s - b) >= std::abs(b - c) / 2) ||
            (!mflag && std::abs(s - b) >= std::abs(c - d) / 2) ||
            (mflag && std::abs(b - c) < tolerance) ||
            (!mflag && std::abs(c - d) < tolerance)) {

            s = (a + b) / 2;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        if (std::abs(fb) < tolerance || std::abs(b - a) < tolerance) {
            return b;
        }
    }

    throw std::runtime_error("Brent's method failed to converge");
}

// Matrix Operations Implementation
std::vector<std::vector<double>> MatrixOperations::cholesky_decomposition(
    const std::vector<std::vector<double>>& matrix) {

    size_t n = matrix.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            if (i == j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += L[j][k] * L[j][k];
                }
                L[j][j] = std::sqrt(matrix[j][j] - sum);
            } else {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (matrix[i][j] - sum) / L[j][j];
            }
        }
    }

    return L;
}

// Interpolation Implementation
double Interpolation::linear_interpolation(
    double x, double x1, double x2, double y1, double y2) {

    if (std::abs(x2 - x1) < EPSILON) {
        return y1;
    }

    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
}

double Interpolation::bilinear_interpolation(
    double x, double y,
    double x1, double x2, double y1, double y2,
    double f11, double f12, double f21, double f22) {

    double t = (x - x1) / (x2 - x1);
    double u = (y - y1) / (y2 - y1);

    return (1 - t) * (1 - u) * f11 +
           t * (1 - u) * f21 +
           (1 - t) * u * f12 +
           t * u * f22;
}

Interpolation::CubicSpline::CubicSpline(
    const std::vector<double>& x, const std::vector<double>& y)
    : x_(x), y_(y) {

    size_t n = x.size();
    if (n < 2) {
        throw std::invalid_argument("Need at least 2 points for spline interpolation");
    }

    b_.resize(n);
    c_.resize(n);
    d_.resize(n);

    // Natural spline boundary conditions
    std::vector<double> h(n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
    }

    std::vector<double> alpha(n - 1);
    for (size_t i = 1; i < n - 1; ++i) {
        alpha[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    std::vector<double> l(n), mu(n), z(n);
    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;

    for (size_t i = 1; i < n - 1; ++i) {
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n - 1] = 1.0;
    z[n - 1] = 0.0;
    c_[n - 1] = 0.0;

    for (int j = n - 2; j >= 0; --j) {
        c_[j] = z[j] - mu[j] * c_[j + 1];
        b_[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c_[j + 1] + 2.0 * c_[j]) / 3.0;
        d_[j] = (c_[j + 1] - c_[j]) / (3.0 * h[j]);
    }
}

double Interpolation::CubicSpline::interpolate(double x) const {
    size_t n = x_.size();
    size_t i = 0;

    // Find the interval
    for (i = 0; i < n - 1; ++i) {
        if (x <= x_[i + 1]) break;
    }

    if (i == n - 1) i = n - 2;  // Extrapolation

    double dx = x - x_[i];
    return y_[i] + b_[i] * dx + c_[i] * dx * dx + d_[i] * dx * dx * dx;
}

}  // namespace MathUtils
}  // namespace OptionsEngine