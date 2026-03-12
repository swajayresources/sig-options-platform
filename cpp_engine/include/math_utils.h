#pragma once

#include <cmath>
#include <vector>
#include <random>

namespace OptionsEngine {
namespace MathUtils {

class NormalDistribution {
public:
    static double pdf(double x);
    static double cdf(double x);
    static double inverse_cdf(double p);

private:
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double SQRT_2PI = 2.5066282746310005024;
};

class RandomNumberGenerator {
public:
    RandomNumberGenerator(unsigned int seed = std::random_device{}());

    double uniform();
    double normal();
    std::vector<double> normal_vector(size_t n);
    std::vector<double> antithetic_normal_vector(size_t n);

private:
    std::mt19937 generator_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;
};

class NumericalMethods {
public:
    static double newton_raphson(
        std::function<double(double)> f,
        std::function<double(double)> df,
        double initial_guess,
        double tolerance = 1e-8,
        int max_iterations = 100
    );

    static double bisection(
        std::function<double(double)> f,
        double a, double b,
        double tolerance = 1e-8,
        int max_iterations = 100
    );

    static double brent_method(
        std::function<double(double)> f,
        double a, double b,
        double tolerance = 1e-8,
        int max_iterations = 100
    );
};

class MatrixOperations {
public:
    static std::vector<std::vector<double>> cholesky_decomposition(
        const std::vector<std::vector<double>>& matrix
    );

    static std::vector<double> matrix_vector_multiply(
        const std::vector<std::vector<double>>& matrix,
        const std::vector<double>& vector
    );

    static std::vector<std::vector<double>> matrix_multiply(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B
    );
};

class Interpolation {
public:
    static double linear_interpolation(
        double x, double x1, double x2, double y1, double y2
    );

    static double bilinear_interpolation(
        double x, double y,
        double x1, double x2, double y1, double y2,
        double f11, double f12, double f21, double f22
    );

    class CubicSpline {
    public:
        CubicSpline(const std::vector<double>& x, const std::vector<double>& y);
        double interpolate(double x) const;

    private:
        std::vector<double> x_, y_, b_, c_, d_;
    };
};

constexpr double EPSILON = 1e-12;
constexpr double LARGE_NUMBER = 1e10;

}  // namespace MathUtils
}  // namespace OptionsEngine