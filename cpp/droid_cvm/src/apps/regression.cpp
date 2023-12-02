#include <iostream>
#include <gtsam/3rdparty/Eigen/Eigen/Dense>
#include <custom_factors/gen/cpp/symforce/sym/squared_error_factor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/GaussianFactor.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/linear/Sampler.h>
#include <gtsam/inference/Symbol.h>
#include <memory>
#include <vector>
#include <random>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

/* We want to build a factor graph with gaussian with factors defined from the
 * gaussian factors first, and then use the function defined by the symforce to
 * get the same output.
 */
const int N = 100;
using namespace gtsam;

template<typename Scalar>
struct Parameters{
    Scalar a = 2.0f;
    Scalar b = 1.5f;
};

template<typename Scalar>
void generate_y_values(std::vector<Scalar> &x_values, Parameters<Scalar> p, std::vector<Scalar> &y_values){
    std::random_device rd;
    std::mt19937 mt_gen(rd());
    std::normal_distribution<Scalar> nd(0, 0.2);
    for (int i = 0; i < x_values.size(); i++){
        y_values.push_back(p.a*x_values[i] + p.b + nd(mt_gen));
    }
}

template<typename Scalar>
void generate_x_values(std::vector<Scalar> &x_values){
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<Scalar> ud(-100, 100);
    for( int i = 0; i < N; i++){
        x_values.push_back(ud(rng));
    }
}

template<typename Scalar>
void print_values(std::vector<Scalar> &x_values, std::vector<Scalar> &y_values){
    for (int i = 0; i < x_values.size(); i++)
        std::cout << " X value = " << x_values[i]
                  << " - Y value = " << y_values[i]
                  << std::endl;
}

class CustomRegressionFactor: public NoiseModelFactor1<Vector2>{
    double mx_, my_;
public:
    CustomRegressionFactor(Key j, double measurement_x, double measurement_y, const SharedNoiseModel &model):
        NoiseModelFactor1<Vector2>(model, j), mx_(measurement_x), my_(measurement_y){};
    ~CustomRegressionFactor() override {};

    /* evaluateError() function computes the residual
     * For a linear regression problem, we have the error defined on the measurement
     * e = my - a*mx - b, the noise model defines the probability based on the
     * noise model provided.
     * In a linear regression problem, it is the parameters we need to solve
     * for that are part of evaluate error function. We are optimizing for a and
     * b. Here in this implementation we send these parameters through a Vector2
     *
     */
    Vector evaluateError(const Vector2 &parameters, boost::optional<Matrix&> H) const override{
        Vector1 error;
        double a, b;
        a = parameters[0];
        b = parameters[1];
        error << my_ - a*mx_ - b;
        if (H){
            (*H) = (Matrix(1,2) << -mx_, -1).finished();
        }
        return error;
    }
};

// Symforce Custom Factor
class SymforceCustomRegressionFactor: public NoiseModelFactor1<Vector2>{
    double mx_, my_;
public:
    SymforceCustomRegressionFactor(Key j, double measurement_x, double measurement_y, const SharedNoiseModel &model):
        NoiseModelFactor1<Vector2>(model, j), mx_(measurement_x), my_(measurement_y){};
    ~SymforceCustomRegressionFactor() override {};

    /* evaluateError() function computes the residual
     * For a linear regression problem, we have the error defined on the measurement
     * e = my - a*mx - b, the noise model defines the probability based on the
     * noise model provided.
     * In a linear regression problem, it is the parameters we need to solve
     * for that are part of evaluate error function. We are optimizing for a and
     * b. Here in this implementation we send these parameters through a Vector2
     *
     */
    Vector evaluateError(const Vector2 &parameters, boost::optional<Matrix&> H) const override{
        double a, b;
        a = parameters[0];
        b = parameters[1];
        Vector1 error;
        Matrix12 jac;
        sym::SquaredErrorFactor<double>(mx_,my_,a, b, &error, &jac);
        if(H){
            (*H) = jac;
        }
        return error;
    }
};

template<typename FactorType>
void construct_factor_graph(const Vector &x, const Vector &y,const SharedDiagonal &noiseModel){

    NonlinearFactorGraph graph;
    Symbol s('k', 0);
    //std::cout << " value at x[10] = " << x[10] << std::endl;
    for (int i = 0; i < y.size() ; i++){
        graph.emplace_shared<FactorType>(s, x[i], y[i], noiseModel);
    }
    Values initial;
    Vector2 initial_parameters = (Vector(2) << 0, 0.4).finished();
    initial.insert(s, initial_parameters);
    std::cout << " initial solution is defined" << std::endl;
    LevenbergMarquardtParams params;
    params.print("solver parameters");
    LevenbergMarquardtOptimizer optimizer(graph, initial);
    Values result = optimizer.optimize();
    result.print("Optimized regression parameters: \n");
    boost::shared_ptr<GaussianFactorGraph> gGraph = graph.linearize(result);
    //std::cout << " Sparse Jacobian:\n" << gGraph->jacobian().first << std::endl;
    //std::cout << " Y:\n" << gGraph->jacobian().second << std::endl;
}

int main(){
    std::vector<double> x_values, y_values;
    generate_x_values<double>(x_values);
    Parameters<double> p = {
        .a = 3.0f,
        .b = 1.5f
    };
    Eigen::VectorXd x, y, e;
    double sigma = 0.1;
    // create random device to get seed
    std::random_device rd;

    // create diagonal noise model for creating n samples.
    gtsam::SharedDiagonal nPtr_N = noiseModel::Diagonal::Sigmas(
            sigma*Eigen::Matrix<double, N, 1>::Ones()
            );
    SharedDiagonal nPtr_1 = noiseModel::Diagonal::Sigmas(
            sigma*Vector1::Ones()
            );

    Sampler sampler(nPtr_N, rd());
    // noise
    e = sampler.sample();
    // create random 100 values in the range of (0, 100)
    x = 100*Eigen::Matrix<double, N, 1>::Random();

    // create noisy y values using the parameters.
    y = p.a*x + p.b*Eigen::Matrix<double, N, 1>::Ones() + e;
    construct_factor_graph<CustomRegressionFactor>(x, y, nPtr_1);
    construct_factor_graph<SymforceCustomRegressionFactor>(x, y, nPtr_1);
}
