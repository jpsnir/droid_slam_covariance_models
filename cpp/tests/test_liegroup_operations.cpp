#include <GeographicLib/Config.h>
#include <GeographicLib/LocalCartesian.hpp>
#include <gtest/gtest.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/SO3.h>
#include <gtsam/navigation/GPSFactor.h>
#include <iostream>

using namespace gtsam;
using namespace GeographicLib;

#if GEOGRAPHICLIB_VERSION_MINOR < 37
static const auto &kWGS84 = Geocentric::WGS84;
#else
static const auto &kWGS84 = Geocentric::WGS84();
#endif


// Numerical derivative usage understanding

double fun(const Vector2 &x){
    return cos(x[0]) + sin(x[1]);
}

Vector3 f3(const double x1, const double x2, const double x3) {
  Vector3 result;
  result << sin(x1)*x3, cos(x2)*x3, x3*x2*x1;
  return result;
}


TEST(numerDerivative, Usage){
    Vector2 x(1, 2);
    Vector e(2); // expected
    e << -sin(x[0]), cos(x[1]);
    auto f = [](const Vector2 &x)->double{
        // define function
        return cos(x[0]) + sin(x[1]);
    };
    Matrix numH;
    double value;

    // Defined by a lambda function or function pointer
    // boost::function as first argument that returns and the
    // second argument is where the numerical derivative need to be computed.
    // The function is templated therefore can be used for any type.
    numH = numericalGradient<Vector2>(f, x);
    ASSERT_TRUE(numH.isApprox(e, 1e-5));

    // Hessian computation
    // Hessian = [partial(f,x1,x1), partial(f,x1,x2)
    //            partial(f,x2,x1), partial(f,x2,x2)
    Matrix22 eHessian;
    eHessian << -cos(x[0]), 0,
                0       , -sin(x[1]);
    // Directly using the lambda function in hessian
    // was giving a compiler error, it was an ambiguous call
    auto numHessian = numericalHessian<Vector2>(fun, x);
    ASSERT_TRUE(eHessian.isApprox(numHessian, 1e-5));

    // 3 variable function
    double d1 = 2, d2 = 3, d3 = 1;
    // This gives only the first column of the jacobian matrix
    // Likewise we can test all other columns of jacobian matrix
    // with numericalDerivate32, numericalDerivative 33
    // The tests written here are mostly for vector spaces.
    // For elements which are lie groups or manifolds, the
    // expected Jacobian is computed using the concepts of lie algebra and
    // manifolds discussed in other tests
    Matrix num3H = numericalDerivative31<Vector3, double, double, double>(f3, d1, d2, d3);
    Matrix31 actualH  = (Matrix(3,1) << cos(d1)*d3, 0, d2*d3).finished();
    ASSERT_EQ(num3H.rows(), 3);
    ASSERT_EQ(num3H.cols(), 1);
    std::cout << "\n Actual H = " << actualH
              << "\n Expected H =" << num3H << std::endl;
    ASSERT_TRUE(actualH.isApprox(num3H, 1e-5));


    Rot3 R = Rot3::RzRyRx(M_PI/2, 0, M_PI/2);
}

// Implementing gps factor test to understand numerical derivatives
TEST(GPSFactor, Constructor) {
  const double lat0 = 33.86998, lon0 = -84.30626, h0 = 274;
  LocalCartesian origin_ENU(lat0, lon0, h0);
  const double lat = 33.87071, lon = -84.30482, h = 274;
  double E, N, U;
  origin_ENU.Forward(lat, lon, h, E, N, U);
  EXPECT_NEAR(133.24, E, 1e-2);
  EXPECT_NEAR(80.98, N, 1e-2);
  EXPECT_NEAR(0, U, 1e-2);

  Key key(1);
  SharedNoiseModel model = noiseModel::Isotropic::Sigma(3, 0.25);
  // factor defined by key, measurement and noise model
  GPSFactor factor(key, Point3(E, N, U), model);
  Pose3 T(Rot3::RzRyRx(0.15, -0.30, 0.45), Point3(E, N, U));
  std::cout << Z_3x1 << std::endl;
  std::cout << "factor eval: " << factor.evaluateError(T);
  Matrix expectedH = numericalDerivative11<Vector, Pose3>(
      [&factor](const Pose3 &T) { return factor.evaluateError(T); }, T);
  Matrix actualH;
  factor.evaluateError(T, actualH);
  std::cout << "\nActual jacobian: " << actualH
            << "\n expected jacobian: " << expectedH << std::endl;
  ASSERT_TRUE(actualH.isApprox(expectedH, 1e-5));

  // Test gps factor 2
  GPSFactor2 factor2(key, Point3(E, N, U), model);
  NavState T1(Rot3::RzRyRx(0.15, -0.30, 0.45), Point3(E, N, U), Vector3::Zero());
  Matrix expectedH2 = numericalDerivative11<Vector, NavState>(
      [&factor2](const NavState &T1) { return factor2.evaluateError(T1); }, T1);
  Matrix actualH2;
  factor2.evaluateError(T1, actualH2);
  std::cout << "\n Nav state : " << std::endl;
  std::cout << "\nActual jacobian: " << actualH2
            << "\n expected jacobian: " << expectedH2 << std::endl;
  ASSERT_TRUE(actualH2.isApprox(expectedH2, 1e-5));

}


// Exploratory tests to understand the maths.
TEST(gtsamPose3, PropertiesTest) {
  // check the properties of pose3
  // adjoint, exponential map,
  static const Point3 P(1,1,0);
  static const Rot3 R = Rot3::Rodrigues(0.3,0,0);
  Matrix63 actualH1, actualH2;
  Pose3 actual = Pose3::Create(R, P,actualH1, actualH2 );
  Pose3 T(R, P);
  ASSERT_TRUE(actual.equals(T));
  std::cout << "\n pose3 H1 : \n" << actualH1 << "\n" << actualH2 << std::endl;

  // Adjoint map
  Vector xi = (Vector(6) << 0.01, 0.02, 0.03, 1.0, 2.0, 3.0).finished();
  auto adj = Pose3::adjointMap(xi);
  std::cout << "\n Vector6\n " << xi
            << "\n adjoint map: \n " << adj << std::endl;

  // retract first order
  Pose3 p;
  // zero vector
  Vector v0 = Vector::Zero(6, 1);
  Vector v = Z_6x1;
  ASSERT_TRUE(v0 == v);
  v(0) = 0.3;
  Pose3 p1(R, Point3(0,0,0));
  ASSERT_TRUE(p1.equals(p.retract(v),1e-2));
  std::cout << " Log map" << std::endl;
  std::cout << Pose3::Logmap(p1) << std::endl;
  std::cout << " Exp map \n";
  std::cout << Pose3::Expmap(v) << std::endl;
  expmap_default<Pose3>(p, v);
  std::cout << "\n Rotation from local coordinates: " <<  Rot3::Expmap(v.head(3));

}
TEST(gtsamPose3, NumericalDerivativeTest) {}

TEST(SymAndGtsamType, ComparisonTest) {
  // Jacobian comparison and properties comparison
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
