#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/SO3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <sym/pose3.h>
#include <sym/rot3.h>
#include <gtest/gtest.h>
#include <iostream>

// Exploratory tests to understand the maths.
TEST(gtsamRot3, PropertiesTest){
}
TEST(gtsamPose3, PropertiesTest){
    // check the properties of pose3
    // adjoint, exponential map,
}

TEST(symPose3, PropertiesTest){
}

TEST(SymAndGtsamType, ComparisonTest){
    // Jacobian comparison and properties comparison
}

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
