#ifndef ORB_SLAM3_IMU_INITIALIZER_H
#define ORB_SLAM3_IMU_INITIALIZER_H

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace ORB_SLAM3 {
class KeyFrame;
namespace IMU { class Preintegrated; }

// Eigen-only VIG-Init scale, gravity and accelerometer-bias solver.
class ImuInitializer {
public:
    ImuInitializer(const std::vector<KeyFrame*>& keyframes, const Eigen::Vector3f& gyro_bias);
    ~ImuInitializer();
    bool Initialize();

    Eigen::Vector3f ba, bg, gravity;
    float scale;

private:
    void Reintegrate();
    void SolveGravityScale();
    void SolveScaleAccelBias();
    void Refine();
    static Eigen::Matrix<float,3,2> TangentBasis(const Eigen::Vector3f& vector);

    std::vector<KeyFrame*> keyframes_;
    std::vector<std::shared_ptr<IMU::Preintegrated>> preintegrations_;
};
} // namespace ORB_SLAM3
#endif
