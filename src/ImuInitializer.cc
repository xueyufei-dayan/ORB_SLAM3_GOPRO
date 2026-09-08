#include "ImuInitializer.h"
#include "ImuTypes.h"
#include "KeyFrame.h"

namespace ORB_SLAM3 {
namespace {
constexpr float kGravity = 9.80665f;
}

ImuInitializer::ImuInitializer(const std::vector<KeyFrame*>& keyframes,
                               const Eigen::Vector3f& gyro_bias)
    : ba(Eigen::Vector3f::Zero()), bg(gyro_bias), gravity(Eigen::Vector3f::Zero()),
      scale(1.f), keyframes_(keyframes), preintegrations_(keyframes.size())
{
    for(size_t i = 0; i < keyframes_.size(); ++i) {
        preintegrations_[i] = std::make_shared<IMU::Preintegrated>();
        if(keyframes_[i]->mpImuPreintegrated)
            preintegrations_[i]->CopyFrom(keyframes_[i]->mpImuPreintegrated);
    }
}

ImuInitializer::~ImuInitializer() = default;

bool ImuInitializer::Initialize()
{
    if(preintegrations_.size() < 3) return false;
    Reintegrate();
    Eigen::Vector3f average = Eigen::Vector3f::Zero();
    float displacement = 0.f;
    for(size_t i=1; i<preintegrations_.size(); ++i) {
        if(preintegrations_[i]->dT <= 0.f) return false;
        average += preintegrations_[i]->dV / preintegrations_[i]->dT;
        displacement += preintegrations_[i]->dP.norm();
    }
    average /= static_cast<float>(preintegrations_.size()-1);
    if(displacement / static_cast<float>(preintegrations_.size()-1) < .02f) return false;
    float variance = 0.f;
    for(size_t i=1; i<preintegrations_.size(); ++i)
        variance += (preintegrations_[i]->dV/preintegrations_[i]->dT-average).squaredNorm();
    if(std::sqrt(variance/static_cast<float>(preintegrations_.size()-1)) < .4f) return false;
    SolveGravityScale(); SolveScaleAccelBias(); Refine();
    return std::isfinite(scale) && gravity.allFinite() && ba.allFinite();
}

void ImuInitializer::Reintegrate()
{
    const IMU::Bias bias(ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z());
    for(size_t i=1; i<preintegrations_.size(); ++i) {
        preintegrations_[i]->SetNewBias(bias);
        preintegrations_[i]->Reintegrate();
    }
}

void ImuInitializer::SolveGravityScale()
{
    Eigen::Matrix4f A = Eigen::Matrix4f::Zero(); Eigen::Vector4f b = Eigen::Vector4f::Zero();
    for(size_t i=2; i<keyframes_.size(); ++i) {
        const auto* a=preintegrations_[i-1].get(); const auto* c=preintegrations_[i].get();
        const Sophus::SE3f p1=keyframes_[i-2]->GetImuPose(), p2=keyframes_[i-1]->GetImuPose(), p3=keyframes_[i]->GetImuPose();
        Eigen::Matrix<float,3,4> C;
        C.leftCols<3>()=-.5f*a->dT*c->dT*(a->dT+c->dT)*Eigen::Matrix3f::Identity();
        C.rightCols<1>()=a->dT*(p3.translation()-p2.translation())-c->dT*(p2.translation()-p1.translation());
        const Eigen::Vector3f d=a->dT*(p2.rotationMatrix()*c->dP)+a->dT*c->dT*(p1.rotationMatrix()*a->dV)-c->dT*(p1.rotationMatrix()*a->dP);
        A.noalias()+=C.transpose()*C; b.noalias()+=C.transpose()*d;
    }
    const Eigen::Vector4f x=A.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV).solve(b);
    gravity=x.head<3>().normalized()*kGravity; scale=x[3];
}

void ImuInitializer::SolveScaleAccelBias()
{
    Eigen::Matrix4f A=Eigen::Matrix4f::Zero(); Eigen::Vector4f b=Eigen::Vector4f::Zero();
    for(size_t i=2; i<keyframes_.size(); ++i) {
        const auto* a=preintegrations_[i-1].get(); const auto* c=preintegrations_[i].get();
        const Sophus::SE3f p1=keyframes_[i-2]->GetImuPose(), p2=keyframes_[i-1]->GetImuPose(), p3=keyframes_[i]->GetImuPose();
        Eigen::Matrix<float,3,4> C;
        C.col(0)=a->dT*(p3.translation()-p2.translation())-c->dT*(p2.translation()-p1.translation());
        C.rightCols<3>()=-(p2.rotationMatrix()*c->JPa*a->dT+p1.rotationMatrix()*a->JVa*a->dT*c->dT-p1.rotationMatrix()*a->JPa*c->dT);
        const Eigen::Vector3f d=.5f*a->dT*c->dT*(a->dT+c->dT)*gravity+a->dT*(p2.rotationMatrix()*c->dP)+a->dT*c->dT*(p1.rotationMatrix()*a->dV)-c->dT*(p1.rotationMatrix()*a->dP);
        A.noalias()+=C.transpose()*C; b.noalias()+=C.transpose()*d;
    }
    const Eigen::Vector4f x=A.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV).solve(b); scale=x[0]; ba=x.tail<3>();
}

Eigen::Matrix<float,3,2> ImuInitializer::TangentBasis(const Eigen::Vector3f& x)
{
    int d=0; if(std::abs(x.y())>std::abs(x[d])) d=1; if(std::abs(x.z())>std::abs(x[d])) d=2;
    const Eigen::Vector3f b1=x.cross(Eigen::Vector3f::Unit((d+1)%3)).normalized();
    Eigen::Matrix<float,3,2> result; result.col(0)=b1; result.col(1)=x.cross(b1).normalized(); return result;
}

void ImuInitializer::Refine()
{
    for(int iteration=0; iteration<3; ++iteration) {
        Reintegrate(); Eigen::Matrix<float,6,6> A=Eigen::Matrix<float,6,6>::Zero(); Eigen::Matrix<float,6,1> b=Eigen::Matrix<float,6,1>::Zero();
        const auto tangent=TangentBasis(gravity);
        for(size_t i=2; i<keyframes_.size(); ++i) {
            const auto* a=preintegrations_[i-1].get(); const auto* c=preintegrations_[i].get();
            const Sophus::SE3f p1=keyframes_[i-2]->GetImuPose(), p2=keyframes_[i-1]->GetImuPose(), p3=keyframes_[i]->GetImuPose();
            Eigen::Matrix<float,3,6> C;
            C.col(0)=a->dT*(p3.translation()-p2.translation())-c->dT*(p2.translation()-p1.translation());
            C.block<3,3>(0,1)=-(p2.rotationMatrix()*c->JPa*a->dT+p1.rotationMatrix()*a->JVa*a->dT*c->dT-p1.rotationMatrix()*a->JPa*c->dT);
            C.rightCols<2>()=-.5f*a->dT*c->dT*(a->dT+c->dT)*tangent;
            const Eigen::Vector3f d=.5f*a->dT*c->dT*(a->dT+c->dT)*gravity+a->dT*(p2.rotationMatrix()*c->dP)+a->dT*c->dT*(p1.rotationMatrix()*a->dV)-c->dT*(p1.rotationMatrix()*a->dP);
            A.noalias()+=C.transpose()*C; b.noalias()+=C.transpose()*d;
        }
        const auto x=A.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV).solve(b);
        scale=x[0]; ba+=.1f*x.segment<3>(1); gravity=(gravity+.1f*tangent*x.tail<2>()).normalized()*kGravity;
    }
}
} // namespace ORB_SLAM3
