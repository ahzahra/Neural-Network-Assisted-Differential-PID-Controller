/*
  Note:
  This code is adapted from code written by members of PRECISE Lab at the University of Pennsylvania

*/

#ifndef __SO3_CONTROL_H__
#define __SO3_CONTROL_H__

#include <Eigen/Geometry>

class differentialPIDControl
{
 public:
  differentialPIDControl();

  void setMass(const float mass);
  void setGravity(const float g);
  void setPosition(const Eigen::Vector3f &position);
  void setVelocity(const Eigen::Vector3f &velocity);
  void setYaw(const float current_yaw);
  void setMaxIntegral(const float max_integral);
  void resetIntegrals(void);

  void calculateControl(const Eigen::Vector3f &des_pos,
                        const Eigen::Vector3f &des_vel,
                        const Eigen::Vector3f &des_acc,
                        const float des_yaw,
                        const Eigen::Vector3f &kx,
                        const Eigen::Vector3f &kv,
                        const Eigen::Vector3f &ki,
                        const float ki_yaw);


  const Eigen::Vector4f &getControls(void);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 private:
  // Inputs for the controller
  float mass_; 
  float g_; // gravitaitonal constant
  Eigen::Vector3f pos_; // [x,y,z] position
  Eigen::Vector3f vel_; // [x,y,z] velocities
  float current_yaw_; 
  Eigen::Vector3f pos_int_; // position integral accumulator
  float yaw_int_;
  float max_pos_int_;

  // Outputs of the controller
  Eigen::Vector4f trpy_; // Thrust, roll, pitch, yaw
};

#endif
