/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    SFMdata.h
 * @brief   Simple example for the structure-from-motion problems
 * @author  Duy-Nguyen Ta
 */

/**
 * A structure-from-motion example with landmarks, default arguments give:
 *  - The landmarks form a 10 meter cube
 *  - The robot rotates around the landmarks, always facing towards the cube
 * Passing function argument allows to specify an initial position, a pose
 * increment and step count.
 */

#pragma once

// As this is a full 3D problem, we will use Pose3 variables to represent the
// camera positions and Point3 variables (x, y, z) to represent the landmark
// coordinates. Camera observations of landmarks (i.e. pixel coordinates) will
// be stored as Point2 (x, y).
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

// We will also need a camera object to hold calibration information and perform
// projections.
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>

namespace gtsam {

/// Create a set of ground-truth landmarks
std::vector<Point3> createPoints() {
  std::vector<Point3> points;
  points.push_back(Point3(10.0, 10.0, 10.0));
  points.push_back(Point3(-10.0, 10.0, 10.0));
  points.push_back(Point3(-10.0, -10.0, 10.0));
  points.push_back(Point3(10.0, -10.0, 10.0));
  points.push_back(Point3(10.0, 10.0, -10.0));
  points.push_back(Point3(-10.0, 10.0, -10.0));
  points.push_back(Point3(-10.0, -10.0, -10.0));
  points.push_back(Point3(10.0, -10.0, -10.0));

  return points;
}

/**
 * Create a set of ground-truth poses
 * Default values give a circular trajectory, radius 30 at pi/4 intervals,
 * always facing the circle center
 */
std::vector<Pose3> createPoses(
    const Pose3& init = Pose3(Rot3::Ypr(M_PI_2, 0, -M_PI_2), {30, 0, 0}),
    const Pose3& delta = Pose3(Rot3::Ypr(0, -M_PI_4, 0),
                               {sin(M_PI_4) * 30, 0, 30 * (1 - sin(M_PI_4))}),
    int steps = 8) {
  std::vector<Pose3> poses;
  poses.reserve(steps);

  poses.push_back(init);
  for (int i = 1; i < steps; ++i) {
    poses.push_back(poses[i - 1].compose(delta));
  }

  return poses;
}

/**
 * Create regularly spaced poses with specified radius and number of cameras
 */
std::vector<Pose3> posesOnCircle(int num_cameras = 8, double R = 30) {
  const double theta = 2 * M_PI / num_cameras;

  // Initial pose at angle 0, position (R, 0, 0), facing the center with Y-axis
  // pointing down
  const Pose3 init(Rot3::Ypr(M_PI_2, 0, -M_PI_2), {R, 0, 0});

  // Delta rotation: rotate by -theta around Z-axis (counterclockwise movement)
  Rot3 delta_rotation = Rot3::Ypr(0, -theta, 0);

  // Delta translation in world frame
  Vector3 delta_translation_world(R * (cos(theta) - 1), R * sin(theta), 0);

  // Transform delta translation to local frame of the camera
  Vector3 delta_translation_local =
      init.rotation().inverse() * delta_translation_world;

  // Define delta pose
  const Pose3 delta(delta_rotation, delta_translation_local);

  // Generate poses using createPoses
  return createPoses(init, delta, num_cameras);
}
}  // namespace gtsam