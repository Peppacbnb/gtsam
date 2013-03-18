/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation, 
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    testConcurrentBatchFilter.cpp
 * @brief   Unit tests for the Concurrent Batch Filter
 * @author  Stephen Williams (swilliams8@gatech.edu)
 * @date    Jan 5, 2013
 */

#include <gtsam_unstable/nonlinear/ConcurrentBatchFilter.h>
#include <gtsam_unstable/nonlinear/LinearizedFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Ordering.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/Key.h>
#include <gtsam/inference/JunctionTree.h>
#include <gtsam/geometry/Pose3.h>
#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;

namespace {

// Set up initial pose, odometry difference, loop closure difference, and initialization errors
const Pose3 poseInitial;
const Pose3 poseOdometry( Rot3::RzRyRx(Vector_(3, 0.05, 0.10, -0.75)), Point3(1.0, -0.25, 0.10) );
const Pose3 poseError( Rot3::RzRyRx(Vector_(3, 0.01, 0.02, -0.1)), Point3(0.05, -0.05, 0.02) );

// Set up noise models for the factors
const SharedDiagonal noisePrior = noiseModel::Isotropic::Sigma(6, 0.10);
const SharedDiagonal noiseOdometery = noiseModel::Diagonal::Sigmas(Vector_(6, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5));
const SharedDiagonal noiseLoop = noiseModel::Diagonal::Sigmas(Vector_(6, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0));

// Create a derived class to allow testing protected member functions
class ConcurrentBatchFilterTester : public ConcurrentBatchFilter {
public:
  ConcurrentBatchFilterTester(const LevenbergMarquardtParams& parameters, double lag) : ConcurrentBatchFilter(parameters, lag) { };
  virtual ~ConcurrentBatchFilterTester() { };

  // Add accessors to the protected members
  void presync() { ConcurrentBatchFilter::presync(); };
  void getSummarizedFactors(NonlinearFactorGraph& summarizedFactors, Values& rootValues) { ConcurrentBatchFilter::getSummarizedFactors(summarizedFactors, rootValues); };
  void getSmootherFactors(NonlinearFactorGraph& smootherFactors, Values& smootherValues) { ConcurrentBatchFilter::getSmootherFactors(smootherFactors, smootherValues); };
  void synchronize(const NonlinearFactorGraph& summarizedFactors) { ConcurrentBatchFilter::synchronize(summarizedFactors); };
  void postsync() { ConcurrentBatchFilter::postsync(); };
};

/* ************************************************************************* */
bool hessian_equal(const NonlinearFactorGraph& expected, const NonlinearFactorGraph& actual, const Values& theta, double tol = 1e-9) {

  FastSet<Key> expectedKeys = expected.keys();
  FastSet<Key> actualKeys = actual.keys();

  // Verify the set of keys in both graphs are the same
  if(!std::equal(expectedKeys.begin(), expectedKeys.end(), actualKeys.begin()))
    return false;

  // Create an ordering
  Ordering ordering;
  BOOST_FOREACH(Key key, expectedKeys) {
    ordering.push_back(key);
  }

  // Linearize each factor graph
  GaussianFactorGraph expectedGaussian;
  BOOST_FOREACH(const NonlinearFactor::shared_ptr& factor, expected) {
    if(factor)
      expectedGaussian.push_back( factor->linearize(theta, ordering) );
  }
  GaussianFactorGraph actualGaussian;
  BOOST_FOREACH(const NonlinearFactor::shared_ptr& factor, actual) {
    if(factor)
      actualGaussian.push_back( factor->linearize(theta, ordering) );
  }

  // Convert linear factor graph into a dense Hessian
  Matrix expectedHessian = expectedGaussian.augmentedHessian();
  Matrix actualHessian = actualGaussian.augmentedHessian();

  // Zero out the lower-right entry. This corresponds to a constant in the optimization,
  // which does not affect the result. Further, in conversions between Jacobians and Hessians,
  // this term is ignored.
  expectedHessian(expectedHessian.rows()-1, expectedHessian.cols()-1) = 0.0;
  actualHessian(actualHessian.rows()-1, actualHessian.cols()-1) = 0.0;

  // Compare Hessians
  return assert_equal(expectedHessian, actualHessian, tol);
}

///* ************************************************************************* */
void CreateFactors(NonlinearFactorGraph& graph, Values& theta, ConcurrentBatchFilter::KeyTimestampMap& timestamps, size_t index1 = 0, size_t index2 = 1) {

  // Calculate all poses
  Pose3 poses[20];
  poses[0] = poseInitial;
  for(size_t index = 1; index < 20; ++index) {
    poses[index] = poses[index-1].compose(poseOdometry);
  }

  // Create all keys
  Key keys[20];
  for(size_t index = 0; index < 20; ++index) {
    keys[index] = Symbol('X', index);
  }

  // Create factors that will form a specific tree structure
  // Loop over the included timestamps
  for(size_t index = index1; index < index2; ++index) {

    switch(index) {
      case 0:
      {
        graph.add(PriorFactor<Pose3>(keys[0], poses[0], noisePrior));
        // Add new variables
        theta.insert(keys[0], poses[0].compose(poseError));
        timestamps[keys[0]] = double(0);
        break;
      }
      case 1:
      {
        // Add odometry factor between 0 and 1
        Pose3 poseDelta = poses[0].between(poses[1]);
        graph.add(BetweenFactor<Pose3>(keys[0], keys[1], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[1], poses[1].compose(poseError));
        timestamps[keys[1]] = double(1);
        break;
      }
      case 2:
      {
        break;
      }
      case 3:
      {
        // Add odometry factor between 1 and 3
        Pose3 poseDelta = poses[1].between(poses[3]);
        graph.add(BetweenFactor<Pose3>(keys[1], keys[3], poseDelta, noiseOdometery));
        // Add odometry factor between 2 and 3
        poseDelta = poses[2].between(poses[3]);
        graph.add(BetweenFactor<Pose3>(keys[2], keys[3], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[2], poses[2].compose(poseError));
        timestamps[keys[2]] = double(2);
        theta.insert(keys[3], poses[3].compose(poseError));
        timestamps[keys[3]] = double(3);
        break;
      }
      case 4:
      {
        break;
      }
      case 5:
      {
        // Add odometry factor between 3 and 5
        Pose3 poseDelta = poses[3].between(poses[5]);
        graph.add(BetweenFactor<Pose3>(keys[3], keys[5], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[5], poses[5].compose(poseError));
        timestamps[keys[5]] = double(5);
        break;
      }
      case 6:
      {
        // Add odometry factor between 3 and 6
        Pose3 poseDelta = poses[3].between(poses[6]);
        graph.add(BetweenFactor<Pose3>(keys[3], keys[6], poseDelta, noiseOdometery));
        // Add odometry factor between 5 and 6
        poseDelta = poses[5].between(poses[6]);
        graph.add(BetweenFactor<Pose3>(keys[5], keys[6], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[6], poses[6].compose(poseError));
        timestamps[keys[6]] = double(6);
        break;
      }
      case 7:
      {
        // Add odometry factor between 4 and 7
        Pose3 poseDelta = poses[4].between(poses[7]);
        graph.add(BetweenFactor<Pose3>(keys[4], keys[7], poseDelta, noiseOdometery));
        // Add odometry factor between 6 and 7
        poseDelta = poses[6].between(poses[7]);
        graph.add(BetweenFactor<Pose3>(keys[6], keys[7], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[4], poses[4].compose(poseError));
        timestamps[keys[4]] = double(4);
        theta.insert(keys[7], poses[7].compose(poseError));
        timestamps[keys[7]] = double(7);
        break;
      }
      case 8:
        break;

      case 9:
      {
        // Add odometry factor between 6 and 9
        Pose3 poseDelta = poses[6].between(poses[9]);
        graph.add(BetweenFactor<Pose3>(keys[6], keys[9], poseDelta, noiseOdometery));
        // Add odometry factor between 7 and 9
        poseDelta = poses[7].between(poses[9]);
        graph.add(BetweenFactor<Pose3>(keys[7], keys[9], poseDelta, noiseOdometery));
        // Add odometry factor between 8 and 9
        poseDelta = poses[8].between(poses[9]);
        graph.add(BetweenFactor<Pose3>(keys[8], keys[9], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[8], poses[8].compose(poseError));
        timestamps[keys[8]] = double(8);
        theta.insert(keys[9], poses[9].compose(poseError));
        timestamps[keys[9]] = double(9);
        break;
      }
      case 10:
      {
        // Add odometry factor between 9 and 10
        Pose3 poseDelta = poses[9].between(poses[10]);
        graph.add(BetweenFactor<Pose3>(keys[9], keys[10], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[10], poses[10].compose(poseError));
        timestamps[keys[10]] = double(10);
        break;
      }
      case 11:
      {
        // Add odometry factor between 10 and 11
        Pose3 poseDelta = poses[10].between(poses[11]);
        graph.add(BetweenFactor<Pose3>(keys[10], keys[11], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[11], poses[11].compose(poseError));
        timestamps[keys[11]] = double(11);
        break;
      }
      case 12:
      {
        // Add odometry factor between 7 and 12
        Pose3 poseDelta = poses[7].between(poses[12]);
        graph.add(BetweenFactor<Pose3>(keys[7], keys[12], poseDelta, noiseOdometery));
        // Add odometry factor between 9 and 12
        poseDelta = poses[9].between(poses[12]);
        graph.add(BetweenFactor<Pose3>(keys[9], keys[12], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[12], poses[12].compose(poseError));
        timestamps[keys[12]] = double(12);
        break;
      }





      case 13:
      {
        // Add odometry factor between 10 and 13
        Pose3 poseDelta = poses[10].between(poses[13]);
        graph.add(BetweenFactor<Pose3>(keys[10], keys[13], poseDelta, noiseOdometery));
        // Add odometry factor between 12 and 13
        poseDelta = poses[12].between(poses[13]);
        graph.add(BetweenFactor<Pose3>(keys[12], keys[13], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[13], poses[13].compose(poseError));
        timestamps[keys[13]] = double(13);
        break;
      }
      case 14:
      {
        // Add odometry factor between 11 and 14
        Pose3 poseDelta = poses[11].between(poses[14]);
        graph.add(BetweenFactor<Pose3>(keys[11], keys[14], poseDelta, noiseOdometery));
        // Add odometry factor between 13 and 14
        poseDelta = poses[13].between(poses[14]);
        graph.add(BetweenFactor<Pose3>(keys[13], keys[14], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[14], poses[14].compose(poseError));
        timestamps[keys[14]] = double(14);
        break;
      }
      case 15:
        break;

      case 16:
      {
        // Add odometry factor between 13 and 16
        Pose3 poseDelta = poses[13].between(poses[16]);
        graph.add(BetweenFactor<Pose3>(keys[13], keys[16], poseDelta, noiseOdometery));
        // Add odometry factor between 14 and 16
        poseDelta = poses[14].between(poses[16]);
        graph.add(BetweenFactor<Pose3>(keys[14], keys[16], poseDelta, noiseOdometery));
        // Add odometry factor between 15 and 16
        poseDelta = poses[15].between(poses[16]);
        graph.add(BetweenFactor<Pose3>(keys[15], keys[16], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[15], poses[15].compose(poseError));
        timestamps[keys[15]] = double(15);
        theta.insert(keys[16], poses[16].compose(poseError));
        timestamps[keys[16]] = double(16);
        break;
      }
      case 17:
      {
        // Add odometry factor between 16 and 17
        Pose3 poseDelta = poses[16].between(poses[17]);
        graph.add(BetweenFactor<Pose3>(keys[16], keys[17], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[17], poses[17].compose(poseError));
        timestamps[keys[17]] = double(17);
        break;
      }
      case 18:
      {
        // Add odometry factor between 17 and 18
        Pose3 poseDelta = poses[17].between(poses[18]);
        graph.add(BetweenFactor<Pose3>(keys[17], keys[18], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[18], poses[18].compose(poseError));
        timestamps[keys[18]] = double(18);
        break;
      }
      case 19:
      {
        // Add odometry factor between 14 and 19
        Pose3 poseDelta = poses[14].between(poses[19]);
        graph.add(BetweenFactor<Pose3>(keys[14], keys[19], poseDelta, noiseOdometery));
        // Add odometry factor between 16 and 19
        poseDelta = poses[16].between(poses[19]);
        graph.add(BetweenFactor<Pose3>(keys[16], keys[19], poseDelta, noiseOdometery));
        // Add new variables
        theta.insert(keys[19], poses[19].compose(poseError));
        timestamps[keys[19]] = double(19);
        break;
      }

    }
  }

  return;
}

/* ************************************************************************* */
Values BatchOptimize(const NonlinearFactorGraph& graph, const Values& theta, const Values& rootValues = Values()) {

  // Create an L-M optimizer
  LevenbergMarquardtParams parameters;
  parameters.linearSolverType = SuccessiveLinearizationParams::MULTIFRONTAL_QR;

  LevenbergMarquardtOptimizer optimizer(graph, theta, parameters);

  // Use a custom optimization loop so the linearization points can be controlled
  double currentError;
  do {
    // Force variables associated with root keys to keep the same linearization point
    if(rootValues.size() > 0) {
      // Put the old values of the root keys back into the optimizer state
      optimizer.state().values.update(rootValues);
      // Update the error value with the new theta
      optimizer.state().error = graph.error(optimizer.state().values);
    }

    // Do next iteration
    currentError = optimizer.error();
    optimizer.iterate();

  } while(optimizer.iterations() < parameters.maxIterations &&
      !checkConvergence(parameters.relativeErrorTol, parameters.absoluteErrorTol,
          parameters.errorTol, currentError, optimizer.error(), parameters.verbosity));

  // return the final optimized values
  return optimizer.values();
}

/* ************************************************************************* */
void FindFactorsWithAny(const std::set<Key>& keys, const NonlinearFactorGraph& sourceFactors, NonlinearFactorGraph& destinationFactors) {

  BOOST_FOREACH(const NonlinearFactor::shared_ptr& factor, sourceFactors) {
    NonlinearFactor::const_iterator key = factor->begin();
    while((key != factor->end()) && (!std::binary_search(keys.begin(), keys.end(), *key))) {
      ++key;
    }
    if(key != factor->end()) {
      destinationFactors.push_back(factor);
    }
  }

}

/* ************************************************************************* */
void FindFactorsWithOnly(const std::set<Key>& keys, const NonlinearFactorGraph& sourceFactors, NonlinearFactorGraph& destinationFactors) {

  BOOST_FOREACH(const NonlinearFactor::shared_ptr& factor, sourceFactors) {
    NonlinearFactor::const_iterator key = factor->begin();
    while((key != factor->end()) && (std::binary_search(keys.begin(), keys.end(), *key))) {
      ++key;
    }
    if(key == factor->end()) {
      destinationFactors.push_back(factor);
    }
  }

}

/* ************************************************************************* */
typedef BayesTree<GaussianConditional,ISAM2Clique>::sharedClique Clique;
void SymbolicPrintTree(const Clique& clique, const Ordering& ordering, const std::string indent = "") {
  std::cout << indent << "P( ";
  BOOST_FOREACH(Index index, clique->conditional()->frontals()){
    std::cout << DefaultKeyFormatter(ordering.key(index)) << " ";
  }
  if(clique->conditional()->nrParents() > 0) {
    std::cout << "| ";
  }
  BOOST_FOREACH(Index index, clique->conditional()->parents()){
    std::cout << DefaultKeyFormatter(ordering.key(index)) << " ";
  }
  std::cout << ")" << std::endl;

  BOOST_FOREACH(const Clique& child, clique->children()) {
    SymbolicPrintTree(child, ordering, indent+"  ");
  }
}

}

/* ************************************************************************* */
TEST_UNSAFE( ConcurrentBatchFilter, update_Batch )
{
  // Test the 'update' function of the ConcurrentBatchFilter in a nonlinear environment.
  // Thus, a full L-M optimization and the ConcurrentBatchFilter results should be identical
  // This tests adds all of the factors to the filter at once (i.e. batch)

  // Create a set of optimizer parameters
  LevenbergMarquardtParams parameters;
  double lag = 4.0;

  // Create a Concurrent Batch Filter
  ConcurrentBatchFilter filter(parameters, lag);

  // Create containers to keep the full graph
  Values fullTheta;
  NonlinearFactorGraph fullGraph;
  ConcurrentBatchFilter::KeyTimestampMap fullTimestamps;

  // Create all factors
  CreateFactors(fullGraph, fullTheta, fullTimestamps, 0, 20);

  // Optimize with Concurrent Batch Filter
  filter.update(fullGraph, fullTheta, fullTimestamps);
  Values actual = filter.calculateEstimate();

  // Optimize with L-M
  Values expected = BatchOptimize(fullGraph, fullTheta);

  // Check smoother versus batch
  CHECK(assert_equal(expected, actual, 1e-4));
}

/* ************************************************************************* */
TEST_UNSAFE( ConcurrentBatchFilter, update_Incremental )
{
  // Test the 'update' function of the ConcurrentBatchFilter in a nonlinear environment.
  // Thus, a full L-M optimization and the ConcurrentBatchFilter results should be identical
  // This tests adds the factors to the filter as they are created (i.e. incrementally)

  // Create a set of optimizer parameters
  LevenbergMarquardtParams parameters;
  double lag = 4.0;

  // Create a Concurrent Batch Filter
  ConcurrentBatchFilter filter(parameters, lag);

  // Create containers to keep the full graph
  Values fullTheta;
  NonlinearFactorGraph fullGraph;

  // Add odometry from time 0 to time 10
  for(size_t i = 0; i < 20; ++i) {
    // Create containers to keep the new factors
    Values newTheta;
    NonlinearFactorGraph newGraph;
    ConcurrentBatchFilter::KeyTimestampMap newTimestamps;

    // Create factors
    CreateFactors(newGraph, newTheta, newTimestamps, i, i+1);

    // Add these entries to the filter
    filter.update(newGraph, newTheta, newTimestamps);
    Values actual = filter.calculateEstimate();

    // Add these entries to the full batch version
    fullGraph.push_back(newGraph);
    fullTheta.insert(newTheta);
    Values expected = BatchOptimize(fullGraph, fullTheta);
    fullTheta = expected;

    // Compare filter solution with full batch
    CHECK(assert_equal(expected, actual, 1e-4));
  }

}

/* ************************************************************************* */
TEST_UNSAFE( ConcurrentBatchFilter, synchronize )
{
  // Test the 'synchronize' function of the ConcurrentBatchFilter in a nonlinear environment.
  // The filter is operating on a known tree structure, so the factors and summarization can
  // be predicted for testing purposes

  // Create a set of optimizer parameters
  LevenbergMarquardtParams parameters;
  double lag = 4.0;

  // Create a Concurrent Batch Filter
  ConcurrentBatchFilterTester filter(parameters, lag);

  // Create containers to keep the full graph
  Values newTheta, fullTheta;
  NonlinearFactorGraph newGraph, fullGraph;
  ConcurrentBatchFilter::KeyTimestampMap newTimestamps;

  // Create all factors
  CreateFactors(newGraph, newTheta, newTimestamps, 0, 13);
  fullTheta.insert(newTheta);
  fullGraph.push_back(newGraph);

  // Optimize with Concurrent Batch Filter
  filter.update(newGraph, newTheta, newTimestamps);
  Values updatedTheta = filter.calculateEstimate();

  // Eliminate the factor graph into a Bayes Tree to create the summarized factors
  // Create Ordering
  std::map<Key, int> constraints;
  constraints[Symbol('X',  8)] = 1;
  constraints[Symbol('X',  9)] = 1;
  constraints[Symbol('X', 10)] = 1;
  constraints[Symbol('X', 11)] = 1;
  constraints[Symbol('X', 12)] = 1;
  Ordering ordering = *fullGraph.orderingCOLAMDConstrained(fullTheta, constraints);
  // Linearize into a Gaussian Factor Graph
  GaussianFactorGraph linearGraph = *fullGraph.linearize(fullTheta, ordering);
  // Eliminate into a Bayes Net with iSAM2-type cliques
  JunctionTree<GaussianFactorGraph, ISAM2Clique> jt(linearGraph);
  ISAM2Clique::shared_ptr root = jt.eliminate(EliminateQR);
  BayesTree<GaussianConditional, ISAM2Clique> bayesTree;
  bayesTree.insert(root);

  //  std::cout << "BAYES TREE:" << std::endl;
  //  SymbolicPrintTree(root, ordering.invert(), "  ");

  //  BAYES TREE:
  //    P( X7 X9 X12 )
  //      P( X10 | X9 )
  //        P( X11 | X10 )
  //      P( X8 | X9 )
  //      P( X6 | X7 X9 )
  //        P( X3 X5 | X6 )
  //          P( X2 | X3 )
  //          P( X1 | X3 )
  //            P( X0 | X1 )
  //      P( X4 | X7 )

  // Extract the nonlinear factors that should be sent to the smoother
  std::vector<Key> smootherKeys;
  smootherKeys.push_back(Symbol('X',  0));
  smootherKeys.push_back(Symbol('X',  1));
  smootherKeys.push_back(Symbol('X',  2));
  smootherKeys.push_back(Symbol('X',  3));
  smootherKeys.push_back(Symbol('X',  4));
  smootherKeys.push_back(Symbol('X',  5));
  smootherKeys.push_back(Symbol('X',  6));
  NonlinearFactorGraph expectedSmootherFactors;
  BOOST_FOREACH(const NonlinearFactor::shared_ptr& factor, fullGraph) {
    if(std::find_first_of(factor->begin(), factor->end(), smootherKeys.begin(), smootherKeys.end()) != factor->end()) {
      expectedSmootherFactors.push_back(factor);
    }
  }

  // Extract smoother values
  Values expectedSmootherValues;
  expectedSmootherValues.insert(Symbol('X',  0), updatedTheta.at(Symbol('X',  0)));
  expectedSmootherValues.insert(Symbol('X',  1), updatedTheta.at(Symbol('X',  1)));
  expectedSmootherValues.insert(Symbol('X',  2), updatedTheta.at(Symbol('X',  2)));
  expectedSmootherValues.insert(Symbol('X',  3), updatedTheta.at(Symbol('X',  3)));
  expectedSmootherValues.insert(Symbol('X',  4), updatedTheta.at(Symbol('X',  4)));
  expectedSmootherValues.insert(Symbol('X',  5), updatedTheta.at(Symbol('X',  5)));
  expectedSmootherValues.insert(Symbol('X',  6), updatedTheta.at(Symbol('X',  6)));

  // Extract the filter summarized factors
  // Cached factors that represent the filter side (i.e. the X8 and X10 clique)
  NonlinearFactorGraph expectedFilterSumarization;
  expectedFilterSumarization.add(LinearizedJacobianFactor(boost::static_pointer_cast<JacobianFactor>(bayesTree.nodes().at(ordering.at(Symbol('X',  8)))->cachedFactor()), ordering, fullTheta));
  expectedFilterSumarization.add(LinearizedJacobianFactor(boost::static_pointer_cast<JacobianFactor>(bayesTree.nodes().at(ordering.at(Symbol('X', 10)))->cachedFactor()), ordering, fullTheta));
  // And any factors that involve only the root (X7, X9, X12)
  std::vector<Key> rootKeys;
  rootKeys.push_back(Symbol('X',  7));
  rootKeys.push_back(Symbol('X',  9));
  rootKeys.push_back(Symbol('X', 12));
  BOOST_FOREACH(const NonlinearFactor::shared_ptr& factor, fullGraph) {
    size_t count = 0;
    BOOST_FOREACH(Key key, *factor) {
      if(std::binary_search(rootKeys.begin(), rootKeys.end(), key)) ++count;
    }
    if(count == factor->size()) expectedFilterSumarization.push_back(factor);
  }

  // Extract the new root values
  Values expectedRootValues;
  expectedRootValues.insert(Symbol('X',  7), updatedTheta.at(Symbol('X',  7)));
  expectedRootValues.insert(Symbol('X',  9), updatedTheta.at(Symbol('X',  9)));
  expectedRootValues.insert(Symbol('X', 12), updatedTheta.at(Symbol('X', 12)));



  // Start the synchronization process
  NonlinearFactorGraph actualSmootherFactors, actualFilterSumarization, smootherSummarization;
  Values actualSmootherValues, actualRootValues;
  filter.presync();
  filter.synchronize(smootherSummarization);  // Supplying an empty factor graph here
  filter.getSmootherFactors(actualSmootherFactors, actualSmootherValues);
  filter.getSummarizedFactors(actualFilterSumarization, actualRootValues);
  filter.postsync();



  // Compare filter sync variables versus the expected
  CHECK(hessian_equal(expectedSmootherFactors, actualSmootherFactors, updatedTheta, 1e-8));
  CHECK(assert_equal(expectedSmootherValues, actualSmootherValues, 1e-4));
  CHECK(hessian_equal(expectedFilterSumarization, actualFilterSumarization, updatedTheta, 1e-9));
  CHECK(assert_equal(expectedRootValues, actualRootValues, 1e-4));






  // Now add additional factors to the filter and re-sync
  newGraph.resize(0);  newTheta.clear();  newTimestamps.clear();
  CreateFactors(newGraph, newTheta, newTimestamps, 13, 20);
  fullTheta.insert(newTheta);
  fullGraph.push_back(newGraph);

  // Optimize with Concurrent Batch Filter
  filter.update(newGraph, newTheta, newTimestamps);
  updatedTheta = filter.calculateEstimate();

  // Eliminate the factor graph into a Bayes Tree to create the summarized factors
  // Create Ordering
  constraints.clear();
  constraints[Symbol('X', 15)] = 1;
  constraints[Symbol('X', 16)] = 1;
  constraints[Symbol('X', 17)] = 1;
  constraints[Symbol('X', 18)] = 1;
  constraints[Symbol('X', 19)] = 1;
  ordering = *fullGraph.orderingCOLAMDConstrained(fullTheta, constraints);
  // Linearize into a Gaussian Factor Graph
  linearGraph = *fullGraph.linearize(fullTheta, ordering);
  // Eliminate into a Bayes Net with iSAM2-type cliques
  jt = JunctionTree<GaussianFactorGraph, ISAM2Clique>(linearGraph);
  root = jt.eliminate(EliminateQR);
  bayesTree = BayesTree<GaussianConditional, ISAM2Clique>();
  bayesTree.insert(root);

  //  std::cout << "BAYES TREE:" << std::endl;
  //  SymbolicPrintTree(root, ordering.invert(), "  ");

  //  BAYES TREE:
  //    P( X14 X16 X19 )
  //      P( X17 | X16 )
  //        P( X18 | X17 )
  //      P( X15 | X16 )
  //      P( X13 | X14 X16 )
  //        P( X11 | X13 X14 )
  //          P( X10 | X11 X13 )
  //            P( X12 | X10 X13 )
  //              P( X9 | X12 X10 )
  //                P( X7 | X9 X12 )
  //                  P( X6 | X7 X9 )
  //                    P( X3 X5 | X6 )
  //                      P( X2 | X3 )
  //                      P( X1 | X3 )
  //                        P( X0 | X1 )
  //                  P( X4 | X7 )
  //                P( X8 | X9 )

  // Extract the cached factors for X4 and X6. These are the new smoother summarized factors
  // TODO: I'm concerned about the linearization point used to create these factors. It may need to be the updated lin points?
  smootherSummarization.resize(0);
  smootherSummarization.add(LinearizedJacobianFactor(boost::static_pointer_cast<JacobianFactor>(bayesTree.nodes().at(ordering.at(Symbol('X', 4)))->cachedFactor()), ordering, fullTheta));
  smootherSummarization.add(LinearizedJacobianFactor(boost::static_pointer_cast<JacobianFactor>(bayesTree.nodes().at(ordering.at(Symbol('X', 6)))->cachedFactor()), ordering, fullTheta));





  // Extract the nonlinear factors that should be sent to the smoother
  smootherKeys.clear();
  smootherKeys.push_back(Symbol('X',  8));
  smootherKeys.push_back(Symbol('X', 10));
  smootherKeys.push_back(Symbol('X', 11));
  smootherKeys.push_back(Symbol('X', 13));
  expectedSmootherFactors.resize(0);
  BOOST_FOREACH(const NonlinearFactor::shared_ptr& factor, fullGraph) {
    if(std::find_first_of(factor->begin(), factor->end(), smootherKeys.begin(), smootherKeys.end()) != factor->end()) {
      expectedSmootherFactors.push_back(factor);
    }
  }
  // And any factors that involve only the old root (X7, X9, X12)
  rootKeys.clear();
  rootKeys.push_back(Symbol('X',  7));
  rootKeys.push_back(Symbol('X',  9));
  rootKeys.push_back(Symbol('X', 12));
  BOOST_FOREACH(const NonlinearFactor::shared_ptr& factor, fullGraph) {
    size_t count = 0;
    BOOST_FOREACH(Key key, *factor) {
      if(std::binary_search(rootKeys.begin(), rootKeys.end(), key)) ++count;
    }
    if(count == factor->size()) expectedSmootherFactors.push_back(factor);
  }


  // Extract smoother Values
  expectedSmootherValues.clear();
  expectedSmootherValues.insert(Symbol('X',  7), updatedTheta.at(Symbol('X',  7)));
  expectedSmootherValues.insert(Symbol('X',  8), updatedTheta.at(Symbol('X',  8)));
  expectedSmootherValues.insert(Symbol('X',  9), updatedTheta.at(Symbol('X',  9)));
  expectedSmootherValues.insert(Symbol('X', 10), updatedTheta.at(Symbol('X', 10)));
  expectedSmootherValues.insert(Symbol('X', 11), updatedTheta.at(Symbol('X', 11)));
  expectedSmootherValues.insert(Symbol('X', 12), updatedTheta.at(Symbol('X', 12)));
  expectedSmootherValues.insert(Symbol('X', 13), updatedTheta.at(Symbol('X', 13)));

  // Extract the filter summarized factors
  // Cached factors that represent the filter side (i.e. the X15 and X17 clique)
  expectedFilterSumarization.resize(0);
  expectedFilterSumarization.add(LinearizedJacobianFactor(boost::static_pointer_cast<JacobianFactor>(bayesTree.nodes().at(ordering.at(Symbol('X', 15)))->cachedFactor()), ordering, fullTheta));
  expectedFilterSumarization.add(LinearizedJacobianFactor(boost::static_pointer_cast<JacobianFactor>(bayesTree.nodes().at(ordering.at(Symbol('X', 17)))->cachedFactor()), ordering, fullTheta));
  // And any factors that involve only the root (X14, X16, X17)
  rootKeys.clear();
  rootKeys.push_back(Symbol('X', 14));
  rootKeys.push_back(Symbol('X', 16));
  rootKeys.push_back(Symbol('X', 19));
  BOOST_FOREACH(const NonlinearFactor::shared_ptr& factor, fullGraph) {
    size_t count = 0;
    BOOST_FOREACH(Key key, *factor) {
      if(std::binary_search(rootKeys.begin(), rootKeys.end(), key)) ++count;
    }
    if(count == factor->size()) expectedFilterSumarization.push_back(factor);
  }

  // Extract the new root keys
  expectedRootValues.clear();
  expectedRootValues.insert(Symbol('X', 14), updatedTheta.at(Symbol('X', 14)));
  expectedRootValues.insert(Symbol('X', 16), updatedTheta.at(Symbol('X', 16)));
  expectedRootValues.insert(Symbol('X', 19), updatedTheta.at(Symbol('X', 19)));



  // Start the synchronization process
  actualSmootherFactors.resize(0); actualFilterSumarization.resize(0);
  actualSmootherValues.clear(); actualRootValues.clear();
  filter.presync();
  filter.synchronize(smootherSummarization);
  filter.getSmootherFactors(actualSmootherFactors, actualSmootherValues);
  filter.getSummarizedFactors(actualFilterSumarization, actualRootValues);
  filter.postsync();



  // Compare filter sync variables versus the expected
  CHECK(hessian_equal(expectedSmootherFactors, actualSmootherFactors, updatedTheta, 1e-8));
  CHECK(assert_equal(expectedSmootherValues, actualSmootherValues, 1e-4));
  CHECK(hessian_equal(expectedFilterSumarization, actualFilterSumarization, updatedTheta, 1e-8));
  CHECK(assert_equal(expectedRootValues, actualRootValues, 1e-4));
}

/* ************************************************************************* */
int main() { TestResult tr; return TestRegistry::runAllTests(tr);}
/* ************************************************************************* */