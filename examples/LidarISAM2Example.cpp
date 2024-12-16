/*
In order for beginners to quickly understand how to use the iSAM2 algorithm, which is one of the better back-end optimization algorithms, 
the factor graphs created in this code only contain odometry and loopback factors.
It only needs to receive an odometry topic of type nav_msgs::Odometry with the name “/odometry” and then it will run
*/

#include <mutex>
#include <thread>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/eigen.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/linear/NoiseModel.h>

using namespace gtsam;

//The definition of point cloud in LIO-SAM is used here, xyz,intensity,roll,pitch,yaw,time, contains most of the basic information.
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

// When reading a point cloud, the frame is stored in the transformin array, and this function converts it into a readable bitmap in GTSAM.
gtsam::Pose3 trans2gtsamPose(float transformIn[]){
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                            gtsam::Point3(transformIn[3],transformIn[4],transformIn[5]));
}

//Converts a PointTypePose type point cloud to a 3D matrix, which is used to calculate angles when keyframes are selected.
Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{ 
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

// Convert PointTypePose type point cloud to a bit pose that can be read in GTSAM.
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
}

//Judging keyframes when the straight line distance is greater than 1 metre
float surroundingkeyframeAddingDistThreshold = 1.0;
// Judgement of keyframes when the angle is greater than 0.2 degrees
float surroundingkeyframeAddingAngleThreshold = 0.2; 
// Array of temporary point clouds
float transformTobeMapped[6];
//Searching for loopbacks requires greater than 10 seconds on both frames
float historyKeyframeSearchRadius = 10.0;
//Record radar time stamps
ros::Time timeLaserInfoStamp;
//convert timestamp to second
double timeLaserInfoCur;
bool aLoopIsClosed = false;
// Construct an empty non-linear factor graph
NonlinearFactorGraph gtSAMgraph;  
// Declare the initial values of the factor graph and the optimisation results
Values initialEstimate; 

/*
The factor graph only models the history of SLAM poses and the relationship between inputs and observations; 
how to solve this factor graph, i.e., how to set the variables in such a way that the whole graph best meets all the constraints (with minimum error) 
requires the use of an optimizer. In addition to the most common Gaussian-Newton and Levenberg-Marquardt optimizers for solving nonlinear problems, 
GTSAM implements two incremental optimizers, iSAM,iSAM2*/
ISAM2 *isam; 
//Store the optimisation results for each factor
Values isamCurrentEstimate;
//Optimisation of the set parameters, which can be adjusted in the main function
ISAM2Params parameters; 

/*
Publish optimised path
*/
ros::Publisher path_pub; 
nav_msgs::Path path;
nav_msgs::Path globalPath;

// Odometer retention queue
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
/*
The thread lock mutex is to prevent data input overload when receiving point cloud data.
After turning on mBuf.lock() the main thread is locked until it finishes processing the just received frame topic then mBuf.unlock() to continue receiving the next frame.
*/
std::mutex mBuf;

//Historical keyframe position
pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<PointType>);
//History keyframe poses
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>);
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>); 

pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>);

std::map <int, int> loopIndexContainer; // from new to old，temporarily stored containers when detecting loopback frames
std::vector<std::pair<int, int>> loopIndexQueue; //Storing two loopback frames as pair
std::vector<gtsam::Pose3> loopPoseQueue; //Stores the relative position between two loopback frames


Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

//Converts the received odometer positions into an array for temporary storage.
void odomTotransform(Eigen::Quaterniond q_wodom_curr, Eigen::Vector3d t_wodom_curr)
{
    Eigen::Vector3d euler = q_wodom_curr.toRotationMatrix().eulerAngles(2,1,0);
    transformTobeMapped[0] = euler[0];
    transformTobeMapped[1] = euler[1];
    transformTobeMapped[2] = euler[2];    
    transformTobeMapped[3] = t_wodom_curr[0];    
    transformTobeMapped[4] = t_wodom_curr[1];
    transformTobeMapped[5] = t_wodom_curr[2];
}

//Determine the key frame, in order to prevent the first frame into the space pointer to report errors added a flag
bool flag = 0;
bool saveFrame(){
    //If it is the first frame then there is no data in cloudkeyposes6D yet, and the data in the temporary storage array is used directly.
    if(flag == 0)
    {
        PointTypePose ttt;
        ttt.x = transformTobeMapped[3];
        ttt.y = transformTobeMapped[4];
        ttt.z = transformTobeMapped[5];
        ttt.roll = transformTobeMapped[0];
        ttt.pitch = transformTobeMapped[1];
        ttt.yaw = transformTobeMapped[2];
        cloudKeyPoses6D->push_back(ttt);
        flag = 1;
    }
    
    //The following calculates the angle and displacement between two neighboring frames, which are treated as keyframes if they are greater than a certain threshold.
    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3],
        transformTobeMapped[4],transformTobeMapped[5],transformTobeMapped[0],transformTobeMapped[1],transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;// Transfer matrix
    float x,y,z,roll,pitch,yaw;
    pcl::getTranslationAndEulerAngles(transBetween,x,y,z,roll,pitch,yaw);// transform the transfer matrix to xyz and Euler angles

    if(abs(roll)  < surroundingkeyframeAddingAngleThreshold && 
        abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
        abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
    {
        return false;
    }
    return true;
}

/*
The function for detecting loopbacks is modeled here after the simplest loopback detection method used in LIO-SAM, where the last keyframe is used as the current frame
The last keyframe is used as the current frame, with two constraints: 
the closest spatial location (using kdtree for distance retrieval) and the time distance is far enough (removing frames that are too close in time).
*/
bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    auto it = loopIndexContainer.find(loopKeyCur);
    if(it != loopIndexContainer.end()){
        return false;
    }
    //construct a kd tree for the keyframes in 3D bitmap and use the current frame position to find the closest frames from the kd tree, and pick the one with the farthest time interval as the matching frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses -> setInputCloud(copy_cloudKeyPoses3D);
    // Find keyframes with similar spatial distances   
    kdtreeHistoryKeyPoses -> radiusSearch(copy_cloudKeyPoses3D -> back(),
                                        historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 5 );

    for(int i = 0; i < (int)pointSearchIndLoop.size(); i++)
    {
        int id = pointSearchIndLoop[i];
        if(abs(copy_cloudKeyPoses6D -> points[id].time - timeLaserInfoCur) > historyKeyframeSearchRadius)
        {
            loopKeyPre = id;
            break;
        }//Here it has to be greater than a threshold in time, otherwise all the frames found are adjacent to the current frame
    }
    
    std::cout << "loopKeyCur = " << loopKeyCur << std::endl;
    std::cout << "loopKeyPre = " << loopKeyPre << std::endl;
    
    if(loopKeyPre == -1 || loopKeyCur == loopKeyPre)
    {
        return false;
    }//Not found

    //Returns the two frames found to the two pointers of the inputs
    *latestID = loopKeyCur;
    *closestID = loopKeyPre;
    return true;
}

//This function is a separate thread that keeps running all the time looking for it, and when it finds it, it pushes it into the loopindexqueue
void performLoopClosure()
{
    if (cloudKeyPoses3D->points.empty() == true)
        return;

    /*The pointers to these two copies are the ones that will be used later in the detectLoopClosureDistance function.
    Because cloudKeyPoses3D and 6D are constantly changing and updating while in use, a copy of the current state poses to detect*/
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)    return;
    //Now loopKeyCur is the current frame and loopKeyPre is the detected loopback frame    
    
    /*The way to calculate the relative pose matrix is generally to give the pose of the two frames from the starting point, 
    and then use poseFrom.between(poseTo) to represent it in this way, because generally the current frame pose is calculated relative to the starting point.*/
    gtsam::Pose3 poseFrom = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyCur]);
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

    loopIndexQueue.push_back(std::make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));   

    loopIndexContainer[loopKeyCur] = loopKeyPre;
}

//Save the point cloud and odometer data received by the topic into two queues while converting the point cloud to PCL format (point cloud data is not used in this code)
void updateInitFialGuess(){
    if((!odometryBuf.empty()))
    {        

        std_msgs::Header frame_header = odometryBuf.front()->header;

        //Receive the latest frame of odometer information
        q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
        q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
        q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
        q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;            
        t_wodom_curr[0] = odometryBuf.front()->pose.pose.position.x;
        t_wodom_curr[1] = odometryBuf.front()->pose.pose.position.y;
        t_wodom_curr[2] = odometryBuf.front()->pose.pose.position.z;
        odometryBuf.pop();

        odomTotransform(q_wodom_curr, t_wodom_curr);//Converted to matrix form for temporary storage
        
        while(!odometryBuf.empty())
        {
            odometryBuf.pop();
            printf("drop lidar frame in mapping for real time performance \n");
        }//lost stacked data to keep the whole program real-time
    }
}

//Update the path from the first to the last frame after each optimization.Save to globalPath can be published directly, you can observe the optimization effect in real time in rviz
void updatePath(const PointTypePose& pose_in){
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id = "camera_init";
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}


// set initial guess
void RecordLastFrame()
{
    q_w_curr =  q_wodom_curr;
    t_w_curr =  t_wodom_curr;
}

void addOdomFactor(){
    Eigen::Vector3d r_w_last = q_w_curr.toRotationMatrix().eulerAngles(2,1,0);//Previous Frame Euler Angle
    gtsam::Rot3 rot_last = gtsam::Rot3::RzRyRx(r_w_last[0], r_w_last[1], r_w_last[2]);
    Eigen::Vector3d t_w_last = t_w_curr;
    
    RecordLastFrame();//This frame is recorded, and the relative position is calculated after the next frame is received.

    Eigen::Vector3d r_w_curr = q_w_curr.toRotationMatrix().eulerAngles(2,1,0);//Current Frame Euler Angle
    gtsam::Rot3 rot_curr = gtsam::Rot3::RzRyRx(r_w_curr[0], r_w_curr[1], r_w_curr[2]);
    
    //There are two ways to join gtsamgraph, one is the first factor PriorFactor and the other is BetweenFactor between two variables
    if(cloudKeyPoses3D->points.empty()){
        
        //Gaussian noise, representing our uncertainty about the factor
        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector6(6) << 1e-2, 1e-2, M_PI*M_PI, 1e4, 1e4, 1e4).finished());
        gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped),priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));// Adding Initial Values to the Set of Initial Estimates
    }//Add the first factor

    else{//Insert odometer factor: relative position between two frames
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom(rot_last, t_w_last);
            gtsam::Pose3 poseTo(rot_curr,t_w_curr);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);// Adding Initial Values to the Set of Initial Estimates
        }
}

void addLoopFactor()
{
    if(loopIndexQueue.empty())
    {
        return;
    }

    for(int i = 0; i < (int)loopIndexQueue.size(); i++)
    {
        int indexFrom = loopIndexQueue[i].first;
        int indexTo = loopIndexQueue[i].second;
        gtsam::Pose3 poseBetween = loopPoseQueue[i];//Relative position between two loopback frames
        auto odometryNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(1e-6), Vector3::Constant(0.03)).finished());
        gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, odometryNoise));
    }
    //Add all loopback factors at once, and clear them to prevent repetition.
    loopIndexQueue.clear();
    loopPoseQueue.clear();
    //The iSAM2 optimization is performed only after the loopback factor is added, otherwise only the odometer factor optimization in the factor graph has no effect
    aLoopIsClosed = true;    
}

//Main Processes
void saveKeyFrameAndFactor()
{
    //Saving keyframes
    if(saveFrame() == false)
    {
        return;
    }

    //Constructing a Factor Map
    addOdomFactor();
    addLoopFactor();
    
    std::cout << "****************************************************" << std::endl;
    gtSAMgraph.print("GTSAM Graph:\n");

    /*Adding the factor graph to the optimizer
    Because the factor graph constructed by this code to show the optimization steps more simply has only the odometry factor and the loopback factor,
    none of the optimizations have any effect until the loopback factor is added.
    The following two optimizations will have no effect, but will be useful when adding e.g. IMU pre-integration factors and GPS factors.*/
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    //5 optimizations when loopback factor is added
    if(aLoopIsClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();            
    }

    /*Clear the current factor map and initial values
    The factor graph has already been added to the optimizer, so it needs to be cleared in preparation for the next factor graph.*/
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    //Save the result of the latest frame after optimization
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;
    
    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
    thisPose6D.roll  = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw   = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);//Here push in to make sure the number of nodes in the queue is correct, and it will be updated again soon.
    updatePath(thisPose6D);

    globalPath.header.stamp = timeLaserInfoStamp;
    globalPath.header.frame_id = "camera_init";
    path_pub.publish(globalPath);

}

//After optimization, all previously saved positions should be updated so that the new odometer positions are used when the loopback is calculated again later on.
void correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {   
        int numPoses = isamCurrentEstimate.size();
        //You have to clear the path and republish each time
        globalPath.poses.clear();
        // update key poses
        std::cout << "isamCurrentEstimate.size(): " << numPoses << std::endl;
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

            updatePath(cloudKeyPoses6D->points[i]);
        }

        aLoopIsClosed = false;
    }
}

/*
A separate loopback detection thread, where the detection method is simpler, so even if it is put into the main thread, it will not affect the operation.
However, when using more advanced and complex loopback detection algorithms, it will take up more memory and may affect the running of the main thread.
When new loop relationships are detected, they are added to the factor graph by the main thread to optimize them.
*/
void loopClosureThread()
{
    ros::Rate rate(1.0);
    while(ros::ok){
        rate.sleep();
        performLoopClosure();
    }
}

//Odometer topic callback function
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();
    timeLaserInfoStamp = odometryBufBuf.front()->header.stamp;
    timeLaserInfoCur = odometryBufBuf.front()->header.stamp.toSec();   
}

/*Main thread This thread is mainly responsible for performing radar-to-map matching to get a more accurate radar odometry, 
and then adding the radar odometry and loopback detection factor to the factor map for optimization to get the global optimized keyframe bit position.*/
void running()
{
    while(1)
    {
        updateInitFialGuess();
        saveKeyFrameAndFactor();
        correctPoses();
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "keyFrame");
	ros::NodeHandle nh;

    //GTSAM optimized initial values
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

    std::thread loopthread(loopClosureThread);//circular thread
    std::thread run(running);//main thread

    path_pub = nh.advertise<nav_msgs::Path>("odom_path", 10, true);//Publishing optimized bit positions

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/odometry", 100, laserOdometryHandler);//Receive odometer topics

    ros::spin();
    return 0;
}
