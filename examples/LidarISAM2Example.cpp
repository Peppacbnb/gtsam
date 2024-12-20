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

struct PointTypePose {
    float x, y, z;          // Position
    float intensity;        // Intensity (used as index)
    float roll, pitch, yaw; // Orientation
    double time;            // Timestamp

    // Default constructor
    PointTypePose() 
        : x(0), y(0), z(0), intensity(0), roll(0), pitch(0), yaw(0), time(0) {}

    // Custom constructor for initialization
    PointTypePose(float _x, float _y, float _z, float _intensity,
                  float _roll, float _pitch, float _yaw, double _time)
        : x(_x), y(_y), z(_z), intensity(_intensity),
          roll(_roll), pitch(_pitch), yaw(_yaw), time(_time) {}

    // Constructor from Pose3
    PointTypePose(const Pose3& pose, float _intensity, double _time)
        : x(pose.translation().x()), y(pose.translation().y()), z(pose.translation().z()),
          intensity(_intensity),
          roll(pose.rotation().roll()), pitch(pose.rotation().pitch()), yaw(pose.rotation().yaw()),
          time(_time) {}
};

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

// Array of temporary point clouds
float gtransformTobeMapped[6];

//Record radar time stamps
ros::Time gtimeLaserInfoStamp;
//convert timestamp to second
double gtimeLaserInfoCur;
bool aLoopIsClosed = false;
// Construct an empty non-linear factor graph
NonlinearFactorGraph gtSAMgraph;  
// Declare the initial values of the factor graph and the optimisation results
Values ginitialEstimate; 

/*
The factor graph only models the history of SLAM poses and the relationship between inputs and observations; 
how to solve this factor graph, i.e., how to set the variables in such a way that the whole graph best meets all the constraints (with minimum error) 
requires the use of an optimizer. In addition to the most common Gaussian-Newton and Levenberg-Marquardt optimizers for solving nonlinear problems, 
GTSAM implements two incremental optimizers, iSAM,iSAM2*/
ISAM2 *gisam; 
//Store the optimisation results for each factor
Values gisamCurrentEstimate;
//Optimisation of the set parameters, which can be adjusted in the main function
ISAM2Params gparameters; 

/*
Publish optimised path
*/
ros::Publisher path_pub; 
nav_msgs::Path globalPath;

// Odometer retention queue
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
/*
The thread lock mutex is to prevent data input overload when receiving point cloud data.
After turning on mBuf.lock() the main thread is locked until it finishes processing the just received frame topic then mBuf.unlock() to continue receiving the next frame.
*/
std::mutex mBuf;

//Historical keyframe position
pcl::PointCloud<PointType>::Ptr gcloudKeyPoses3D(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr copy_gcloudKeyPoses3D(new pcl::PointCloud<PointType>);
//History keyframe poses
pcl::PointCloud<PointTypePose>::Ptr gcloudKeyPoses6D(new pcl::PointCloud<PointTypePose>);
pcl::PointCloud<PointTypePose>::Ptr copy_gcloudKeyPoses6D(new pcl::PointCloud<PointTypePose>); 

pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>);

std::map <int, int> loopIndexContainer; // from new to old，temporarily stored containers when detecting loopback frames
std::vector<std::pair<int, int>> loopIndexQueue; //Storing two loopback frames as pair
std::vector<gtsam::Pose3> loopPoseQueue; //Stores the relative position between two loopback frames


Eigen::Quaterniond g_q(1, 0, 0, 0);
Eigen::Vector3d g_t(0, 0, 0);
Eigen::Quaterniond g_q_odom(1, 0, 0, 0);
Eigen::Vector3d g_t_odom(0, 0, 0);

//Converts the received odometer positions into an array for temporary storage.
void odomTotransform(Eigen::Quaterniond g_q_odom, Eigen::Vector3d g_t_odom)
{
    Eigen::Vector3d euler = g_q_odom.toRotationMatrix().eulerAngles(2,1,0);
    gtransformTobeMapped[0] = euler[0];
    gtransformTobeMapped[1] = euler[1];
    gtransformTobeMapped[2] = euler[2];    
    gtransformTobeMapped[3] = g_t_odom[0];    
    gtransformTobeMapped[4] = g_t_odom[1];
    gtransformTobeMapped[5] = g_t_odom[2];
}

//Determine the key frame, in order to prevent the first frame into the space pointer to report errors added a flag

bool saveFrame()
{
    static bool flag = false;
    //If it is the first frame then there is no data in gcloudkeyposes6D yet, and the data in the temporary storage array is used directly.
    if(flag == false)
    {
        //Use the constructor to create and add directly to gcloudKeyPoses6D
        gcloudKeyPoses6D->push_back(PointTypePose(
            gtransformTobeMapped[3], gtransformTobeMapped[4], gtransformTobeMapped[5], 0,  // x, y, z, intensity
            gtransformTobeMapped[0], gtransformTobeMapped[1], gtransformTobeMapped[2], 0   // roll, pitch, yaw, time
        ));
        flag = true;
    }
    
    //The following calculates the angle and displacement between two neighboring frames, which are treated as keyframes if they are greater than a certain threshold.
    Eigen::Affine3f transStart = pclPointToAffine3f(gcloudKeyPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(gtransformTobeMapped[3],
        gtransformTobeMapped[4],gtransformTobeMapped[5],gtransformTobeMapped[0],gtransformTobeMapped[1],gtransformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;// Transfer matrix
    float x,y,z,roll,pitch,yaw;
    pcl::getTranslationAndEulerAngles(transBetween,x,y,z,roll,pitch,yaw);// transform the transfer matrix to xyz and Euler angles
    
    //Judging keyframes when the straight line distance is greater than 1 metre
    float surroundingkeyframeAddingDistThreshold = 1.0;
    // Judgement of keyframes when the angle is greater than 0.2 degrees
    float surroundingkeyframeAddingAngleThreshold = 0.2; 

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
    int loopKeyCur = copy_gcloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;
    
    float historyKeyframeSearchRadius = 10.0; //Searching for loopbacks requires greater than 10 seconds on both frames

    auto it = loopIndexContainer.find(loopKeyCur);
    if(it != loopIndexContainer.end()){
        return false;
    }
    //construct a kd tree for the keyframes in 3D bitmap and use the current frame position to find the closest frames from the kd tree, and pick the one with the farthest time interval as the matching frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses -> setInputCloud(copy_gcloudKeyPoses3D);
    // Find keyframes with similar spatial distances   
    kdtreeHistoryKeyPoses -> radiusSearch(copy_gcloudKeyPoses3D -> back(),
                                        historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 5 );

    for(int i = 0; i < (int)pointSearchIndLoop.size(); i++)
    {
        int id = pointSearchIndLoop[i];
        if(abs(copy_gcloudKeyPoses6D -> points[id].time - gtimeLaserInfoCur) > historyKeyframeSearchRadius)
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
    if (gcloudKeyPoses3D->points.empty() == true)
        return;

    /*The pointers to these two copies are the ones that will be used later in the detectLoopClosureDistance function.
    Because gcloudKeyPoses3D and 6D are constantly changing and updating while in use, a copy of the current state poses to detect*/
    *copy_gcloudKeyPoses3D = *gcloudKeyPoses3D;
    *copy_gcloudKeyPoses6D = *gcloudKeyPoses6D;

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)    return;
    //Now loopKeyCur is the current frame and loopKeyPre is the detected loopback frame    
    
    /*The way to calculate the relative pose matrix is generally to give the pose of the two frames from the starting point, 
    and then use poseFrom.between(poseTo) to represent it in this way, because generally the current frame pose is calculated relative to the starting point.*/
    gtsam::Pose3 poseFrom = pclPointTogtsamPose3(copy_gcloudKeyPoses6D->points[loopKeyCur]);
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_gcloudKeyPoses6D->points[loopKeyPre]);

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
        g_q_odom.x() = odometryBuf.front()->pose.pose.orientation.x;
        g_q_odom.y() = odometryBuf.front()->pose.pose.orientation.y;
        g_q_odom.z() = odometryBuf.front()->pose.pose.orientation.z;
        g_q_odom.w() = odometryBuf.front()->pose.pose.orientation.w;            
        g_t_odom[0] = odometryBuf.front()->pose.pose.position.x;
        g_t_odom[1] = odometryBuf.front()->pose.pose.position.y;
        g_t_odom[2] = odometryBuf.front()->pose.pose.position.z;
        odometryBuf.pop();

        odomTotransform(g_q_odom, g_t_odom);//Converted to matrix form for temporary storage
        
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
    g_q =  g_q_odom;
    g_t =  g_t_odom;
}

void addOdomFactor(){
    Eigen::Vector3d r_w_last = g_q.toRotationMatrix().eulerAngles(2,1,0);//Previous Frame Euler Angle
    gtsam::Rot3 rot_last = gtsam::Rot3::RzRyRx(r_w_last[0], r_w_last[1], r_w_last[2]);
    Eigen::Vector3d t_w_last = g_t;
    
    RecordLastFrame();//This frame is recorded, and the relative position is calculated after the next frame is received.

    Eigen::Vector3d r_w_curr = g_q.toRotationMatrix().eulerAngles(2,1,0);//Current Frame Euler Angle
    gtsam::Rot3 rot_curr = gtsam::Rot3::RzRyRx(r_w_curr[0], r_w_curr[1], r_w_curr[2]);
    
    //There are two ways to join gtsamgraph, one is the first factor PriorFactor and the other is BetweenFactor between two variables
    if(gcloudKeyPoses3D->points.empty()){
        
        //Gaussian noise, representing our uncertainty about the factor
        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector6(6) << 1e-2, 1e-2, M_PI*M_PI, 1e4, 1e4, 1e4).finished());
        gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(gtransformTobeMapped),priorNoise));
        ginitialEstimate.insert(0, trans2gtsamPose(gtransformTobeMapped));// Adding Initial Values to the Set of Initial Estimates
    }//Add the first factor

    else{//Insert odometer factor: relative position between two frames
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom(rot_last, t_w_last);
            gtsam::Pose3 poseTo(rot_curr,g_t);
            gtSAMgraph.add(BetweenFactor<Pose3>(gcloudKeyPoses3D->size()-1, gcloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            ginitialEstimate.insert(gcloudKeyPoses3D->size(), poseTo);// Adding Initial Values to the Set of Initial Estimates
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
    gisam->update(gtSAMgraph, ginitialEstimate);
    gisam->update();

    //5 optimizations when loopback factor is added
    if(aLoopIsClosed == true)
    {
        gisam->update();
        gisam->update();
        gisam->update();
        gisam->update();
        gisam->update();            
    }

    /*Clear the current factor map and initial values
    The factor graph has already been added to the optimizer, so it needs to be cleared in preparation for the next factor graph.*/
    gtSAMgraph.resize(0);
    ginitialEstimate.clear();

    //Save the result of the latest frame after optimization
    PointType thisPose3D;
    Pose3 latestEstimate;

    // Get the latest estimate from iSAM
    gisamCurrentEstimate = gisam->calculateEstimate();
    latestEstimate = gisamCurrentEstimate.at<Pose3>(gisamCurrentEstimate.size() - 1);

    // Populate thisPose3D with translation data
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = gcloudKeyPoses3D->size(); // Use intensity as index
    gcloudKeyPoses3D->push_back(thisPose3D);

    // Use the new constructor to create thisPose6D directly
    PointTypePose thisPose6D(latestEstimate, thisPose3D.intensity, gtimeLaserInfoCur);
    gcloudKeyPoses6D->push_back(thisPose6D);

    // Update the global path
    updatePath(thisPose6D);

    // Publish the global path
    globalPath.header.stamp = gtimeLaserInfoStamp;
    globalPath.header.frame_id = "camera_init";
    path_pub.publish(globalPath);


}

//After optimization, all previously saved positions should be updated so that the new odometer positions are used when the loopback is calculated again later on.
void correctPoses()
{
    if (gcloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {   
        int numPoses = gisamCurrentEstimate.size();
        //You have to clear the path and republish each time
        globalPath.poses.clear();
        // update key poses
        std::cout << "gisamCurrentEstimate.size(): " << numPoses << std::endl;
        for (int i = 0; i < numPoses; ++i)
        {
            gcloudKeyPoses3D->points[i].x = gisamCurrentEstimate.at<Pose3>(i).translation().x();
            gcloudKeyPoses3D->points[i].y = gisamCurrentEstimate.at<Pose3>(i).translation().y();
            gcloudKeyPoses3D->points[i].z = gisamCurrentEstimate.at<Pose3>(i).translation().z();

            gcloudKeyPoses6D->points[i].x = gcloudKeyPoses3D->points[i].x;
            gcloudKeyPoses6D->points[i].y = gcloudKeyPoses3D->points[i].y;
            gcloudKeyPoses6D->points[i].z = gcloudKeyPoses3D->points[i].z;
            gcloudKeyPoses6D->points[i].roll  = gisamCurrentEstimate.at<Pose3>(i).rotation().roll();
            gcloudKeyPoses6D->points[i].pitch = gisamCurrentEstimate.at<Pose3>(i).rotation().pitch();
            gcloudKeyPoses6D->points[i].yaw   = gisamCurrentEstimate.at<Pose3>(i).rotation().yaw();

            updatePath(gcloudKeyPoses6D->points[i]);
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
    gtimeLaserInfoStamp = odometryBufBuf.front()->header.stamp;
    gtimeLaserInfoCur = odometryBufBuf.front()->header.stamp.toSec();   
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
    gparameters.relinearizeThreshold = 0.1;
    gparameters.relinearizeSkip = 1;
    gisam = new ISAM2(gparameters);

    std::thread loopthread(loopClosureThread);//circular thread
    std::thread run(running);//main thread

    path_pub = nh.advertise<nav_msgs::Path>("odom_path", 10, true);//Publishing optimized bit positions

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/odometry", 100, laserOdometryHandler);//Receive odometer topics

    ros::spin();
    return 0;
}
