/*
    @file scanlinerun.cpp
    @brief ROS Node for scan line run

    This is a ROS node to perform scan line run clustring.
    Implementation accoriding to <Fast Segmentation of 3D Point Clouds: A Paradigm>

    In this case, it's assumed that the x,y axis points at sea-level,
    and z-axis points up. The sort of height is based on the Z-axis value.

    @author Vincent Cheung(VincentCheungm)
    @bug .
*/
#include <forward_list>
#include <iostream>
// For disable PCL complile lib, to use PointXYZIR, and customized pointcloud
#define PCL_NO_PRECOMPILE

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <velodyne_pointcloud/point_types.h>
#include <visualization_msgs/MarkerArray.h>
#include <boost/algorithm/clamp.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

// Customed Point Struct for holding clustered points
namespace scan_line_run {
/** Euclidean Velodyne coordinate, including intensity and ring number, and label. */
struct PointXYZIRL
{
  PCL_ADD_POINT4D;                // quad-word XYZ
  float intensity;                ///< laser intensity reading
  uint16_t ring;                  ///< laser ring number
  uint16_t label;                 ///< point label
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
} EIGEN_ALIGN16;

} // namespace scan_line_run

#define SLRPointXYZIRL scan_line_run::PointXYZIRL
#define VPoint velodyne_pointcloud::PointXYZIR
#define RUN pcl::PointCloud<SLRPointXYZIRL>

// clang-format off
// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(scan_line_run::PointXYZIRL,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  (uint16_t, label, label))
// clang-format on

#define dist(a, b) sqrt(((a).x - (b).x) * ((a).x - (b).x) + ((a).y - (b).y) * ((a).y - (b).y))

const int ANGLE_BUCKETS = 2251;
const std::string WINDOW_NAME = "VELO DEPTH";
cv::Mat heat_map(5 * 32, ANGLE_BUCKETS / 2, CV_8UC3, cv::Scalar(0, 0, 0));

/*
    @brief Scan Line Run ROS Node.
    @param Velodyne Pointcloud Non Ground topic.
    @param Sensor Model.
    @param Threshold between points belong to the same run
    @param Threshold between runs

    @subscirbe:/all_points
    @publish:/slr
*/
class ScanLineRun
{
 public:
  ScanLineRun();

 private:
  ros::NodeHandle node_handle_;
  ros::Subscriber points_node_sub_;
  ros::Publisher cluster_points_pub_;
  ros::Publisher ring_points_pub_;
  ros::Publisher cluster_viz_pub_;

  std::string point_topic_;
  std::string point_frame_;

  int sensor_model_; // also means number of sensor scan line.
  double th_run_;    // thresold of distance of points belong to the same run.
  double th_merge_;  // threshold of distance of runs to be merged.

  // For organization of points.
  std::vector<std::vector<SLRPointXYZIRL>> laser_frame_;
  std::vector<SLRPointXYZIRL> laser_row_;

  std::vector<std::forward_list<SLRPointXYZIRL*>> runs_; // For holding all runs.
  uint16_t max_label_;                   // max run labels, for disinguish different runs.
  std::vector<std::vector<int>> ng_idx_; // non ground point index ('row' in that
                                         // sensor_frame[scan_line)

  // Call back funtion.
  void velodyne_callback_(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);
  // For finding runs on a scanline.
  void find_runs_(int scanline);
  // For update points cluster label after merge action.
  void update_labels_(int scanline);
  // For merge `current` run to `target` run.
  void merge_runs_(uint16_t cur_label, uint16_t target_label);

  /// @deprecated methods for smart index
  // Smart idx according to paper, but not useful in my case.
  int smart_idx_(int local_idx, int n_i, int n_j, bool inverse);

  // Dummy object to occupy idx 0.
  std::forward_list<SLRPointXYZIRL*> dummy_;
};

/*
    @brief Constructor of SLR Node.
    @return void
*/
ScanLineRun::ScanLineRun() : node_handle_("~")
{
  // Init ROS related
  ROS_INFO("Inititalizing Scan Line Run Cluster...");
  node_handle_.param<std::string>("point_topic", point_topic_, "/all_points");
  ROS_INFO("point_topic: %s", point_topic_.c_str());

  node_handle_.param<std::string>("point_frame", point_frame_, "/velodyne");
  ROS_INFO("point_frame: %s", point_frame_.c_str());

  node_handle_.param("sensor_model", sensor_model_, 32);
  ROS_INFO("Sensor Model: %d", sensor_model_);

  // Init Ptrs with vectors
  for (int i = 0; i < sensor_model_; i++) {
    std::vector<int> dummy_vec;
    ng_idx_.push_back(dummy_vec);
  }

  // Init LiDAR frames with vectors and points
  SLRPointXYZIRL p_dummy;
  p_dummy.intensity = -1; // Means unoccupy by any points
  laser_row_ = std::vector<SLRPointXYZIRL>(ANGLE_BUCKETS, p_dummy);
  laser_frame_ = std::vector<std::vector<SLRPointXYZIRL>>(32, laser_row_);

  // Init runs, idx 0 for interest point, and idx for ground points
  max_label_ = 1;
  runs_.push_back(dummy_);
  runs_.push_back(dummy_);

  node_handle_.param("th_run", th_run_, 0.15);
  ROS_INFO("Point-to-Run Threshold: %f", th_run_);

  node_handle_.param("th_merge", th_merge_, 0.5);
  ROS_INFO("RUN-to-RUN Distance Threshold: %f", th_merge_);

  // Subscriber to velodyne topic
  points_node_sub_ =
      node_handle_.subscribe(point_topic_, 2, &ScanLineRun::velodyne_callback_, this);

  // Publisher Init
  std::string cluster_topic;
  node_handle_.param<std::string>("cluster", cluster_topic, "/slr");
  ROS_INFO("Cluster Output Point Cloud: %s", cluster_topic.c_str());
  cluster_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>(cluster_topic, 10);

  cluster_viz_pub_ = node_handle_.advertise<visualization_msgs::MarkerArray>("cluster_centers", 10);
}

/*
    @brief Read points from the given scan_line.
    The distance of two continuous points will be labelled to the same run.
    Clusterred points(`Runs`) stored in `runs_[cluster_id]`.

    @param scan_line: The scan line to find runs.
    @return void
*/
void ScanLineRun::find_runs_(int scan_line)
{
  // If there is no non-ground points of current scanline, skip.
  int point_size = ng_idx_[scan_line].size();
  if (point_size <= 0) {
    ROS_INFO("No non-ground points of scanline %d", scan_line);
    return;
  }

  size_t runs_before = runs_.size();

  int non_g_pt_row = ng_idx_[scan_line][0];                // The first non ground point
  int non_g_pt_row_l = ng_idx_[scan_line][point_size - 1]; // The last non ground point

  /* Iterate all non-ground points, and compute and compare the distance
  of each two continous points. At least two non-ground points are needed.
  */
  for (int i_idx = 0; i_idx < point_size - 1; i_idx++) {
    int row_cur = ng_idx_[scan_line][i_idx];
    int row_nxt = ng_idx_[scan_line][i_idx + 1];

    if (i_idx == 0) {
      // The first point, make a new run.
      auto& p_0 = laser_frame_[scan_line][row_cur];
      max_label_ += 1;
      runs_.push_back(dummy_);
      laser_frame_[scan_line][row_cur].label = max_label_;
      runs_[p_0.label].insert_after(runs_[p_0.label].cbefore_begin(),
                                    &laser_frame_[scan_line][row_cur]);

      if (p_0.label == 0) {
        ROS_ERROR("p_0.label == 0");
      }
    }

    // We can have a bunch of duplicate: row_cur == row_nxt...

    // Compare with the next (non-ground) point
    auto& p_i =
        laser_frame_[scan_line][row_cur]; // what happens if this is a point we have seen before??
    auto& p_i1 = laser_frame_[scan_line][row_nxt];

    // If next point is ground point, skip.
    if (p_i1.label == 1u) {
      // Add to ground run `runs_[1]`
      runs_[p_i1.label].insert_after(runs_[p_i1.label].cbefore_begin(),
                                     &laser_frame_[scan_line][row_nxt]);
      continue;
    }

    /* If curr point is not ground and next point is within threshold, then add it to the same run.
       Else, make a new run.
    */
    if (p_i.label != 1u && dist(p_i, p_i1) < th_run_) {
      p_i1.label = p_i.label;
    } else {
      max_label_ += 1;
      p_i1.label = max_label_;
      runs_.push_back(dummy_);
    }

    // Insert the index.
    runs_[p_i1.label].insert_after(runs_[p_i1.label].cbefore_begin(),
                                   &laser_frame_[scan_line][row_nxt]);

    if (p_i1.label == 0) {
      ROS_ERROR("p_i1.label == 0");
    }
  }

  // Compare the last point and the first point, for laser scans is a ring.
  if (point_size > 1) {
    auto& p_0 = laser_frame_[scan_line][non_g_pt_row];
    auto& p_l = laser_frame_[scan_line][non_g_pt_row_l];

    // Skip, if one of the start point or the last point is ground point.
    if (p_0.label == 1u || p_l.label == 1u) {
      return;
    } else if (dist(p_0, p_l) < th_run_) {
      if (p_0.label == 0) {
        ROS_ERROR("Ring Merge to 0 label");
      }
      /// If next point is within threshold, then merge it into the same run.
      merge_runs_(p_l.label, p_0.label);
    }
  } else if (point_size == 1) {
    // The only point, make a new run.
    auto& p_0 = laser_frame_[scan_line][non_g_pt_row];
    max_label_ += 1;
    runs_.push_back(dummy_);
    laser_frame_[scan_line][non_g_pt_row].label = max_label_;
    runs_[p_0.label].insert_after(runs_[p_0.label].cbefore_begin(),
                                  &laser_frame_[scan_line][non_g_pt_row]);
  }

  ROS_INFO("For scan_line %d found: %lu runs", scan_line, runs_.size() - runs_before);
}


/*
    @brief Update label between points and their smart `neighbour` point
    above `scan_line`.

    @param scan_line: The current scan line number.
*/
void ScanLineRun::update_labels_(int scan_line)
{
  // Iterate each point of this scan line to update the labels.
  int point_size_j_idx = ng_idx_[scan_line].size();
  // Current scan line is emtpy, do nothing.
  if (point_size_j_idx == 0)
    return;

  // Iterate each point of this scan line to update the labels.
  for (int j_idx = 0; j_idx < point_size_j_idx; j_idx++) {
    int j = ng_idx_[scan_line][j_idx];

    auto& p_j = laser_frame_[scan_line][j];

    // Runs above from scan line 0 to scan_line
    for (int l = scan_line - 1; l >= 0; l--) {
      if (ng_idx_[l].size() == 0)
        continue;

      // Smart index for the near enough point, after re-organized these points.
      int nn_idx = j;

      if (laser_frame_[l][nn_idx].intensity == -1 || laser_frame_[l][nn_idx].label == 1u) {
        continue;
      }

      // Nearest neighbour point
      auto& p_nn = laser_frame_[l][nn_idx];
      // Skip, if these two points already belong to the same run.
      if (p_j.label == p_nn.label) {
        continue;
      }
      double dist_min = dist(p_j, p_nn);

      /* Otherwise,
      If the distance of the `nearest point` is within `th_merge_`,
      then merge to the smaller run.
      */
      if (dist_min < th_merge_) {
        uint16_t cur_label = 0, target_label = 0;

        if (p_j.label == 0 || p_nn.label == 0) {
          ROS_ERROR("p_j.label:%u, p_nn.label:%u", p_j.label, p_nn.label);
        }
        // Merge to a smaller label cluster
        if (p_j.label > p_nn.label) {
          cur_label = p_j.label;
          target_label = p_nn.label;
        } else {
          cur_label = p_nn.label;
          target_label = p_j.label;
        }

        // Merge these two runs.
        merge_runs_(cur_label, target_label);
      }
    }
  }
}

/*
    @brief Merge current run to the target run.

    @param cur_label: The run label of current run.
    @param target_label: The run label of target run.
*/
void ScanLineRun::merge_runs_(uint16_t cur_label, uint16_t target_label)
{
  if (cur_label == 0 || target_label == 0) {
    ROS_ERROR("Error merging runs cur_label:%u target_label:%u", cur_label, target_label);
  }
  // First, modify the label of current run.
  for (auto& p : runs_[cur_label]) {
    p->label = target_label;
  }
  // Then, insert points of current run into target run.
  runs_[target_label].insert_after(runs_[target_label].cbefore_begin(), runs_[cur_label].begin(),
                                   runs_[cur_label].end());
  runs_[cur_label].clear();
}

/*
    @brief Smart index for nearest neighbour on scanline `i` and scanline `j`.

    @param local_idx: The local index of point on current scanline.
    @param n_i: The number of points on scanline `i`.
    @param n_j: The number of points on scanline `j`.
    @param inverse: If true, means `local_idx` is on the outsider ring `j`.
    Otherwise, it's on the insider ring `i`.

    @return The smart index.
*/
[[deprecated("Not useful in my case.")]] int ScanLineRun::smart_idx_(int local_idx, int n_i,
                                                                     int n_j, bool inverse = false)
{
  if (inverse == false) {
    // In case of zero-divide.
    if (n_i == 0)
      return 0;
    float rate = (n_j * 1.0f) / n_i;
    int idx = floor(rate * local_idx);

    // In case of overflow
    if (idx > n_j) {
      idx = n_j > 1 ? n_j - 1 : 0;
    }
    return idx;
  } else {
    // In case of zero-divide.
    if (n_j == 0)
      return 0;
    float rate = (n_i * 1.0f) / n_j;
    int idx = ceil(rate * local_idx);

    // In case of overflow
    if (idx > n_i) {
      idx = n_i > 1 ? n_i - 1 : 0;
    }
    return idx;
  }
}

visualization_msgs::Marker clusterCenterMarker(const std::string& frame_id, size_t id,
                                               const Eigen::Vector3f& center)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time::now();
  marker.id = id;
  marker.ns = "cluster_centers";
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.pose.position.x = center(0);
  marker.pose.position.y = center(1);
  marker.pose.position.z = center(2);
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;
  marker.color.a = 1.0;
  marker.color.g = 1.0;
  return marker;
}

/*
    @brief Velodyne pointcloud callback function, which subscribe `/all_points`
    and publish cluster points `slr`.
*/
void ScanLineRun::velodyne_callback_(const sensor_msgs::PointCloud2ConstPtr& in_cloud_msg)
{
  // Msg to pointcloud
  pcl::PointCloud<SLRPointXYZIRL> laserCloudIn;
  pcl::fromROSMsg(*in_cloud_msg, laserCloudIn);

  /// Clear and init.
  // Clear runs in the previous scan.
  max_label_ = 1;
  if (!runs_.empty()) {
    runs_.clear();
    runs_.push_back(dummy_); // dummy for index `0`
    runs_.push_back(dummy_); // for ground points
  }

  // Init laser frame.
  SLRPointXYZIRL p_dummy;
  p_dummy.intensity = -1;
  laser_row_ = std::vector<SLRPointXYZIRL>(ANGLE_BUCKETS, p_dummy);
  laser_frame_ = std::vector<std::vector<SLRPointXYZIRL>>(32, laser_row_);

  // Init non-ground index holder.
  for (int i = 0; i < sensor_model_; i++) {
    ng_idx_[i].clear();
  }

  // Organize Pointcloud in scanline
  double range = 0;
  int row = 0;

  int min_row = ANGLE_BUCKETS;
  int max_row = 0;
  int num_collisions = 0;
  int num_non_ground_collisions = 0;
  int num_exact_duplicate = 0;
  std::set<int> unique_rows;
  std::vector<std::set<int>> unique_rows_per_ring(32);
  std::vector<int> collisions_per_ring(32, 0);
  float min_duplicate_dist = 1000.0;
  float max_duplicate_dist = 0.0;
  int n_prints = 0;

  ROS_INFO("PC size = %lu \"image\" size %d", laserCloudIn.points.size(), 32 * ANGLE_BUCKETS);

  // Fill in 2d grid of points (laser_frame_)
  // Fill in non-ground point indices (ng_idx_)
  for (auto& point : laserCloudIn.points) {
    if (point.ring < sensor_model_ && point.ring >= 0) {
      // Digitize into ANGLE_BUCKETS different angles
      float angle = std::atan2(point.y, point.x);
      row = int(float(ANGLE_BUCKETS) * (angle + M_PI) / (2 * M_PI));

      min_row = std::min(min_row, row);
      max_row = std::max(max_row, row);
      unique_rows.insert(row);

      unique_rows_per_ring[point.ring].insert(row);

      bool duplicate = false;

      if (row >= ANGLE_BUCKETS || row < 0) {
        ROS_ERROR("Row: %d is out of index.", row);
        return;
      } else {
        // How the hell can we have so many collisions ~ 50% -- best guess is dual return mode.
        if (laser_frame_[point.ring][row].intensity != -1) {
          collisions_per_ring[point.ring]++;
          num_collisions++;
          duplicate = true;

          assert(laser_frame_[point.ring][row].ring == point.ring);

          float duplicate_dist = dist(point, laser_frame_[point.ring][row]);

          if (duplicate_dist < 0.00001f) {
            num_exact_duplicate++;
          }

          // Often exactly the same values...
          if (++n_prints < 10 || duplicate_dist > 1.0) {
            ROS_WARN("ring %d, row: %d our angle %.5f prev angle %.5f dist = %.3f \n"
                     "our  pnt: (%.5f, %.5f, %.5f)\n"
                     "prev pnt: (%.5f, %.5f, %.5f)",
                     point.ring, row, angle,
                     std::atan2(laser_frame_[point.ring][row].y, laser_frame_[point.ring][row].x),
                     duplicate_dist, point.x, point.y, point.z, laser_frame_[point.ring][row].x,
                     laser_frame_[point.ring][row].y, laser_frame_[point.ring][row].z);
          }

          min_duplicate_dist = std::min(min_duplicate_dist, duplicate_dist);
          max_duplicate_dist = std::max(max_duplicate_dist, duplicate_dist);

          // Don't over-write...
          if (point.label != 1u) {
            // laser_frame_[point.ring][row] = point;
            num_non_ground_collisions++;
          }

        } else {
          laser_frame_[point.ring][row] = point;
        }
      }

      if (!duplicate) {
        if (point.label != 1u) {
          // Not ground
          ng_idx_[point.ring].push_back(row);
        } else {
          // Ground, add to runs idx=1
          runs_[1].insert_after(runs_[1].cbefore_begin(), &point);
        }
      }
    }
  } // end for points

  // Print an image of this
  // cv::Mat image(5 * 32, ANGLE_BUCKETS / 2, CV_32FC1, cv::Scalar(-50));

  // roll-our-own heatmap...
  heat_map = cv::Scalar(0, 0, 0);
  for (int ring = 0; ring < 32; ++ring) {
    for (int row = 0; row < ANGLE_BUCKETS; row += 2) {
      if (laser_frame_[ring][row].intensity != -1 && laser_frame_[ring][row].label != 1u) {
        //        float height = boost::algorithm::clamp(laser_frame_[ring][row].z, -5, 5);
        //        float t = (height + 5.) / 10.;
        float range = std::sqrt(laser_frame_[ring][row].x * laser_frame_[ring][row].x +
                                laser_frame_[ring][row].y * laser_frame_[ring][row].y);
                                //laser_frame_[ring][row].z * laser_frame_[ring][row].z);
        range = boost::algorithm::clamp(range, 0.0f, 50.0f);
        float t = range/50.0f;
        cv::Vec3b val = t * cv::Vec3b(0, 0, 255) + (1 - t) * cv::Vec3b(255, 0, 0);

        for (int j = 0; j < 5; ++j) {
          // image.at<float>(5 * ring + j, row) = height;
          heat_map.at<cv::Vec3b>(5 * ring + j, row/2) = val;
        }
      }
    }
  }
  cv::flip(heat_map, heat_map, 0); // flip vertically

  ROS_INFO("Min row: %d, max row: %d, unique rows: %d, collisions: %d (non ground: %d), exact "
           "duplicate: %d\n"
           "min_dup. dist = %.8f max_dup. dist = %.8f",
           min_row, max_row, static_cast<int>(unique_rows.size()), num_collisions,
           num_non_ground_collisions, num_exact_duplicate, min_duplicate_dist, max_duplicate_dist);

  // Main processing
  for (int i = 0; i < sensor_model_; i++) {
    // get runs on current scan line i
    find_runs_(i);
    update_labels_(i);
  }

  // Extract Clusters
  // re-organize scan-line points into cluster point cloud
  pcl::PointCloud<SLRPointXYZIRL>::Ptr laserCloud(new pcl::PointCloud<SLRPointXYZIRL>());
  pcl::PointCloud<SLRPointXYZIRL>::Ptr clusters(new pcl::PointCloud<SLRPointXYZIRL>());

  int cnt = 0;

  std::vector<Eigen::Vector3f> cluster_centers;

  // Re-organize pointcloud clusters for PCD saving or publish
  for (size_t i = 2; i < runs_.size(); i++) {
    if (!runs_[i].empty()) {
      cnt++;

      int ccnt = 0;

      Eigen::Vector3f cluster_mean(0., 0., 0.);

      // adding run current for publishing
      for (auto& p : runs_[i]) {
        // Reorder the label id
        ccnt++;
        p->label = cnt;
        laserCloud->points.push_back(*p);
        // clusters->points.push_back(*p);

        cluster_mean(0) += p->x;
        cluster_mean(1) += p->y;
        cluster_mean(2) += p->z;
      }
      cluster_mean *= 1.0 / ccnt;
      if (ccnt > 5) {
        cluster_centers.push_back(cluster_mean);
      }
      // clusters->clear();
    }
  }
  ROS_INFO("Total cluster: %d", cnt);
  // Publish Cluster Points
  if (laserCloud->points.size() > 0) {
    sensor_msgs::PointCloud2 cluster_msg;
    pcl::toROSMsg(*laserCloud, cluster_msg);
    // cluster_msg.header.frame_id = point_frame_;
    cluster_msg.header = in_cloud_msg->header;

    cluster_points_pub_.publish(cluster_msg);
  }

  visualization_msgs::MarkerArray markers;
  // clear previous data
  visualization_msgs::Marker marker;
  marker.action = 3; // DELETEALL
  markers.markers.push_back(marker);

  // Publish Cluster Centers
  for (size_t i = 0; i < cluster_centers.size(); ++i) {
    const Eigen::Vector3f& center = cluster_centers[i];
    markers.markers.push_back(clusterCenterMarker(point_frame_, i, center));
  }
  cluster_viz_pub_.publish(markers);
}

int main(int argc, char** argv)
{
  cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(WINDOW_NAME, 40, 900); // Move to a place we can see...

  ros::init(argc, argv, "ScanLineRun");
  ScanLineRun node;

  const ros::WallDuration timeout(0.1f);
  while (ros::ok()) {
    ros::getGlobalCallbackQueue()->callAvailable(timeout);

    // So this doesn't time out when paused
    cv::imshow(WINDOW_NAME, heat_map);
    cv::waitKey(2);
  }
  // ros::spin();

  return 0;
}
