/****************************************************************************
 *
 *   Copyright (c) 2018 Mohammed Kabir. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name Mohammed Kabir nor the names of their contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

#ifndef DNN_DETECTOR_H_
#define DNN_DETECTOR_H_

#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>

#include <Eigen/Dense>

#include "ros/ros.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Range.h>
#include <sensor_msgs/image_encodings.h>

#include <perception_msgs/WorldEvidence.h>
#include <perception_msgs/ObjectEvidence.h>

#include <problib/conversions.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include "dnn_detector/utils.h"

namespace dnn_detector
{

class DNNDetector
{
private:

	ros::NodeHandle nh_;
	ros::NodeHandle nh_private_;

	ros::Publisher world_evidence_pub_; //!<  ROS publisher that publishes detections from the network
	image_transport::Publisher image_pub_; //!< ROS image publisher that publishes the visualisation image

	message_filters::Subscriber<sensor_msgs::Image> image_sub_; //!< The ROS subscriber for raw camera images
	message_filters::Subscriber<sensor_msgs::Image> depth_sub_; //!< The ROS subscriber for depth images
	message_filters::Subscriber<sensor_msgs::Range> range_sub_; //!< The ROS subscriber for rangefinder data

	ros::Subscriber camera_info_sub_; //!< The ROS subscriber for camera calibration

	// Topic synchronisation
	typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ImageDepthPolicy;
  	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Range> ImageRangePolicy;
  	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Range> ImageDepthRangePolicy;
  	typedef message_filters::Synchronizer<ImageDepthPolicy> ImageDepthSync;
  	typedef message_filters::Synchronizer<ImageRangePolicy> ImageRangeSync;
  	typedef message_filters::Synchronizer<ImageDepthRangePolicy> ImageDepthRangeSync;
  	boost::shared_ptr<ImageDepthSync> image_depth_sync_;
  	boost::shared_ptr<ImageRangeSync> image_range_sync_;
  	boost::shared_ptr<ImageDepthRangeSync> image_depth_range_sync_;

  	// Camera parameters
	cv::Size 				camera_image_size_;
	cv::Mat 				camera_matrix_K_; //!< Variable to store the camera matrix as an OpenCV matrix
	std::vector<double> 	camera_distortion_coeffs_; //!< Variable to store the camera distortion parameters
	bool 					camera_equidistant_distortion_;
	bool 					have_camera_info_; //!< Variable that indicates whether the camera calibration parameters have been obtained from the camera

	// Depth sources
	bool use_rangefinder_;
	bool use_depth_;
	float fixed_depth_;

	// Network config
	cv::dnn::Net network_;
	std::string output_layer_type_;
	std::vector<cv::String> output_names_;
	cv::Size input_size_;
	cv::Scalar mean_;
	float scale_;
	bool swap_rb_;

	// Detection config
	float confidence_threshold_;
	float nms_threshold_;
	std::vector<std::string> classes_;

  	struct NetworkPrediction 
	{
		ros::Time timestamp;
		std::string frame_id;
	    std::vector<int> class_ids;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;
		std::vector<cv::Point2f> centroids;
		std::vector<float> depth;
	};

	void processImage(const cv::Mat &image,
								NetworkPrediction &prediction);

	void processOutput(const std::vector<cv::Mat> &outputs, 
								NetworkPrediction &prediction);

	void generateWorldEvidence(const NetworkPrediction &prediction);

	void addObjectEvidence(const Eigen::Vector3f &position,
							 const Eigen::Quaternionf &orientation,
							 const std::string &class_label, 
							 float class_confidence,
							 perception_msgs::WorldEvidence &world_evidence);

	void generateVisualization(cv::Mat &frame, NetworkPrediction &prediction);

public:

	DNNDetector(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
	DNNDetector() : DNNDetector(ros::NodeHandle(), ros::NodeHandle("~")) {}
	~DNNDetector();

	void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg);

	void imageCallback(const sensor_msgs::Image::ConstPtr &image_msg);
	void imageDepthCallback(const sensor_msgs::Image::ConstPtr &image_msg, const sensor_msgs::Image::ConstPtr &depth_msg);
	void imageRangeCallback(const sensor_msgs::Image::ConstPtr &image_msg, const sensor_msgs::Range::ConstPtr &range_msg);
	void imageDepthRangeCallback(const sensor_msgs::Image::ConstPtr &image_msg, const sensor_msgs::Image::ConstPtr &depth_msg, const sensor_msgs::Range::ConstPtr &range_msg);

};

} // namespace dnn_detector

#endif /* DNN_DETECTOR_H_ */
