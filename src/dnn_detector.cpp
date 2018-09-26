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

#include "dnn_detector/dnn_detector.h"

namespace dnn_detector
{

DNNDetector::DNNDetector(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private) :
	nh_(nh),
	nh_private_(nh_private),
	have_camera_info_(false),
	confidence_threshold_(0.2),
	nms_threshold_(0.40),
	camera_equidistant_distortion_(false), 
	mean_(cv::Scalar(0,0,0)) // TODO : init all class members
{

	// Path to network configuration and weights
	std::string config_path;
	std::string model_path;
	nh_private_.param<std::string>("network/config_path", config_path, "yolo.cfg");
	nh_private_.param<std::string>("network/model_path", model_path, "yolo.weights");

	// Network parameters
	int input_width, input_height;
	nh_private_.param<int>("network/input_width", input_width, 416);
	nh_private_.param<int>("network/input_height", input_height, 416);
	nh_private_.param<float>("network/scale", scale_, 0.00392);
	nh_private_.param<bool>("network/swap_rb", swap_rb_, true);

	input_size_ = cv::Size(input_width, input_height);

	// Detection parameters
	std::string classes_path;
	nh_private_.param<std::string>("network/classes_path", classes_path, "classes.txt");
	nh_private_.param<float>("detector/confidence_threshold", confidence_threshold_, 0.2);
	nh_private_.param<float>("detector/nms_threshold_", nms_threshold_, 0.4);

	// Load network configuration
    ROS_INFO("Loading network.");

    // Load configuration files
	network_ = cv::dnn::readNet(model_path, config_path);

    network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    network_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

    // Get network configuration
	std::vector<int> output_layers = network_.getUnconnectedOutLayers();
	output_layer_type_ = network_.getLayer(output_layers[0])->type;

    std::vector<cv::String> layer_names = network_.getLayerNames();
    output_names_.resize(output_layers.size());
    for (size_t i = 0; i < output_layers.size(); ++i) {
        output_names_[i] = layer_names[output_layers[i] - 1];
    }

    ROS_INFO("Loaded network.");

    ROS_INFO("Loading classes.");
    // Load object classes
    std::ifstream ifs(classes_path.c_str());
    if (!ifs.is_open()) {
        ROS_ERROR("Class file %s not found", classes_path.c_str());
    } else {
	    std::string line;
	    while (std::getline(ifs, line))
	    {
	        classes_.push_back(line);
	    }
	}
    ROS_INFO("Loaded classes.");

    // Depth source control
    nh_private_.param<bool>("use_rangefinder", use_rangefinder_, false);
    nh_private_.param<bool>("use_depth", use_depth_, false);
    nh_private_.param<float>("fixed_depth", fixed_depth_, 1.5);

    // Camera subscriber
	image_sub_.subscribe(nh_, "/camera/image_raw", 1);
	camera_info_sub_ = nh_.subscribe("/camera/camera_info", 1, &DNNDetector::cameraInfoCallback, this);

	// Depth subscribers
    if(use_rangefinder_) { 
    	// TODO : load rangefinder calibration?
    	ROS_INFO("Depth from rangefinder enabled.");
    	range_sub_.subscribe(nh_, "range", 1);
    }

    if(use_depth_) {
    	ROS_INFO("Depth from stereo enabled.");
    	depth_sub_.subscribe(nh_, "image_depth", 1);
    }

    // Generate synchronised subscribers TODO : sanity check queue sizes
    if (use_depth_ && use_rangefinder_) {
    	image_depth_range_sync_.reset(new ImageDepthRangeSync(ImageDepthRangePolicy(50),
                                     image_sub_, depth_sub_, range_sub_));
    	image_depth_range_sync_->registerCallback(boost::bind(&DNNDetector::imageDepthRangeCallback,
                                              this, _1, _2, _3));

    } else if (use_depth_) {
    	image_depth_sync_.reset(new ImageDepthSync(ImageDepthPolicy(5),
                                     image_sub_, depth_sub_));
    	image_depth_sync_->registerCallback(boost::bind(&DNNDetector::imageDepthCallback,
                                              this, _1, _2));

    } else if (use_rangefinder_) {
    	image_range_sync_.reset(new ImageRangeSync(ImageRangePolicy(50),
                                     image_sub_, range_sub_));
    	image_range_sync_->registerCallback(boost::bind(&DNNDetector::imageRangeCallback,
                                              this, _1, _2));
    } else {
    	image_sub_.registerCallback(&DNNDetector::imageCallback, this);
    }

	// Initialize target publisher
	world_evidence_pub_ = nh_.advertise<perception_msgs::WorldEvidence>("world_evidence", 1);

	// Initialize image publisher for visualization
	image_transport::ImageTransport image_transport(nh_);
	image_pub_ = image_transport.advertise("detections_image", 1);

}

DNNDetector::~DNNDetector()
{

}

void DNNDetector::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg)
{
	if (!have_camera_info_) {

		if (msg->distortion_model.compare("plumb_bob") != 0 && msg->distortion_model.compare("equidistant") != 0) {
			ROS_ERROR_ONCE("Camera is not calibrated, or incompatible camera model (%s). Please check calibration.", 
				msg->distortion_model.c_str());
			return;
		}

		// Image Size
		camera_image_size_ = cv::Size(msg->width, msg->height);

		// Intrinsics
		camera_matrix_K_ = cv::Mat::eye(3, 3, CV_64FC1);
		camera_matrix_K_.at<double>(0, 0) = msg->K[0];
		camera_matrix_K_.at<double>(0, 1) = msg->K[1];
		camera_matrix_K_.at<double>(0, 2) = msg->K[2];
		camera_matrix_K_.at<double>(1, 0) = msg->K[3];
		camera_matrix_K_.at<double>(1, 1) = msg->K[4];
		camera_matrix_K_.at<double>(1, 2) = msg->K[5];
		camera_matrix_K_.at<double>(2, 0) = msg->K[6];
		camera_matrix_K_.at<double>(2, 1) = msg->K[7];
		camera_matrix_K_.at<double>(2, 2) = msg->K[8];

		// Distortion
		camera_equidistant_distortion_ = (msg->distortion_model.compare("equidistant") == 0);
		// TODO : verify
		camera_distortion_coeffs_.push_back(msg->D[0]);
		camera_distortion_coeffs_.push_back(msg->D[1]);
		camera_distortion_coeffs_.push_back(msg->D[2]);
		camera_distortion_coeffs_.push_back(msg->D[3]);

		have_camera_info_ = true;

		ROS_INFO("DNN detector initialized with camera parameters :");
		ROS_INFO("Image dimensions : %dx%d", msg->width, msg->height);
		ROS_INFO("fx = %f, fy = %f, cx = %f, cy = %f", msg->K[0], msg->K[4], msg->K[2], msg->K[5]);
		ROS_INFO("Distortion type : %s", camera_equidistant_distortion_ ? "Equidistant" : "Radial-Tangential");
		if (camera_equidistant_distortion_) {
			ROS_INFO("k1 = %f, k2 = %f, k3 = %f, k4 = %f", msg->D[0], msg->D[1], msg->D[2], msg->D[3]);
		} else {
			ROS_INFO("k1 = %f, k2 = %f, t1 = %f, t2 = %f", msg->D[0], msg->D[1], msg->D[2], msg->D[3]);
		}
	
	}

}

void DNNDetector::imageDepthRangeCallback(const sensor_msgs::Image::ConstPtr &image_msg, 
										  const sensor_msgs::Image::ConstPtr &depth_msg, 
										  const sensor_msgs::Range::ConstPtr &range_msg)
{
	// nothing
}

void DNNDetector::imageDepthCallback(const sensor_msgs::Image::ConstPtr &image_msg, 
										  const sensor_msgs::Image::ConstPtr &depth_msg)
{

}

void DNNDetector::imageRangeCallback(const sensor_msgs::Image::ConstPtr &image_msg,
									 const sensor_msgs::Range::ConstPtr &range_msg)
{

	// TODO : in the future compensate for angle from planar scene
	// Check whether already received the camera calibration data
	if (!have_camera_info_) {
		ROS_WARN("No camera calibration available yet.");
		return;
	}

	cv::Mat image;
	Utils::rosImageToCV(image_msg, image);

	NetworkPrediction prediction;
	prediction.timestamp = image_msg->header.stamp;
	prediction.frame_id = image_msg->header.frame_id;
	processImage(image, prediction);

	if (prediction.class_ids.size() > 0)
	{
		float depth = range_msg->range;	// Single point depth information available.

		for (size_t i = 0; i < prediction.class_ids.size(); ++i)
		{
		    prediction.depth.push_back(depth);
	    }

	    generateWorldEvidence(prediction);

	}

	generateVisualization(image, prediction); // TODO const image?
}

void DNNDetector::imageCallback(const sensor_msgs::Image::ConstPtr &image_msg)
{
	// Check whether already received the camera calibration data
	if (!have_camera_info_) {
		ROS_WARN("No camera calibration available yet.");
		return;
	}

	cv::Mat image;
	Utils::rosImageToCV(image_msg, image);

	NetworkPrediction prediction;
	prediction.timestamp = image_msg->header.stamp;
	prediction.frame_id = image_msg->header.frame_id;
	processImage(image, prediction);

	if (prediction.class_ids.size() > 0)
	{
		float depth = 1.0f;	// No scene depth information available, 
							// triangulation is only accurate upto a scale factor

		for (size_t i = 0; i < prediction.class_ids.size(); ++i)
		{
		    prediction.depth.push_back(depth);
	    }

	    generateWorldEvidence(prediction);

	}

	generateVisualization(image, prediction); // TODO const image?

}

void DNNDetector::processImage(const cv::Mat &image,
								NetworkPrediction &prediction)
{

	// Create a 4D blob from image frame
	cv::Mat blob = cv::dnn::blobFromImage(image, scale_, input_size_, mean_, swap_rb_, false);

	// Run inference
	network_.setInput(blob);
	if (network_.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
	    cv::resize(image, image, input_size_);
	    cv::Mat image_info = (cv::Mat_<float>(1, 3) << input_size_.height, input_size_.width, 1.6f);
	    network_.setInput(image_info, "im_info");
	}
	std::vector<cv::Mat> outputs;
	network_.forward(outputs, output_names_);

	// Get output from network
	processOutput(outputs, prediction);

	// Calculate performance
	std::vector<double> layer_times;
	double freq = cv::getTickFrequency() / 1000;
	double t = network_.getPerfProfile(layer_times) / freq;
	ROS_DEBUG("Inference time: %.2f ms", t);

}

void DNNDetector::processOutput(const std::vector<cv::Mat> &outputs, 
								NetworkPrediction &prediction) 
{

	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<cv::Point2f> centroids;

	if (network_.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
	    // Network produces output blob with a shape 1x1xNx7 where N is a number of
	    // detections and an every detection is a vector of values
	    // [batch_id, class_id, confidence, left, top, right, bottom]
	    CV_Assert(outputs.size() == 1);
	    float* data = (float*)outputs[0].data;
	    for (size_t i = 0; i < outputs[0].total(); i += 7)
	    {
	        float confidence = data[i + 2];
	        if (confidence > confidence_threshold_)
	        {
	            int left = (int)data[i + 3];
	            int top = (int)data[i + 4];
	            int right = (int)data[i + 5];
	            int bottom = (int)data[i + 6];
	            int width = right - left + 1;
	            int height = bottom - top + 1;
	            // TODO : calculate centroids
	            class_ids.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
	            boxes.push_back(cv::Rect(left, top, width, height));
	            confidences.push_back(confidence);
	        }
	    }
	}
	else if (output_layer_type_ == "DetectionOutput")
	{
	    // Network produces output blob with a shape 1x1xNx7 where N is a number of
	    // detections and an every detection is a vector of values
	    // [batch_id, class_id, confidence, left, top, right, bottom]
	    CV_Assert(outputs.size() == 1);
	    float* data = (float*)outputs[0].data;
	    for (size_t i = 0; i < outputs[0].total(); i += 7)
	    {
	        float confidence = data[i + 2];
	        if (confidence > confidence_threshold_)
	        {
	            int left = (int)(data[i + 3] * camera_image_size_.width);
	            int top = (int)(data[i + 4] * camera_image_size_.height);
	            int right = (int)(data[i + 5] * camera_image_size_.width);
	            int bottom = (int)(data[i + 6] * camera_image_size_.height);
	            int width = right - left + 1;
	            int height = bottom - top + 1;
	            // TODO : calculate centroids
	            class_ids.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
	            boxes.push_back(cv::Rect(left, top, width, height));
	            confidences.push_back(confidence);
	        }
	    }
	}
	else if (output_layer_type_ == "Region")
	{
	    for (size_t i = 0; i < outputs.size(); ++i)
	    {
	        // Network produces output blob with a shape NxC where N is a number of
	        // detected objects and C is a number of classes + 4 where the first 4
	        // numbers are [center_x, center_y, width, height]
	        float* data = (float*)outputs[i].data;
	        for (int j = 0; j < outputs[i].rows; ++j, data += outputs[i].cols)
	        {
	            cv::Mat scores = outputs[i].row(j).colRange(5, outputs[i].cols);
	            cv::Point class_id_point;
	            double confidence;
	            cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
	            if (confidence > confidence_threshold_)
	            {
	                int center_x = (int)(data[0] * camera_image_size_.width);
	                int center_y = (int)(data[1] * camera_image_size_.height);
	                int width = (int)(data[2] * camera_image_size_.width);
	                int height = (int)(data[3] * camera_image_size_.height);
	                int left = center_x - width / 2;
	                int top = center_y - height / 2;

	                class_ids.push_back(class_id_point.x);
	                boxes.push_back(cv::Rect(left, top, width, height));
	                centroids.push_back(cv::Point2f(center_x, center_y));
	                confidences.push_back((float)confidence);
	            }
	        }
	    }
	}
	else {
	    ROS_ERROR("Unknown network output layer type: %s", output_layer_type_.c_str());
	    return;
	}

	// Perform non-maximum suppression
	std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, indices);

	for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        prediction.class_ids.push_back(class_ids[idx]);
        prediction.boxes.push_back(boxes[idx]);
        prediction.centroids.push_back(centroids[idx]);
        prediction.confidences.push_back(confidences[idx]);
    }

}

void DNNDetector::generateVisualization(cv::Mat &image, NetworkPrediction &prediction) 
{

	if (image_pub_.getNumSubscribers() > 0) {

		for (size_t i = 0; i < prediction.class_ids.size(); ++i)
	    {
	        cv::Rect bbx = prediction.boxes[i];
	        Utils::drawPrediction(prediction.class_ids[i], prediction.confidences[i],
	        						bbx.x, bbx.y, bbx.x + bbx.width, bbx.y + bbx.height, 
	        						image);

			cv::circle(image, prediction.centroids[i], 1, CV_RGB(255, 0, 0), 2);
	    }

		cv_bridge::CvImage visualization_msg;
		visualization_msg.header.stamp = prediction.timestamp;
		visualization_msg.header.frame_id = prediction.frame_id;
		visualization_msg.encoding = sensor_msgs::image_encodings::RGB8;
		visualization_msg.image = image;

		// Publish the visualisation image
		image_pub_.publish(visualization_msg.toImageMsg());
	}

}

void DNNDetector::generateWorldEvidence(const NetworkPrediction &prediction)
{
	perception_msgs::WorldEvidence world_evidence;

	world_evidence.header.stamp = prediction.timestamp;
	world_evidence.header.frame_id = prediction.frame_id;

	// Undistort all centroids
	std::vector<cv::Point2f> undistorted_centroids;
	if(camera_equidistant_distortion_) {
		cv::fisheye::undistortPoints(prediction.centroids, undistorted_centroids, camera_matrix_K_, camera_distortion_coeffs_, cv::noArray(), camera_matrix_K_);
	} else {
		cv::undistortPoints(prediction.centroids, undistorted_centroids, camera_matrix_K_, camera_distortion_coeffs_, cv::noArray(), camera_matrix_K_);
	}

	for (size_t i = 0; i < prediction.class_ids.size(); ++i)
	{
	    Eigen::Vector3f position_camera; // Position of detection in camera frame
	    Eigen::Quaternionf orientation_camera; // Orientation of detection in camera frame
		Utils::imagePointToPosition(undistorted_centroids[i], prediction.depth[i], camera_matrix_K_, position_camera);

		addObjectEvidence(position_camera, orientation_camera,
						  classes_[prediction.class_ids[i]], prediction.confidences[i],
						  world_evidence);
	}

	// Publish
	world_evidence_pub_.publish(world_evidence);

}

void DNNDetector::addObjectEvidence(const Eigen::Vector3f &position,
									 const Eigen::Quaternionf &orientation,
									 const std::string &class_label, 
									 float class_confidence,
									 perception_msgs::WorldEvidence &world_evidence) 
{

	// TODO : estimate variance of position using squared scene depth
	// TODO : use orientation and covariances

    perception_msgs::ObjectEvidence obj_evidence;

    // Position
    perception_msgs::Property position_property;
    position_property.attribute = "position";

    // Set position, with covariance matrix as 0.05*identity_matrix
    pbl::PDFtoMsg(pbl::Gaussian(pbl::Vector3(position.x(), position.y(), position.z()), pbl::Matrix3(0.05, 0.05, 0.05)), position_property.pdf);
    obj_evidence.properties.push_back(position_property);

    // Orientation
    perception_msgs::Property orientation_property;
    orientation_property.attribute = "orientation";

    // Set the orientation to unity (0,0,0,1), with covariance matrix 0.01*identity_matrix
    pbl::PDFtoMsg(pbl::Gaussian(pbl::Vector4(0, 0, 0, 1), pbl::Matrix4(0.01, 0.01, 0.01, 0.01)), orientation_property.pdf);
    obj_evidence.properties.push_back(orientation_property);

    // Object type
    perception_msgs::Property class_property;
    class_property.attribute = "class_label"; 
    pbl::PMF class_pmf;

    // Use confidence from network as probability of the prediction being correct
    class_pmf.setProbability(class_label, class_confidence);
    pbl::PDFtoMsg(class_pmf, class_property.pdf);
    obj_evidence.properties.push_back(class_property);

    // Add all properties to the array
    world_evidence.object_evidence.push_back(obj_evidence);
}

} // namespace dnn_detector
             
