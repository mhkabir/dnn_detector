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

#include "dnn_detector/utils.h"

namespace dnn_detector
{

void Utils::drawPrediction(int class_id, float conf,
							int left, int top, int right, int bottom, 
							cv::Mat &image)
{
    cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

    std::string label = cv::format("%.2f", conf);
    /*
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }*/

    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    top = std::max(top, label_size.height);
    // Draw bounding box
    cv::rectangle(image, cv::Point(left, top - label_size.height),
              cv::Point(left + label_size.width, top + baseline), cv::Scalar::all(255), cv::FILLED);
    cv::putText(image, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());

}

void Utils::rosImageToCV(const sensor_msgs::Image::ConstPtr &image_msg, cv::Mat &image) {

	cv_bridge::CvImageConstPtr cv_ptr;

	try {
		cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::RGB8);

	} catch (cv_bridge::Exception &e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	image = cv_ptr->image;
}

void Utils::imagePointToPosition(const cv::Point2f &point, float depth, const cv::Mat &camera_matrix_K, 
								Eigen::Vector3f &position_camera)
{
	double fx = camera_matrix_K.at<double>(0, 0);
	double fy = camera_matrix_K.at<double>(1, 1);
	double cx = camera_matrix_K.at<double>(0, 2);
	double cy = camera_matrix_K.at<double>(1, 2);

	// TODO : check math
	position_camera = Eigen::Vector3f(depth * (point.x - cx) / fx,
									  depth * (point.y - cy) / fy,
									  depth);

}

}