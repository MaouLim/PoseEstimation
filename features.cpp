#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

const std::string image1_path = "pose2d/1.jpg";
const std::string image2_path = "pose2d/2.jpg";

int main() {
	std::cout << "OpenCV Version: " << cv::getVersionString() << std::endl;

	cv::Mat img1 = cv::imread(image1_path, cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread(image2_path, cv::IMREAD_COLOR);

	std::vector<cv::KeyPoint> key_points1, key_points2;
	cv::Mat desc1, desc2;

	cv::Ptr<cv::Feature2D> detector = cv::ORB::create();

	detector->detect(img1, key_points1);
	detector->detect(img2, key_points2);

	detector->compute(img1, key_points1, desc1);
	detector->compute(img2, key_points2, desc2);

	std::cout << "Key Points Num: " << key_points1.size() << std::endl;
	std::cout << "desc1 shape: " << desc1.size << std::endl;
	std::cout << "desc1 depth: " << desc1.depth() << std::endl;

	cv::Mat draw_kp1, draw_kp2;
	cv::drawKeypoints(img1, key_points1, draw_kp1);
	cv::drawKeypoints(img2, key_points2, draw_kp2);

	cv::imshow("draw_kp1", draw_kp1);
	cv::imshow("draw_kp2", draw_kp2);
	cv::waitKey();

	std::vector<cv::DMatch> matches(cv::NORM_HAMMING);
	cv::BFMatcher matcher;
	matcher.match(desc1, desc2, matches);

	double min_dis = UINT32_MAX, max_dis = 0.;

	for (auto i = 0; i < desc1.rows; ++i) {
		double dis = matches[i].distance;
		if (dis < min_dis) { min_dis = dis; }
		if (max_dis < dis) { max_dis = dis; }
	}

	std::cout << "Max distance: " << max_dis << std::endl;
	std::cout << "Min distance: " << min_dis << std::endl;

	std::vector<cv::DMatch> good_matches;
	for (auto i = 0; i < desc1.rows; ++i) {
		if (matches[i].distance <= min_dis * 2) {
			good_matches.push_back(matches[i]);
		}
	}

	cv::Mat draw_matches, draw_good_matches;
	cv::drawMatches(img1, key_points1, img2, key_points2, matches, draw_matches);
	cv::drawMatches(img1, key_points1, img2, key_points2, good_matches, draw_good_matches);

	cv::imshow("matches", draw_matches);
	cv::imshow("good matches", draw_good_matches);

	cv::waitKey();

	return 0;
}














