/*
 * Created by Maou Lim on 2019/12/20.
 */

#ifndef _ORB_EXTRACT_HPP_
#define _ORB_EXTRACT_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

void triangulation(
	const std::vector<cv::Point2f>& points1,
	const std::vector<cv::Point2f>& points2,
	const cv::Mat&                  rotation,
	const cv::Mat&                  translation,
	const cv::Mat&                  cam_mat,
	std::vector<cv::Point3f>&       points3d
) {
	cv::Mat t1 = (cv::Mat_<float>(3, 4) <<
	                                    1.f, 0.f, 0.f, 0.f,
		0.f, 1.f, 0.f, 0.f,
		0.f, 0.f, 1.f, 0.f
	);

	const auto& r = rotation;
	const auto& t = translation;

	cv::Mat t2 = (
		cv::Mat_<float>(3, 4) <<
		                      r.at<double>(0, 0), r.at<double>(0, 1), r.at<double>(0, 2), t.at<double>(0, 0),
			r.at<double>(1, 0), r.at<double>(1, 1), r.at<double>(1, 2), t.at<double>(1, 0),
			r.at<double>(2, 0), r.at<double>(2, 1), r.at<double>(2, 2), t.at<double>(2, 0)
	);

	cv::Mat cam_mat_inv = cam_mat.inv();
	cam_mat_inv.convertTo(cam_mat_inv, CV_32F);

	const size_t n_matches = points1.size();
	// normalized camera coordinates without z-axis
	std::vector<cv::Point2f> points1_cam;
	std::vector<cv::Point2f> points2_cam;
	points1_cam.reserve(n_matches);
	points2_cam.reserve(n_matches);

	for (auto i = 0; i < n_matches; ++i) {
		auto& p = points1[i];
		cv::Mat p_cam = cam_mat_inv * cv::Vec3f(p.x, p.y, 1.f);
		points1_cam.emplace_back(p_cam.at<float>(0, 0), p_cam.at<float>(1, 0));
		auto& q = points2[i];
		cv::Mat q_cam = cam_mat_inv * cv::Vec3f(q.x, q.y, 1.f);
		points2_cam.emplace_back(q_cam.at<float>(0, 0), q_cam.at<float>(1, 0));
	}
	cv::Mat points3d_homo;
	cv::triangulatePoints(t1, t2, points1_cam, points2_cam, points3d_homo);

	points3d.clear();
	for (auto i = 0; i < points3d_homo.cols; ++i) {
		cv::Mat x = points3d_homo.col(i);
		x /= x.at<float>(3, 0);
		points3d.emplace_back(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
	}
}

inline void extract_orb(
	const cv::Mat& img1, const cv::Mat& img2,
	std::vector<cv::KeyPoint>& key_points1,
	std::vector<cv::KeyPoint>& key_points2,
	std::vector<cv::DMatch>&   matches
) {
	cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

	orb->detect(img1, key_points1);
	orb->detect(img2, key_points2);

	cv::Mat desc1, desc2;
	orb->compute(img1, key_points1, desc1);
	orb->compute(img2, key_points2, desc2);

	matcher->match(desc1, desc2, matches);

	double min_dis = UINT32_MAX, max_dis = 0.;

	for (auto i = 0; i < desc1.rows; ++i) {
		double dis = matches[i].distance;
		if (dis < min_dis) { min_dis = dis; }
		if (max_dis < dis) { max_dis = dis; }
	}

	int count_good = 0;
	for (auto i = 0; i < desc1.rows; ++i) {
		if (matches[i].distance <= min_dis * 3) {
			matches[count_good++] = matches[i];
		}
	}
	matches.resize(count_good);
}

#endif //_ORB_EXTRACT_HPP_
