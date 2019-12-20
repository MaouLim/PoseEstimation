/*
 * Created by Maou Lim on 2019/12/16.
 */

#include "orb_extract.hpp"

const std::string image1_path = "pose2d/1.jpg";
const std::string image2_path = "pose2d/2.jpg";

//float intrinsic[] = {
//	698.3,   0.0, 297.9,
//	  0.0, 698.4, 294.8,
//	  0.0,   0.0,   1.0,
//};

//const cv::Mat camera_mat(3, 3, CV_32F, intrinsic);
const cv::Mat camera_mat = (
	cv::Mat_<float>(3, 3) <<
		698.3,   0.0, 297.9,
          0.0, 698.4, 294.8,
          0.0,   0.0,   1.0
);
int main(int argc, char** argv) {

	cv::Mat img1 = cv::imread(image1_path, cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread(image2_path, cv::IMREAD_COLOR);

	std::vector<cv::KeyPoint> key_points1, key_points2;
	std::vector<cv::DMatch> matches;
	extract_orb(img1, img2, key_points1, key_points2, matches);

	cv::Mat draw_matches;
	cv::drawMatches(img1, key_points1, img2, key_points2, matches, draw_matches);
	cv::imshow("matches", draw_matches);
	cv::waitKey();

	std::vector<cv::Point2f> points1, points2;
	points1.reserve(matches.size());
	points2.reserve(matches.size());

	for (auto& a_match : matches) {
		points1.emplace_back(key_points1[a_match.queryIdx].pt);
		points2.emplace_back(key_points1[a_match.trainIdx].pt);
	}

	cv::Mat fundamental = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
	cv::Mat essential = cv::findEssentialMat(points1, points2, camera_mat);
	cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC);

	std::cout << "fundamental:\n" << fundamental << std::endl;
	std::cout << "essential:\n" << essential << std::endl;
	std::cout << "homography:\n" << homography << std::endl;

	cv::Mat rotation, translation;
	cv::recoverPose(essential, points1, points2, camera_mat, rotation, translation);

	std::cout << "rotation:\n" << rotation << std::endl;
	std::cout << "translation:\n" << translation << std::endl;
	std::cout << "norm of translation: " << cv::norm(translation) << std::endl;

	double t0 = translation.at<double>(0);
	double t1 = translation.at<double>(1);
	double t2 = translation.at<double>(2);

	cv::Mat t_hat = (
		cv::Mat_<double>(3, 3) <<
		      0, -t2,  t1,
			 t2,   0, -t0,
			-t1,  t0,   0
	);

	cv::Mat essential_bar = t_hat * rotation;
	std::cout << "essential_bar:\n" << essential_bar << std::endl;

	double ratio = essential.at<double>(0, 0) / essential_bar.at<double>(0, 0);
	essential_bar *= ratio;
	std::cout << "after scaling essential_bar:\n" << essential_bar << std::endl;

	essential_bar.convertTo(essential_bar, CV_32F);
	cv::Mat cam_mat_inv = camera_mat.inv();
	for (auto i = 0; i < matches.size(); ++i) {
		auto p1 = points1[i];
		cv::Vec3f p1_homo(p1.x, p1.y, 1.f);
		auto p2 = points2[i];
		cv::Vec3f p2_homo(p2.x, p2.y, 1.f);
		auto res = (cam_mat_inv * p2_homo).t() * essential_bar * (cam_mat_inv * p1_homo);
		std::cout << i << " res: " << res << std::endl;
	}

	std::vector<cv::Point3f> points3d;
	triangulation(points1, points2, rotation, translation, camera_mat, points3d);

	for (auto i = 0; i < points3d.size(); ++i) {
		std::cout << "---------------------" << i << "---------------------" << std::endl;
		auto& p1_s = points1[i];
		auto& p1_c = points3d[i];
		std::cout << "point in img1 (normalized camera coordinate): "
		          << cam_mat_inv * cv::Vec3f(p1_s.x, p1_s.y, 1.f) << std::endl;
		std::cout << "point projected from triangulation: " << p1_c << std::endl;
		std::cout << "point projected from triangulation (normalized): "
		          << p1_c / p1_c.z << std::endl;

		auto& p2_s = points2[i];
		cv::Mat p2_c = rotation * cv::Vec3d(p1_c.x, p1_c.y, p1_c.z) + translation;
		std::cout << "point in img2 (normalized camera coordinate): "
		          << cam_mat_inv * cv::Vec3f(p2_s.x, p2_s.y, 1.f) << std::endl;
		std::cout << "point projected from triangulation: " << p2_c << std::endl;
		std::cout << "point projected from triangulation (normalized): "
		          << p2_c / p2_c.at<double>(2, 0) << std::endl;
		std::cout << "---------------------" << i << "---------------------" << std::endl;
	}

	return 0;
}
