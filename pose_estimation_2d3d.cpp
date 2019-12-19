/*
 * Created by Maou Lim on 2019/12/18.
 */

#include <chrono>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace chrono = std::chrono;

void extract_orb(
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

typedef Eigen::Vector3f            vec3f;
typedef Eigen::Vector2f            vec2f;
typedef Eigen::Matrix<float, 6, 1> vec6f;
typedef Eigen::Matrix<float, 6, 6> mat6f;
typedef Eigen::Matrix<float, 2, 6> mat2x6f;

int bundle_adjustment_gaussnewton(
	const std::vector<vec3f>& obj_points,
	const std::vector<vec2f>& img_points,
	const cv::Mat&            cam_mat,
	Sophus::SE3f&             pose
) {
	assert(obj_points.size() == img_points.size());

	const size_t n_points = obj_points.size();
	const size_t max_iterations = 20;

	const float fx = cam_mat.at<float>(0, 0);
	const float fy = cam_mat.at<float>(1, 1);
	const float cx = cam_mat.at<float>(0, 2);
	const float cy = cam_mat.at<float>(1, 2);

	float prev_loss = 0.f;
	auto iteration = 0;

	while (iteration < max_iterations) {
		mat6f h = mat6f::Zero();
		vec6f g = vec6f::Zero();

		float loss = 0.f;

		for (auto j = 0; j < n_points; ++j) {
			/* compute the 3d position under pose */
			vec3f p = pose * obj_points[j];
			const float x = p[0], y = p[1], z = p[2];
			const float x2 = x * x, y2 = y * y, z2 = z * z;

			/* compute the 2d screen coordiate in view2 */
			vec2f p_proj(fx * x / z + cx, fy * y / z + cy);

			auto err = img_points[j] - p_proj;
			loss += err.squaredNorm() * 0.5f;

			mat2x6f jacobian;
			jacobian <<
				-fx / z2,     0.f, fx * x / z2,   fx * x * y / z2, -fx - fx * x2 / z2,  fx * y / z,
				     0.f, -fy / z, fy * y / z2, fy + fy * y2 / z2,   -fy * x * y / z2, -fy * x / z;

			h += jacobian.transpose() * jacobian;
			g += jacobian.transpose() * -err;
		}

		vec6f dx = h.ldlt().solve(g);

		if (std::isnan(dx[0])) { std::cerr << "result is nan." << std::endl; break; }
		if (0 < iteration && prev_loss < loss * 1.1) { std::cerr << "loss is increasing." << std::endl; break; }
		if (dx.norm() < 1e-8) { break; }

		pose = Sophus::SE3f::exp(dx) * pose;
		prev_loss = loss;
		++iteration;
	}

	//std::cout << "iteration: " << iteration << std::endl;
	//std::cout << "loss: " << prev_loss << std::endl;
	return iteration;
}

const std::string image1_path = "pose3d/1.png";
const std::string depth1_path = "pose3d/1_depth.png";
const std::string image2_path = "pose3d/2.png";

const cv::Mat cam_mat = (
	cv::Mat_<float>(3, 3) <<
	    520.9f,   0.f, 325.1f,
	       0.f, 521.f, 249.7f,
	       0.f,   0.f,    1.f
);

const cv::Mat cam_mat_inv = cam_mat.inv();

int main(int argc, char** argv) {

	cv::Mat img1 = cv::imread(image1_path, cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread(image2_path, cv::IMREAD_COLOR);

	const float depth_scale = 1. / 5000.;
	cv::Mat depth1 = cv::imread(depth1_path, cv::IMREAD_UNCHANGED);
	std::cout << depth1.depth() << std::endl;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector<cv::DMatch> matches;
	extract_orb(img1, img2, keypoints1, keypoints2, matches);

	std::vector<cv::Point3f> points3d; // created by img1 and depth1
	std::vector<cv::Point2f> points2d; // created by img2

	points3d.reserve(matches.size());
	points2d.reserve(matches.size());

	for (auto i = 0; i < matches.size(); ++i) {
		auto kp1 = keypoints1[matches[i].queryIdx].pt;
		auto kp2 = keypoints2[matches[i].trainIdx].pt;

		uint16_t d = depth1.at<uint16_t>(kp1.y, kp1.x);
		std::cout << d << std::endl;
		if (0 == d) { continue; }

		float z = float(d) * depth_scale;
		cv::Mat p1_c = cam_mat_inv * cv::Vec3f(kp1.x, kp1.y, 1.f) * z;
		points3d.emplace_back(p1_c.at<float>(0, 0), p1_c.at<float>(1, 0), p1_c.at<float>(2, 0));
		points2d.emplace_back(kp2);
	}

	std::cout << "n_pairs: " << points2d.size() << std::endl;

	/**
	 * @brief pnp algorithm
	 *        EPNP      time used: < 1ms
	 *        DLS       time used: < 1ms
	 *        ITERATIVE time used: 8ms
	 */ {
		cv::Mat rvec, rotation, translation;

		chrono::steady_clock::time_point start = chrono::steady_clock::now();
		cv::solvePnP(points3d, points2d, cam_mat, cv::Mat(), rvec, translation, false, cv::SOLVEPNP_ITERATIVE);
		cv::Rodrigues(rvec, rotation);
		chrono::steady_clock::time_point end = chrono::steady_clock::now();

		std::cout << "rotation:\n" << rotation << std::endl;
		std::cout << "translation:\n" << translation << std::endl;
		std::cout << "time used: "
		          << chrono::duration_cast<chrono::milliseconds>(end - start).count()
		          << std::endl;
	}

	/**
	 * @brief using bundle adjustment solve 2d-3d pose estimation
	 *        optimizer: gauss newton
	 *        time used: 6ms
	 */ {
		std::vector<vec3f> obj_points;
		std::vector<vec2f> img_points;
		Sophus::SE3f pose;

		obj_points.reserve(points3d.size());
		img_points.reserve(points2d.size());
		for (auto& each : points3d) {
			obj_points.emplace_back(each.x, each.y, each.z);
		}
		for (auto& each : points2d) {
			img_points.emplace_back(each.x, each.y);
		}
		chrono::steady_clock::time_point start = chrono::steady_clock::now();
		bundle_adjustment_gaussnewton(obj_points, img_points, cam_mat, pose);
		auto rotation = pose.rotationMatrix();
		auto translation = pose.translation();
		chrono::steady_clock::time_point end = chrono::steady_clock::now();

		std::cout << "rotation:\n" << rotation << std::endl;
		std::cout << "translation:\n" << translation << std::endl;
		std::cout << "time used: "
		          << chrono::duration_cast<chrono::milliseconds>(end - start).count()
		          << std::endl;
	 }

	return 0;
}