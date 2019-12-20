/*
 * Created by Maou Lim on 2019/12/20.
 */

#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "orb_extract.hpp"

namespace chrono = std::chrono;

typedef Eigen::Vector3d             vec3d;
typedef Eigen::Matrix3d             mat3d;
typedef Eigen::Matrix<double, 6, 1> vec6d;
typedef Eigen::Matrix<double, 6, 6> mat6d;
typedef Eigen::Matrix<double, 3, 6> mat3x6d;

void get_3d_point_sets(
	const mat3d&        cam_mat,
	std::vector<vec3d>& points1,
	std::vector<vec3d>& points2
) {
	static const std::string img1_path = "pose3d/1.png";
	static const std::string img2_path = "pose3d/2.png";
	static const std::string dep1_path = "pose3d/1_depth.png";
	static const std::string dep2_path = "pose3d/2_depth.png";
	static const double depth_scale = 1. / 5000.;

	cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_COLOR);
	cv::Mat depth1 = cv::imread(dep1_path, cv::IMREAD_UNCHANGED);
	cv::Mat depth2 = cv::imread(dep2_path, cv::IMREAD_UNCHANGED);

	std::vector<cv::KeyPoint> key_points1, key_points2;
	std::vector<cv::DMatch> matches;

	extract_orb(img1, img2, key_points1, key_points2, matches);

	auto cam_mat_inv = cam_mat.inverse();

	for (auto& each : matches) {
		auto& p = key_points1[each.queryIdx].pt;
		auto& q = key_points2[each.trainIdx].pt;

		uint16_t pdepth = depth1.at<uint16_t>(p.y, p.x);
		uint16_t qdepth = depth2.at<uint16_t>(q.y, q.x);
		if (0 == pdepth || 0 == qdepth) { continue; }

		double pd = double(pdepth) * depth_scale;
		double qd = double(qdepth) * depth_scale;

		points1.emplace_back(cam_mat_inv * vec3d(p.x, p.y, 1.) * pd);
		points2.emplace_back(cam_mat_inv * vec3d(q.x, q.y, 1.) * qd);
	}
}

template <typename _EigenMatrix>
_EigenMatrix _center_of(const std::vector<_EigenMatrix>& set) {
	typedef typename _EigenMatrix::value_type number_t;
	_EigenMatrix res = _EigenMatrix::Zero();
	for (auto& member : set) {
		res += member;
	}
	return res / (number_t) set.size();
}

void icp_svd(
	const std::vector<vec3d>& points1,
    const std::vector<vec3d>& points2,
    mat3d&                    rotation,
    vec3d&                    translation
) {
	assert(points1.size() == points2.size());
	size_t n_pairs = points1.size();

	const auto c1 = _center_of(points1);
	const auto c2 = _center_of(points2);

	mat3d w = mat3d::Zero();
	for (auto i = 0; i < n_pairs; ++i) {
		auto q1 = points1[i] - c1;
		auto q2 = points2[i] - c2;
		w += q1 * q2.transpose();
	}

	Eigen::JacobiSVD<mat3d> jacobi_svd(w, Eigen::ComputeFullU | Eigen::ComputeFullV);
	mat3d u = jacobi_svd.matrixU();
	mat3d v = jacobi_svd.matrixV();

	rotation = u * v.transpose();
	translation = c1 - rotation * c2;
}

inline mat3d rodriguez(const vec3d& x) {
	double theta = x.norm();
	vec3d rho = x / theta;

	const double cos_theta = std::cos(theta);
	const double sin_theta = std::sin(theta);
	const mat3d identity = mat3d::Identity();
	mat3d x_hat;
	x_hat <<
	       0., -x[2],  x[1],
	     x[2],    0., -x[0],
	    -x[1],  x[0],    0.;
	return identity * cos_theta + (1. - cos_theta) * rho * rho.transpose() + x_hat * sin_theta;
}

int icp_gauss_newton(
	const std::vector<vec3d>& points1,
	const std::vector<vec3d>& points2,
	mat3d&                    rotation,
	vec3d&                    translation
) {
	assert(points1.size() == points2.size());
	size_t n_pairs = points1.size();

	const auto max_iterations = 10;
	double prev_loss = 0.;
	auto iter = 0;

	while (iter < max_iterations) {

		double loss = 0.;
		mat6d h = mat6d::Zero();
		vec6d g = vec6d::Zero();

		for (auto i = 0; i < n_pairs; ++i) {

			auto& p1 = points1[i];
			auto& p2 = points2[i];
			double x = p2[0], y = p2[0], z = p2[0];

			auto err = p1 - (rotation * p2 + translation);
			loss += 0.5 * err.squaredNorm();

			mat3x6d jacobian;
			jacobian <<
				1., 0., 0., 0., -z,  y,
				0., 1., 0.,  z, 0., -x,
				0., 0., 1., -y,  x, 0.;
			jacobian *= -1.;

			h += jacobian.transpose() * jacobian;
			g += jacobian.transpose() * -err;
		}

		auto dx = h.ldlt().solve(g);

		if (std::isnan(dx[0])) { std::cerr << "result is nan." << std::endl; break; }
		if (0 < iter && prev_loss < loss * 1.1) { std::cerr << "loss is increasing." << std::endl; break; }
		if (dx.norm() < 1e-8) { break; }

		auto phi = vec3d(dx[3], dx[4], dx[5]);
		auto rho = vec3d(dx[0], dx[1], dx[2]);

		auto delta_r = rodriguez(phi);

		rotation = delta_r * rotation;
		translation = delta_r * translation + rho;
		++iter;
	}

	return iter;
}

int main(int argc, char** argv) {

	static mat3d cam_mat;
	cam_mat <<
		520.9,   0., 325.1,
	       0., 521., 249.7,
	       0.,   0.,    1.;

	std::vector<vec3d> pts1, pts2;
	get_3d_point_sets(cam_mat, pts1, pts2);

	/**
	 * @brief self-implementation of icp using svd decomposition
	 */ {
		mat3d rotation;
		vec3d translation;
		chrono::steady_clock::time_point start = chrono::steady_clock::now();
		icp_svd(pts1, pts2, rotation, translation);
		chrono::steady_clock::time_point end = chrono::steady_clock::now();

		std::cout << "rotation:\n" << rotation << std::endl;
		std::cout << "translation:\n" << translation << std::endl;
		std::cout << "time used: "
		          << chrono::duration_cast<chrono::milliseconds>(end - start).count()
		          << std::endl;
	}

	/**
	 * @brief self-implementation of icp using BA
	 *        optimizer: gauss-newton
	 */ {
		mat3d rotation;
		vec3d translation;
		chrono::steady_clock::time_point start = chrono::steady_clock::now();
		icp_gauss_newton(pts1, pts2, rotation, translation);
		chrono::steady_clock::time_point end = chrono::steady_clock::now();

		std::cout << "rotation:\n" << rotation << std::endl;
		std::cout << "translation:\n" << translation << std::endl;
		std::cout << "time used: "
		          << chrono::duration_cast<chrono::milliseconds>(end - start).count()
		          << std::endl;
	}

	return 0;
}