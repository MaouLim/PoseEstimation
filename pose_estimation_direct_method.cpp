/**
 * Created by Maou Lim on 2019/12/26.
 * @note pose estimation using sparse direct method
 */

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Vector2d             vec2d;
typedef Eigen::Vector3d             vec3d;
typedef Eigen::Matrix3d             mat3d;
typedef Eigen::Matrix<double, 6, 6> mat6d;
typedef Eigen::Matrix<double, 2, 6> mat2x6d;
typedef Eigen::Matrix<double, 6, 1> vec6d;

inline double _val(const cv::Mat& img, double x, double y) {
	int x_ = std::floor(x);
	int y_ = std::floor(y);

	if (x_ < 0) { x_ = 0; }
	if (x_ >= img.cols) { x_ = img.cols - 1; }
	if (y_ < 0) { y_ = 0; }
	if (y_ >= img.rows) { y_ = img.rows - 1; }

	double v00 = (double) img.at<uint8_t>(y_, x_);
	double v01 = (double) img.at<uint8_t>(y_, x_ + 1);
	double v10 = (double) img.at<uint8_t>(y_ + 1, x_);
	double v11 = (double) img.at<uint8_t>(y_ + 1, x_ + 1);

	return v11 * (x - x_) * (y - y_) +
	       v10 * (x_ + 1 - x) * (y - y_) +
	       v01 * (x - x_) * (y_ + 1 - y) +
	       v00 * (x_ + 1 - x) * (y_ + 1 - y);
}

inline vec2d _grad(const cv::Mat& img, double x, double y) {
	int x_ = std::floor(x);
	int y_ = std::floor(y);

	if (x_ < 0) { x_ = 0; }
	if (x_ >= img.cols) { x_ = img.cols - 1; }
	if (y_ < 0) { y_ = 0; }
	if (y_ >= img.rows) { y_ = img.rows - 1; }

	double v00 = (double) img.at<uint8_t>(y_, x_);
	double v01 = (double) img.at<uint8_t>(y_, x_ + 1);
	double v10 = (double) img.at<uint8_t>(y_ + 1, x_);
	double v11 = (double) img.at<uint8_t>(y_ + 1, x_ + 1);

	return vec2d(
		(y_ + 1 - y) * (v01 - v00) + (y - y_) * (v11 - v10),
		(x_ + 1 - x) * (v10 - v00) + (x - x_) * (v11 - v01)
	);
}

inline mat3d rodriguez(const vec3d& x) {
	double theta = x.norm();
	vec3d rho = x / theta;

	const double cos_theta = std::cos(theta);
	const double sin_theta = std::sin(theta);
	const mat3d identity = mat3d::Identity();
	mat3d rho_hat;
	rho_hat <<
	        0., -rho[2],  rho[1],
		rho[2],      0., -rho[0],
		-rho[1],  rho[0],      0.;
	return identity * cos_theta + (1. - cos_theta) * rho * rho.transpose() + rho_hat * sin_theta;
}

inline vec3d jacobian(const vec6d& x) {
	const vec3d rho = x.head<3>();
	const vec3d phi = x.tail<3>();

	const double theta = phi.norm();
	const vec3d a = phi / theta;

	const double sin = std::sin(theta);
	const double cos = std::cos(theta);

	mat3d a_hat;
	a_hat <<
	      0., -a.z(),  a.y(),
		a.z(),     0., -a.x(),
		-a.y(),  a.x(),     0.;

	mat3d j =
		(sin / theta) * mat3d::Identity() +
		(1. - sin / theta) * a * a.transpose() +
		(1. - cos) / theta * a_hat;

	return j * rho;
}

void direct_method_gauss_newton(
	const cv::Mat&            prev,
	const cv::Mat&            next,
	const std::vector<vec3d>& points,
	const mat3d&              cam_mat,
	mat3d&                    rotation,
	vec3d&                    translation
) {
	const int max_iterations = 10;
	const double fx = cam_mat(0, 0);
	const double fy = cam_mat(1, 1);

	double prev_loss = 0.;
	auto iter = 0;
	while (iter < max_iterations) {

		mat6d h = mat6d::Zero();
		vec6d g = vec6d::Zero();
		double loss = 0.;

		for (auto i = 0; i < points.size(); ++i) {
			const vec3d& p = points[i];

			vec3d p1_homo = cam_mat * p;
			vec2d p1 = vec2d(p1_homo[0]/ p1_homo[2], p1_homo[1]/ p1_homo[2]);
			vec3d p2_homo = cam_mat * (rotation * p + translation);
			vec2d p2 = vec2d(p2_homo[0]/ p2_homo[2], p2_homo[1]/ p2_homo[2]);

			double err = _val(prev, p1.x(), p1.y()) - _val(next, p2.x(), p2.y());
			loss += 0.5 * err * err;
			/**
			 * @var i: image
			 *      u: pixel
			 */
			vec2d didu = _grad(next, p2.x(), p2.y());
			/**
			 * @note delta x is Lie algebra
			 */

			double x = p2_homo.x(), y = p2_homo.y(), z = p2_homo.z();
			double x2 = x * x, y2 = y * y, z2 = z * z;

			mat2x6d dudx;
			dudx <<
			     fx / z,    0.f,    -fx * x / z2,   -fx * x * y / z2,   fx + fx * x2 / z2,  -fx * y / z,
				    0.f, fy / z,    -fy * y / z2, -fy - fy * y2 / z2,     fy * x * y / z2,   fy * x / z;

			/**
			 * @note linear part of loss function (扰动模型).
			 */
			vec6d jacobian = -didu.transpose() * dudx;

			h += jacobian * jacobian.transpose();
			g += jacobian * -err;
		}

		vec6d dx = h.ldlt().solve(g);

		if (std::isnan(dx[0])) { break; }
		//if (0 != iter && prev_loss * 2 < loss) { std::cerr << "loss inc." << std::endl; break; }
		if (dx.norm() < 1e-8) { break; }

		// update
		mat3d delta_r = rodriguez(dx.tail<3>());
		vec3d delta_t = jacobian(dx);

		rotation = delta_r * rotation;
		translation = delta_r * translation + delta_t;

		prev_loss = loss;
		++iter;
	}
}

int main(int argc, char** argv) {

	const int n_images = 13;

	std::vector<cv::Mat> images(n_images);
	for (auto i = 0; i < n_images; ++i) {
		char tmp[16];
		sprintf(tmp, "seq/gry%.2d.jpg", i + 1);
		images[i] = cv::imread(std::string(tmp), cv::IMREAD_GRAYSCALE);
	}

	const cv::Mat& origin = images.front();

	std::vector<cv::KeyPoint> key_points;
	cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(200, 0.01, 20);
	detector->detect(origin, key_points);



	return 0;
}














