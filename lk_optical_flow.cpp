/*
 * Created by Maou Lim on 2019/12/24.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
//#include <Eigen/LU>

typedef Eigen::Vector2d  vec2d;
typedef Eigen::Matrix2d  mat2d;
typedef Eigen::MatrixX2d matxx2d;
typedef Eigen::VectorXd  vecxd;

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

void lk_optical_flow_single1(
	const cv::Mat&                   prev,
	const cv::Mat&                   next,
	const std::vector<cv::Point2f>&  pts_prev,
	size_t                           win_sz,
	std::vector<cv::Point2f>&        pts_next,
	std::vector<uint8_t>&            status,
	bool                             use_init = false
) {
	assert(1 == win_sz % 2);

	const int    n_points = pts_prev.size();
	const int    half_w = win_sz / 2;
	const int    max_iterations = 10;

	pts_next.resize(n_points);
	status.resize(n_points, 1);

	for (auto i = 0; i < n_points; ++i) {

		const cv::Point2f& p = pts_prev[i];
		double loss = 0., prev_loss = 0.;
		double u = 0., v = 0.;
		if (use_init) {
			u = pts_next[i].x - pts_prev[i].x;
			v = pts_next[i].y - pts_prev[i].y;
		}

		mat2d hessian = mat2d::Zero();
		vec2d g       = vec2d::Zero();
		vec2d jacobi;

		auto iter = 0;

		while (iter < max_iterations) {
			hessian = mat2d::Zero();
			g       = vec2d::Zero();
			loss    = 0.;

			for (auto x = -half_w; x <= half_w; ++x) {
				for (auto y = -half_w; y <= half_w; ++y) {
					double err = _val(prev, p.x + x, p.y + y) -
						         _val(next, p.x + x + u, p.y + y + v);
					loss += 0.5 * err * err;

					jacobi <<
						-0.5 * (_val(next, p.x + x + u + 1, p.y + y + v) -
							    _val(next, p.x + x + u - 1, p.y + y + v)),
						-0.5 * (_val(next, p.x + x + u, p.y + y + v + 1) -
						        _val(next, p.x + x + u, p.y + y + v - 1));
					g += jacobi * -err;
					hessian += jacobi * jacobi.transpose();
				}
			}

			vec2d update = hessian.ldlt().solve(g);

			if (std::isnan(update[0])) {
				std::cerr << "update is nan." << std::endl;
				status[i] = 0;
				break;
			}
			if (0 < iter && prev_loss * 1.5 < loss) { break; }
			if (update.norm() < 1e-2) { break; }

			u += update[0], v += update[1];
			prev_loss = loss;
			++iter;
		}

		if (status[i]) { pts_next[i] = p + cv::Point2f(u, v); }
	}
}

void lk_optical_flow_single2(
	const cv::Mat&                   prev,
	const cv::Mat&                   next,
	const std::vector<cv::Point2f>&  pts_prev,
	size_t                           win_sz,
	std::vector<cv::Point2f>&        pts_next,
	std::vector<uint8_t>&            status,
	bool                             use_init = false
) {
	assert(1 == win_sz % 2);

	const int    n_points = pts_prev.size();
	const int    half_w = win_sz / 2;
	const int    max_iterations = 10;
	const size_t n_pixels = win_sz * win_sz;

	pts_next.resize(n_points);
	status.resize(n_points, 1);

	for (auto i = 0; i < n_points; ++i) {
		const cv::Point2f& p = pts_prev[i];
		double u = 0., v = 0.;
		if (use_init) {
			u = pts_next[i].x - pts_prev[i].x;
			v = pts_next[i].y - pts_prev[i].y;
		}

		auto iter = 0;

		while (iter < max_iterations) {

			matxx2d a = matxx2d::Zero(n_pixels, 2);
			vecxd   b = vecxd::Zero(n_pixels);

			auto k = 0;
			for (auto x = -half_w; x <= half_w; ++x) {
				for (auto y = -half_w; y <= half_w; ++y) {
					///**
					a(k, 0) = 0.25 * (_val(next, p.x + u + x + 1, p.y + v + y) + _val(prev, p.x + x + 1, p.y + y) -
						              _val(next, p.x + u + x - 1, p.y + v + y) - _val(prev, p.x + x - 1, p.y + y));
					a(k, 1) = 0.25 * (_val(next, p.x + u + x, p.y + v + y + 1) + _val(prev, p.x + x, p.y + y + 1) -
					                  _val(next, p.x + u + x, p.y + v + y - 1) - _val(prev, p.x + x, p.y + y - 1));
					//*/
					/** @brief same as single1
					a(k, 0) = 0.5 * (_val(next, p.x + u + x + 1, p.y + v + y) -
					                 _val(next, p.x + u + x - 1, p.y + v + y));
					a(k, 1) = 0.5 * (_val(next, p.x + u + x, p.y + v + y + 1) -
						             _val(next, p.x + u + x, p.y + v + y - 1));
					*/
					/**
					a(k, 0) = 0.5 * (_val(prev, p.x + x + 1, p.y + y) -
					                 _val(prev, p.x + x - 1, p.y + y));
					a(k, 1) = 0.5 * (_val(prev, p.x + x, p.y + y + 1) -
						             _val(prev, p.x + x, p.y + y - 1));
					*/
					b[k] = _val(next, p.x + x + u, p.y + y + v) -
						   _val(prev, p.x + x, p.y + y);
					++k;
				}
			}

			vec2d update = -(a.transpose() * a).inverse() * a.transpose() * b;

			if (std::isnan(update[0])) {
				std::cerr << "update is nan. iter: " << iter << std::endl;
				status[i] = 0;
				break;
			}
			if (update.norm() < 1e-2) { break; }

			u += update[0], v += update[1];
			++iter;
		}

		if (status[i]) { pts_next[i] = p + cv::Point2f(u, v); }
	}
}

void lk_optical_flow_multi(
	const cv::Mat&                   prev,
	const cv::Mat&                   next,
	const std::vector<cv::Point2f>&  pts_prev,
	size_t                           win_sz,
	std::vector<cv::Point2f>&        pts_next,
	std::vector<uint8_t>&            status,
	size_t                           level    = 4,
	bool                             use_init = false
) {
	assert(0 < level && level < 10);

	const double pyramid_scale = 0.5;
	const double top_scale = std::pow(pyramid_scale, level - 1);
	const size_t n_points = pts_prev.size();

	/**
	 * @brief create pyramid
	 */
	std::vector<cv::Mat> pyramid_prev, pyramid_next;
	pyramid_prev.push_back(prev);
	pyramid_next.push_back(next);
	for (auto i = 1; i < level; ++i) {
		cv::Mat tmp_prev, tmp_next;
		cv::resize(pyramid_prev[i - 1], tmp_prev, cv::Size(), pyramid_scale, pyramid_scale);
		cv::resize(pyramid_next[i - 1], tmp_next, cv::Size(), pyramid_scale, pyramid_scale);
		pyramid_prev.push_back(tmp_prev);
		pyramid_next.push_back(tmp_next);
	}

	std::vector<cv::Point2f> tmp_pts_prev, tmp_pts_next;
	for (auto i = 0; i < n_points; ++i) {
		tmp_pts_prev.push_back(pts_prev[i] * top_scale);
		if (use_init) { tmp_pts_next.push_back(pts_next[i] * top_scale); }
		else /*  */   { tmp_pts_next.push_back(pts_prev[i] * top_scale); }
	}

	/**
	 * @brief perform single level LK algorithm to each layer of pyramid.
	 */
	for (auto i = level; 0 < i; --i) {

		status.clear();
		size_t current_level = i - 1;

		//lk_optical_flow_single1(
		lk_optical_flow_single2(
			pyramid_prev[current_level], pyramid_next[current_level],
			tmp_pts_prev, 21, tmp_pts_next, status, true
		);

		if (0 != current_level) {
			for (auto& each : tmp_pts_prev) {
				each /= pyramid_scale;
			}
			for (auto& each : tmp_pts_next) {
				each /= pyramid_scale;
			}
		}
	}

	pts_next.swap(tmp_pts_next);
}

const std::string seq1_path = "seq/gry01.jpg";
const std::string seq2_path = "seq/gry02.jpg";

int main(int argc, char** argv) {
//	const int n_images = 13;
//
//	for (auto i = 0; i < n_images; ++i) {
//		char tmp[16];
//		sprintf(tmp, "seq/rgb%.2d.jpg", i + 1);
//		const auto path = std::string(tmp);
//		cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
//		cv::resize(img, img, cv::Size2i(600, 600));
//		sprintf(tmp, "seq/gry%.2d.jpg", i + 1);
//		cv::imwrite(std::string(tmp), img);
//	}

	cv::Mat seq1 = cv::imread(seq1_path, cv::IMREAD_GRAYSCALE);
	cv::Mat seq2 = cv::imread(seq2_path, cv::IMREAD_GRAYSCALE);

	std::vector<cv::KeyPoint> key_points1;
	cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);
	detector->detect(seq1, key_points1);

	std::vector<cv::Point2f> pts1;
	for (auto& each : key_points1) {
		pts1.emplace_back(each.pt);
	}
	for (auto i = 0; i < pts1.size(); ++i) {
		cv::circle(seq1, pts1[i], 2, cv::Scalar_<uint8_t>(255), 2);
	}
	cv::imshow("seq1", seq1);
	cv::waitKey();
	/**
	 * @brief self-implementation of single level LK optical flow
	 *        using Gauss-Newton BA
	 */ {
	 	cv::Mat seq2_clone = seq2.clone();

		std::vector<cv::Point2f> pts2;
		std::vector<uint8_t> status;

		lk_optical_flow_single1(seq1, seq2, pts1, 21, pts2, status);

		for (auto i = 0; i < status.size(); ++i) {
			if (!status[i]) { continue; }
			cv::circle(seq2_clone, pts2[i], 2, cv::Scalar_<uint8_t>(255), 2);
			cv::line(seq2_clone, pts1[i], pts2[i], cv::Scalar_<uint8_t>(255), 1);
		}

		cv::imshow("seq2_single1", seq2_clone);
		//cv::waitKey();
	}

	/**
	 * @brief self-implementation of single level LK optical flow
	 *        using Least Square Solution A[u, v]T = -b, iteratively.
	 */ {
		cv::Mat seq2_clone = seq2.clone();

		std::vector<cv::Point2f> pts2;
		std::vector<uint8_t> status;

		lk_optical_flow_single2(seq1, seq2, pts1, 21, pts2, status);

		for (auto i = 0; i < status.size(); ++i) {
			if (!status[i]) { continue; }
			cv::circle(seq2_clone, pts2[i], 2, cv::Scalar_<uint8_t>(255), 2);
			cv::line(seq2_clone, pts1[i], pts2[i], cv::Scalar_<uint8_t>(255), 1);
		}

		cv::imshow("seq2_single2", seq2_clone);
		//cv::waitKey();
	}

	/**
	 * @brief self-implementation of multi-level LK optical flow
	 *
	 */ {
		cv::Mat seq2_clone = seq2.clone();

		std::vector<cv::Point2f> pts2;
		std::vector<uint8_t> status;

		lk_optical_flow_multi(seq1, seq2, pts1, 21, pts2, status);

		for (auto i = 0; i < status.size(); ++i) {
			if (!status[i]) { continue; }
			cv::circle(seq2_clone, pts2[i], 2, cv::Scalar_<uint8_t>(255), 2);
			cv::line(seq2_clone, pts1[i], pts2[i], cv::Scalar_<uint8_t>(255), 1);
		}

		cv::imshow("seq2_multi1", seq2_clone);
		//cv::waitKey();
	}

	/**
	 * @brief OpenCV Multi-level LK optical flow
	 *
	 */ {
		cv::Mat seq2_clone = seq2.clone();

		std::vector<cv::Point2f> pts2;
		std::vector<uchar> status;
		std::vector<float> err;

		cv::calcOpticalFlowPyrLK(seq1, seq2, pts1, pts2, status, err);

		for (auto i = 0; i < status.size(); ++i) {
			if (!status[i]) { continue; }
			cv::circle(seq2_clone, pts2[i], 2, cv::Scalar_<uint8_t>(255), 2);
			cv::line(seq2_clone, pts1[i], pts2[i], cv::Scalar_<uint8_t>(255), 1);
		}

		cv::imshow("seq2_opencv", seq2_clone);
		cv::waitKey();
	}

	return 0;
}


















