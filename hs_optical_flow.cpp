/*
 * Created by Maou Lim on 2019/12/25.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

inline cv::Mat _dx(const cv::Mat& source) {
	const cv::Mat kernel =
		(cv::Mat_<float>(2, 2) << -1.f, 1.f, -1.f, 1.f) * 0.5;
	cv::Mat res;
	cv::filter2D(source, res, CV_32F, kernel);
	return res;
}

inline cv::Mat _dy(const cv::Mat& source) {
	const cv::Mat kernel =
		(cv::Mat_<float>(2, 2) << -1.f, -1.f, 1.f, 1.f) * 0.5;
	cv::Mat res;
	cv::filter2D(source, res, CV_32F, kernel);
	return res;
}

inline cv::Mat _dt(
	const cv::Mat& prev, const cv::Mat& next
) { return next - prev; }

inline void _smooth_inplace(cv::Mat& source) {
	const cv::Mat gauss =
		(cv::Mat_<float>(3, 3) << 1.f, 2.f, 1.f, 2.f, 4.f, 2.f, 1.f, 2.f, 1.f) * 0.0625f;
	cv::Mat res;
	cv::filter2D(source, res, CV_32F, gauss);
	source = res;
}

inline cv::Mat _mean_filter(cv::Mat& source) {
	const cv::Mat kernel =
		(cv::Mat_<float>(2, 2) << 1.f, 1.f, 1.f, 1.f) * 0.25f;
	cv::Mat res;
	cv::filter2D(source, res, CV_32F, kernel);
	return res;
}

template <typename _Tp>
inline _Tp _square(_Tp a, _Tp b) {
	return a * a + b * b;
}

void hs_optical_flow(
	const cv::Mat& prev,
	const cv::Mat& next,
	cv::Mat&       flow_x,
	cv::Mat&       flow_y,
	float          smoothness,
	float          threshold,
	size_t         level = 4
) {
	assert(0 < smoothness);

	const double pyramid_scale = 0.5;
	const int    max_iterations = 100;

	std::vector<cv::Size2i> size_seq;
	size_seq.reserve(level);
	size_seq.emplace_back(prev.cols, prev.rows);

	for (auto i = 1; i < level; ++i) {
		size_seq.emplace_back(
			size_seq.back().width  * pyramid_scale,
			size_seq.back().height * pyramid_scale
		);
	}

	std::vector<cv::Mat> pyr_prev, pyr_next;
	pyr_prev.push_back(prev);
	pyr_next.push_back(next);

	for (auto i = 1; i < level; ++i) {
		cv::Mat tmp_prev, tmp_next;
		cv::resize(pyr_prev[i - 1], tmp_prev, size_seq[i]);
		cv::resize(pyr_next[i - 1], tmp_next, size_seq[i]);
		pyr_prev.push_back(tmp_prev);
		pyr_next.push_back(tmp_next);
	}

	cv::Mat u, v;

	for (auto i = level; 0 < i; --i) {

		size_t current_level = i - 1;

		if (level == i) {
			u = cv::Mat::zeros(size_seq[current_level], CV_32F);
			v = cv::Mat::zeros(size_seq[current_level], CV_32F);
		}

		const cv::Mat& p = pyr_prev[current_level];
		const cv::Mat& n = pyr_next[current_level];

		cv::Mat dx = (_dx(p) + _dx(p)) * 0.5f,
		        dy = (_dy(p) + _dy(p)) * 0.5f,
		        dt = _dt(p, n);

		auto iter = 0;
		while (iter < max_iterations) {
			cv::Mat u_bar = _mean_filter(u), v_bar = _mean_filter(v);
			// calculate u, k iteratively.

			for (auto x = 0; x < u.cols; ++x) {
				for (auto y = 0; y < u.rows; ++y) {

					float ix = dx.at<float>(y, x);
					float iy = dy.at<float>(y, x);
					float it = dt.at<float>(y, x);

					float grad2 = _square(ix, iy);
					if (threshold < grad2) { u.at<float>(y, x) = u_bar.at<float>(y, x); }
					else {
						float k = (ix * u_bar.at<float>(y, x) +
						           iy * v_bar.at<float>(y, x) +
						           it) / (smoothness + grad2);
						float delta_u = ix * k;
						float delta_v = iy * k;
						u.at<float>(y, x) = u_bar.at<float>(y, x) - delta_u;
						v.at<float>(y, x) = v_bar.at<float>(y, x) - delta_v;
					}
				}
			}
			++iter;
		}

		if (0 < current_level) {
			_smooth_inplace(u);
			_smooth_inplace(v);
			// upsample x2 LINEAR
			cv::resize(u, u, size_seq[current_level - 1], cv::INTER_LINEAR);
			cv::resize(v, v, size_seq[current_level - 1], cv::INTER_LINEAR);
		}
	}

	flow_x = u;
	flow_y = v;
}

const std::string seq1_path = "seq/LK1.png";
const std::string seq2_path = "seq/LK2.png";

float _mean(const cv::Mat& source) {
	assert(source.depth() == CV_32F);
	float res = 0.f;
	for (auto i = 0; i < source.rows; ++i) {
		for (auto j = 0; j < source.cols; ++j) {
			res += source.at<float>(i, j);
		}
	}
	return res / (source.rows * source.cols);
}

float _stddev(const cv::Mat& source, float mean) {
	assert(source.depth() == CV_32F);
	float res = 0.f;
	for (auto i = 0; i < source.rows; ++i) {
		for (auto j = 0; j < source.cols; ++j) {
			float diff = source.at<float>(i, j) - mean;
			res += diff * diff;
		}
	}
	return std::sqrt(res / (source.rows * source.cols));
}

cv::Mat _visualize(const cv::Mat& u, const cv::Mat& v) {
	cv::Mat u2, v2, i;
	cv::multiply(u, u, u2);
	cv::multiply(v, v, v2);
	cv::sqrt(u2 + v2, i);
	float mean = _mean(i);
	float stddev = _stddev(i, mean);
	return ((i - mean) / stddev + 1.f) / 2.f;
}

cv::Mat _vis_arrow(const cv::Mat& source, const cv::Mat& u, const cv::Mat& v) {
	cv::Mat u2, v2, i;
	cv::multiply(u, u, u2);
	cv::multiply(v, v, v2);
	cv::sqrt(u2 + v2, i);
	float mean = _mean(i);
	float stddev = _stddev(i, mean);
	return ((i - mean) / stddev + 1.f) / 2.f;
}

int main(int argc, char** argv) {

	cv::Mat seq1 = cv::imread(seq1_path, cv::IMREAD_GRAYSCALE);
	cv::Mat seq2 = cv::imread(seq2_path, cv::IMREAD_GRAYSCALE);

	seq1.convertTo(seq1, CV_32F, 1.f / 255);
	seq2.convertTo(seq2, CV_32F, 1.f / 255);

	cv::Mat u, v;
	hs_optical_flow(seq1, seq2, u, v, 10., 1e-2);

	cv::imshow("seq1", seq1);
	cv::imshow("i", _visualize(u, v));
	cv::waitKey();

	return 0;
}





















