/*
 * Created by Maou Lim on 2019/12/16.
 */

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {

//	for (auto i = 0; i < 16; ++i) {
//		char tmp[8];
//		sprintf(tmp, "calib/%.2d.jpg", i);
//		auto path = std::string(tmp);
//		cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
//		cv::resize(img, img, cv::Size2i(600, 600));
//		cv::imwrite(path, img);
//	}

	typedef std::vector<cv::Point3f> point3f_set;
	typedef std::vector<cv::Point2f> point2f_set;

	const cv::Size2i image_sz(600, 600);
	const cv::Size2i pattern(4, 6);
	point3f_set corners_w_templ;
	corners_w_templ.reserve(pattern.area());

	std::vector<point3f_set> corners_w_seq;
	std::vector<point2f_set> corners_s_seq;

	for (auto y = 0; y < pattern.height; ++y) {
		for (auto x = 0; x < pattern.width; ++x) {
			corners_w_templ.emplace_back(x, y, 0.0f);
		}
	}

	cv::Mat image; //  保存标定板的图像
	int n_images = 16, count = 0;

	corners_w_seq.reserve(n_images);
	corners_s_seq.reserve(n_images);

	for (auto i = 0; i < n_images; ++i) {

		char tmp[8];
		sprintf(tmp, "calib/%.2d.jpg", i);
		auto path = std::string(tmp);

		image = cv::imread(path, cv::IMREAD_GRAYSCALE);

		std::vector<cv::Point2f> corners_s;
		bool found = cv::findChessboardCorners(image, pattern, corners_s);

		if (found && pattern.area() == corners_s.size()) {

			cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.1);
			cv::cornerSubPix(image, corners_s, cv::Size2i(5, 5), cv::Size2i(-1, -1), criteria);

			corners_w_seq.emplace_back(corners_w_templ);
			corners_s_seq.emplace_back(std::move(corners_s));

			std::cout << "good frame: " << path << std::endl;

			++count;
		}
	}

	cv::Mat intrinsic, dist_coeffs;
	std::vector<cv::Mat> rvecs, tvecs;
	auto err = cv::calibrateCamera(corners_w_seq, corners_s_seq, image_sz, intrinsic, dist_coeffs, rvecs, tvecs);

	std::cout << "count good: " << count << std::endl;
	std::cout << "error of calib: " << err << std::endl;

	std::cout << "intrinsic: " << std::endl << intrinsic << std::endl;
	std::cout << "dis_coeffs: " << std::endl << dist_coeffs << std::endl;

	point3f_set points_w;
	point2f_set points_s;

	points_w.emplace_back(-1.f, 0.f, 0.f);
	points_w.emplace_back(-1.f, 1.f, 0.f);
	points_w.emplace_back(-1.f, 2.f, 0.f);
	points_w.emplace_back(-1.f, 3.f, 0.f);

	cv::Mat jpg_00 = cv::imread("calib/00.jpg", cv::IMREAD_COLOR);

	cv::projectPoints(points_w, rvecs[0], tvecs[0], intrinsic, dist_coeffs, points_s);
	for (auto& each : points_s) {
		cv::circle(jpg_00, each, 3, cv::Scalar_<int>(255, 0, 0), 2);
	}

	cv::imshow("projection", jpg_00);
	cv::waitKey();

	return 0;
}