#include <opencv2/opencv.hpp>
//#include <opencv2/core/types.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/imgproc/imgproc.hpp"

#define _USE_MATH_DEFINES 
#include <math.h>

using namespace cv;
using namespace std;


void generateInliers(cv::Mat &img, vector<cv::Point> &points, int count, int noise, cv::Point center, int radius);

void generateOutliers(cv::Mat &img, vector<cv::Point> &points, int count, cv::Size window_size);

void fitCircleRANSAC(cv::Mat &img, vector<cv::Point> &points, double confidence, int treshold, vector<cv::Point> &found_inliers, cv::Mat &found_circle);

void fitCircle(std::vector<cv::Point> sample_points, cv::Point &center, int &radius, bool debug);

int getIterationCount(double confidence, int inliers_count, int points_count, int sample_size);

void drawResult(cv::Mat &img, std::vector<cv::Point>inliers, cv::Mat circle);

double random0to1();


int main(int argc, char** argv)
{	
	//namedWindow("MainWindow", WINDOW_AUTOSIZE);
	int window_width = 624;
	int window_height = 768;
	int inlier_count = 300;
	int outlier_count = 500;
	int noise = 30;
	int ransac_threshold = 10;
	double ransac_confidence = 0.99;
	cv::Point circle_center = Point(300, 300);
	int circle_radius = 250;


	Mat image = Mat::zeros(window_height, window_width, CV_8UC3);
	vector<cv::Point> best_inliers;
	cv::Mat best_circle;
	std::vector<cv::Point> points;
	points.reserve(inlier_count + outlier_count);


	generateInliers(image, points, inlier_count, noise, circle_center, circle_radius);
	generateOutliers(image, points, outlier_count, cv::Size(image.cols, image.rows));
	imshow("Input", image);

	fitCircleRANSAC(image, points,ransac_confidence,ransac_threshold,best_inliers,best_circle);
	drawResult(image, best_inliers, best_circle);
	imshow("Output", image);
	waitKey();

	return 0;
}

void generateInliers(cv::Mat &img, vector<cv::Point> &points, int count, int noise, cv::Point center, int radius)
{
	double freq_angle = 2 * M_PI / count;
	
	for (int i = 0; i < count; i++)
	{
		//double directional_noise = random0to1() * noise - noise/2;
		//double x = cos(freq_angle*i)*(radius + directional_noise);
		//double y = sin(freq_angle*i)*(radius + directional_noise);

		double x = cos(freq_angle*i)*radius + random0to1() * noise;
		double y = sin(freq_angle*i)*radius + random0to1() * noise;

		cv::Point point = Point(center.x + x, center.y + y);

		points.emplace_back(point);
		cv::circle(img, point, 2, cv::Scalar(255, 255, 255), cv::FILLED);
	}
}

void generateOutliers(cv::Mat &img, vector<cv::Point> &points, int count, cv::Size window_size)
{
	for (int i = 0; i < count; i++)
	{
		int x = random0to1() * window_size.width;
		int y = random0to1() * window_size.height;

		cv::Point point = Point(x, y);

		points.emplace_back(point);
		circle(img, point, 2, cv::Scalar(255, 255, 255), cv::FILLED);
	}
}

void fitCircleRANSAC(cv::Mat &img, vector<cv::Point> &points, double confidence, int treshold, vector<cv::Point> &found_inliers, cv::Mat &found_circle)
{
	int itercount_estimation = std::numeric_limits<int>::max();
	int minimal_iterations = 0;
	found_circle.create(3, 1, CV_64F);


	//for (int iteration = 0; iteration < 1000; iteration++)
	for (int iteration = 0; iteration < itercount_estimation || iteration < minimal_iterations; iteration++)
	{
		std::vector<cv::Point> sample;
		cv::Point tmp_center;
		int tmp_radius;
		std::vector<cv::Point> tmp_inliers;

		// Get 3 random points
		for (int i = 0; i < 3; i++)
		{
			cv::Point selected_point = points[random0to1() * (points.size() - 1)];
			bool alreadySelected = false;

			for (int j = 0; j < sample.size(); j++)
			{
				if (selected_point == sample[j])
				{
					i--;
					alreadySelected = true;
					break;
				}
			}

			if (!alreadySelected)
				sample.emplace_back(selected_point);
		}
		// Determine the circle that fits the sample points
		fitCircle(sample, tmp_center, tmp_radius, false);

		// DEBUG
		//if (iteration == 1338)
		//	fitCircle(sample, tmp_center, tmp_radius,true);
		//else
		//	fitCircle(sample, tmp_center, tmp_radius,false);

		for (int i = 0; i < points.size(); i++)
		{
			int distance = abs(cv::norm(points[i] - tmp_center) - tmp_radius);
			if (distance <= treshold)
				tmp_inliers.emplace_back(points[i]);
		}

		if (tmp_inliers.size() > found_inliers.size())
		{
			found_inliers.swap(tmp_inliers);
			found_circle.at<double>(0) = tmp_center.x;
			found_circle.at<double>(1) = tmp_center.y;
			found_circle.at<double>(2) = tmp_radius;

			itercount_estimation = getIterationCount(confidence,found_inliers.size(),points.size(),3);

			cout << "Better circle found at iteration: "<<iteration << endl;
			cout << "New iteration number estimate: " << itercount_estimation << endl;
			cout << "Inliers size: " << found_inliers.size() << endl << endl;
		}
	}
}

void fitCircle(std::vector<cv::Point> sample_points, cv::Point &center, int &radius, bool debug)
{
	cv::Point2d point1 = sample_points[0];
	cv::Point2d point2 = sample_points[1];
	cv::Point2d point3 = sample_points[2];

	cv::Point2d point12 = (point2 + point1) / 2;
	cv::Point2d point13 = (point1 + point3) / 2;

	// slope of line between point 1 and 2
	double slope_12 = (point2.y - point1.y) / (point2.x - point1.x);
	double slope_13 = (point3.y - point1.y) / (point3.x - point1.x);

	// these slopes break calculations
	if (slope_12 == 0 || slope_13 == 0 || slope_12 == slope_13)
		return;

	// slope of bisector line of the line between point 1 and 2
	double slope_bisector12 = -1 / slope_12;
	double slope_bisector13 = -1 / slope_13;
	// y = m*x + b ----> b = y - m*x
	double offset_bisector12 = point12.y - slope_bisector12 * point12.x;
	double offset_bisector13 = point13.y - slope_bisector13 * point13.x;

	// y = m1*x + b1; y = m2*x + b2
	// m1*x + b1 = m2*x + b2
	double center_x = (offset_bisector13 - offset_bisector12) / (slope_bisector12 - slope_bisector13);
	double center_y = center_x * slope_bisector12 + offset_bisector12;

	center = Point(center_x, center_y);
	radius = cv::norm(center - Point(point1.x, point1.y));

	if (debug)
	{
		cout << "debug";
	}
}

// 1-(1-(inlier_count / point_count)^sample_size)^iteration_count = confidence
// rearange to: iteration_count = log(1-confidence) / log(1-(inlier_count / point_count)^sample_size)
int getIterationCount(double confidence, int inliers_count, int points_count, int sample_size)
{
	double result = log(1.0 - confidence) / log(1.0 - std::pow(static_cast<double>(inliers_count) / points_count, sample_size));
	return ceil(result);
}

void drawResult(cv::Mat &img, std::vector<cv::Point>inliers, cv::Mat circle)
{
	int best_circle_x = circle.at<double>(0);
	int best_circle_y = circle.at<double>(1);
	int best_circle_r = circle.at<double>(2);
	
	// draw circle
	cv::circle(img, cv::Point(best_circle_x, best_circle_y), best_circle_r, cv::Scalar(255, 0, 255), 1);

	// re-draw inliers with different colour
	for (int i = 0; i < inliers.size(); i++)
	{
		cv::circle(img, inliers[i], 2, cv::Scalar(0, 0, 255), cv::FILLED);
	}
}

double random0to1()
{
	return (static_cast<double> (rand()) / RAND_MAX);
}