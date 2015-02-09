/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#if FINDOBJECT_NONFREE == 0
#include <opencv2/nonfree/gpu.hpp>
#endif
#include <opencv2/gpu/gpu.hpp>
#include "Kinect2Grabber.hpp"
 #include <chrono>


class GPUFeature2D
{
public:
 	GPUFeature2D() {}
 	virtual ~GPUFeature2D() {}

 	virtual void detectKeypoints(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & mask = cv::Mat()) = 0;

 	virtual void computeDescriptors(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, cv::gpu::GpuMat & descriptors) = 0;
};

class GPUSurf : public GPUFeature2D
{
public:
 	
 	GPUSurf(double hessianThreshold, int nOctaves, int nOctaveLayers, bool extended, float keypointsRatio, bool upright) :
		surf_(hessianThreshold, nOctaves, nOctaveLayers, extended, keypointsRatio, upright)
 	{}

 	virtual ~GPUSurf() {}

 	void detectKeypoints(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & mask = cv::Mat())
  	{
		cv::gpu::GpuMat imgGpu(image);
		cv::gpu::GpuMat maskGpu(mask);
		try
		{
	  		surf_(imgGpu, maskGpu, keypoints);
		}
		catch(cv::Exception &e)
		{
	  		std::cout << "error detecting keypoints" << std::endl;
		}
  	}

  	void computeDescriptors( const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, cv::gpu::GpuMat & descriptors)
  	{
		std::vector<float> d;
		cv::gpu::GpuMat imgGpu(image);
		try
		{
		  	surf_(imgGpu, cv::gpu::GpuMat(), keypoints, descriptors, true);
		}
		catch(cv::Exception &e)
		{
			std::cout << "error computing descriptors" << std::endl;
		}
	}
private:
  	cv::gpu::SURF_GPU surf_; 
};


using namespace cv;

int main( int argc, char** argv )
{
	if( argc != 2 )
		return -1; 
	

	Mat image = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	Mat frame;

	if( !image.data ){ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	
	GPUSurf surf(400, 4, 2, true, 0.01, false);

	std::vector<KeyPoint> keypoints_image, keypoints_frame;

	surf.detectKeypoints(image, keypoints_image);

	cv::gpu::GpuMat descriptors_image, descriptors_frame;

	surf.computeDescriptors(image, keypoints_image, descriptors_image);

	std::vector<std::vector<cv::DMatch> > matches;
	cv::gpu::BruteForceMatcher_GPU<L2< float>> matcher;

	std::vector< DMatch > good_matches;

	Mat img_matches;
	namedWindow("Good Matches", WINDOW_NORMAL);
	double dist;

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	Mat H;
	std::vector<Point2f> obj_corners(4);
	std::vector<Point2f> scene_corners(4);

	obj_corners[0] = cvPoint(0,0); 
	obj_corners[1] = cvPoint( image.cols, 0 );
	obj_corners[2] = cvPoint( image.cols, image.rows ); 
	obj_corners[3] = cvPoint( 0, image.rows );

	Kinect2Grabber::Kinect2Grabber<pcl::PointXYZRGB> k2g("../calibration/rgb_calibration.yaml", "../calibration/depth_calibration.yaml", "../calibration/pose_calibration.yaml");
	int counter = 0;

	while(true){
		using namespace std::chrono;

		auto tnow = high_resolution_clock::now();
		
		std::cout << "getting frame " << std::endl;
		Kinect2Grabber::CvFrame<pcl::PointXYZRGB> kinect_frame (k2g);

		frame = kinect_frame.data_;

		cvtColor(frame, frame, CV_BGR2GRAY);

		std::cout << "detecting surf" << std::endl;
		surf.detectKeypoints(frame, keypoints_frame);

		std::cout << "computing descriptors" << std::endl;
		surf.computeDescriptors(frame, keypoints_frame, descriptors_frame);

		std::cout << "matching descriptors" << std::endl;
		matcher.knnMatch( descriptors_image, descriptors_frame, matches, 2 );

		for(int k = 0; k < min(descriptors_image.rows-1,(int) matches.size()); k++)   
			{  
			if((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int) matches[k].size()<=2 && (int) matches[k].size()>0))  
			{  
				good_matches.push_back(matches[k][0]);  
			}  
		}  

		std::cout << "drawing matches" << std::endl;
		drawMatches( image, keypoints_image, frame, keypoints_frame, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				     std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		if (good_matches.size() >= 4){
			for( int i = 0; i < good_matches.size(); i++ ){
				obj.push_back( keypoints_image[ good_matches[i].queryIdx ].pt );
				scene.push_back( keypoints_frame[ good_matches[i].trainIdx ].pt );
			}

			H = findHomography( obj, scene, CV_RANSAC );

			if(H.rows == 3 && H.cols == 3){
				perspectiveTransform( obj_corners, scene_corners, H);
				line( img_matches, scene_corners[0] + Point2f( image.cols, 0), scene_corners[1] + Point2f( image.cols, 0), Scalar(0, 255, 0), 4 );
				line( img_matches, scene_corners[1] + Point2f( image.cols, 0), scene_corners[2] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
				line( img_matches, scene_corners[2] + Point2f( image.cols, 0), scene_corners[3] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
				line( img_matches, scene_corners[3] + Point2f( image.cols, 0), scene_corners[0] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
			}
		}

		//-- Show detected matches
		std::cout << "showing image" << std::endl;
		imshow( "Good Matches", img_matches );
		matches.clear();
		good_matches.clear();
		obj.clear();
		scene.clear();
		scene_corners.resize(4);
		char key = waitKey(30);
		if(key == 27) 
			break;

		auto tpost = high_resolution_clock::now();
		std::cout << "time " << duration_cast<duration<double>>(tpost-tnow).count()*1000 << std::endl;

	}
	return 0;
}
