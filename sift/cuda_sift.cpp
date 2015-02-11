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
#include "sift.h"



int main( int argc, char** argv )
{
	if( argc != 2 )
		return -1; 
	

	cv::Mat image = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	cv::Mat frame;

	if( !image.data ){ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}

	std::vector<cv::KeyPoint>  keypoints_image;
	std::vector<float>  descriptors_image;
	std::vector<cv::KeyPoint>  keypoints_frame;
	std::vector<float>  descriptors_frame;
	SiftGPUWrapper * sift = SiftGPUWrapper::getInstance();

	sift->detect(image, keypoints_image, descriptors_image );
	std::cout << "got sift descriptors for image" << std::endl;

	std::vector< cv::DMatch > good_matches;

	cv::Mat img_matches;
	namedWindow("Good Matches", cv::WINDOW_NORMAL);
	double dist;

	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	cv::Mat H;
	std::vector<cv::Point2f> obj_corners(4);
	std::vector<cv::Point2f> scene_corners(4);

	obj_corners[0] = cvPoint(0,0); 
	obj_corners[1] = cvPoint( image.cols, 0 );
	obj_corners[2] = cvPoint( image.cols, image.rows ); 
	obj_corners[3] = cvPoint( 0, image.rows );

	Kinect2Grabber::Kinect2Grabber<pcl::PointXYZRGB> k2g("../../calibration/rgb_calibration.yaml", "../../calibration/depth_calibration.yaml", "../../calibration/pose_calibration.yaml");
	int counter = 0;

	while(true){
		using namespace std::chrono;

		auto tnow = high_resolution_clock::now();
		Kinect2Grabber::CvFrame<pcl::PointXYZRGB> kinect_frame (k2g);

		cv::Mat sift_frame = kinect_frame.data_;
		
		sift->detect(sift_frame, keypoints_frame, descriptors_frame );
		sift->match(descriptors_image, 128, descriptors_frame, 128, &good_matches);

		auto tpost = high_resolution_clock::now();
		std::cout << "time " << duration_cast<duration<double>>(tpost-tnow).count()*1000 << std::endl;
		/*
		for(int k = 0; k < cv::min(descriptors_image.rows-1,(int) matches.size()); k++)   
		{  
			if((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int) matches[k].size()<=2 && (int) matches[k].size()>0))  
			{  
				good_matches.push_back(matches[k][0]);  
			}  
		}  */
		std::cout << "keypoints image " << keypoints_image.size() << " keypoints frame " << keypoints_frame.size() << " matches " << good_matches.size() << std::endl;
		cv::drawMatches( image, keypoints_image, sift_frame, keypoints_frame, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
				     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		if (good_matches.size() >= 10){
			for( int i = 0; i < good_matches.size(); i++ ){
				obj.push_back( keypoints_image[ good_matches[i].queryIdx ].pt );
				scene.push_back( keypoints_frame[ good_matches[i].trainIdx ].pt );
			}

			H = cv::findHomography( obj, scene, CV_RANSAC );

			if(H.rows == 3 && H.cols == 3){
				cv::perspectiveTransform( obj_corners, scene_corners, H);
				cv::line( img_matches, scene_corners[0] + cv::Point2f( image.cols, 0), scene_corners[1] + cv::Point2f( image.cols, 0), cv::Scalar(0, 255, 0), 4 );
				cv::line( img_matches, scene_corners[1] + cv::Point2f( image.cols, 0), scene_corners[2] + cv::Point2f( image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
				cv::line( img_matches, scene_corners[2] + cv::Point2f( image.cols, 0), scene_corners[3] + cv::Point2f( image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
				cv::line( img_matches, scene_corners[3] + cv::Point2f( image.cols, 0), scene_corners[0] + cv::Point2f( image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
			}
		}

		cv::imshow( "Good Matches", img_matches );
		good_matches.clear();
		obj.clear();
		scene.clear();
		scene_corners.resize(4);
		char key = cv::waitKey(30);
		if(key == 27) 
			break;
	}
	return 0;
}
