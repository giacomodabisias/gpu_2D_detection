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


class GPUSurf
{
public:
 	
 	GPUSurf(double hessianThreshold, int nOctaves, int nOctaveLayers, bool extended, float keypointsRatio, bool upright) :
		surf_(hessianThreshold, nOctaves, nOctaveLayers, extended, keypointsRatio, upright)
 	{}

 	virtual ~GPUSurf() {}

 	void detectKeypoints(const cv::gpu::GpuMat & image, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & mask = cv::Mat())
  	{
		cv::gpu::GpuMat maskGpu(mask);
		try
		{
	  		surf_(image, maskGpu, keypoints);
		}
		catch(cv::Exception &e)
		{
	  		std::cout << "error detecting keypoints" << std::endl;
		}
  	}

  	void computeDescriptors( const cv::gpu::GpuMat & image, std::vector<cv::KeyPoint> & keypoints, cv::gpu::GpuMat & descriptors)
  	{
		std::vector<float> d;
		try
		{
		  	surf_(image, cv::gpu::GpuMat(), keypoints, descriptors, true);
		}
		catch(cv::Exception &e)
		{
			std::cout << "error computing descriptors" << std::endl;
		}
	}
private:
  	cv::gpu::SURF_GPU surf_; 
};

class Grabber {
public:

	Grabber(int id): id_(id){
		if(id_ == 1)
			k2g_ = new Kinect2Grabber::Kinect2Grabber<pcl::PointXYZRGB>("../../calibration/rgb_calibration.yaml", "../../calibration/depth_calibration.yaml", "../../calibration/pose_calibration.yaml");
		else
			capture_ = new cv::VideoCapture(0);

	}


void getFrame(cv::Mat & frame){
	if(id_ == 1){
		Kinect2Grabber::CvFrame<pcl::PointXYZRGB> kinect_frame (*k2g_);
		frame = kinect_frame.data_;
	} else{
		cv::Mat tmp;
        capture_->read(tmp);
        cv::flip(tmp, frame, 1); 
	}
}

private:
	Kinect2Grabber::Kinect2Grabber<pcl::PointXYZRGB> * k2g_;
	cv::VideoCapture * capture_;
	int id_;
};



int main( int argc, char** argv )
{
	int id = 0;

	if( argc < 2){
		std::cout << "use: ./program image_path_file 1 for kinect2" << std::endl;
		return -1; 
	}
	if(argc == 3){
		if(atoi(argv[2]) == 1)
			id = 1;
	}
	
	cv::Mat frame;
	std::vector<cv::Mat> images;
	std::vector<cv::gpu::GpuMat> gpu_images;
  	std::ifstream infile(argv[1]);
  
 	std::string path;
 	int image_number = 0;
 	//open the N images
  	while (infile >> path){
  		std::cout << "loading " << path << std::endl;
	    cv::Mat image = cv::imread( path, CV_LOAD_IMAGE_GRAYSCALE );
	    if(! image.data )                             
	    {
	        cout <<  "Could not open or find image " << path <<  std::endl ;
	        return -1;
	    }
	    images.push_back(image);
	    cv::gpu::GpuMat gpu_image(image);
    	gpu_images.push_back(gpu_image);
    	//namedWindow("Good Matches"+image_number, cv::WINDOW_NORMAL);
    	image_number++;
	}

	namedWindow("Good Matches", cv::WINDOW_NORMAL);

	GPUSurf detector(300, 4, 2, true, 0.01, false); //400

	std::vector<std::vector<cv::KeyPoint>> keypoints_images (image_number);
	std::vector<cv::KeyPoint> keypoints_frame;

	cv::gpu::GpuMat descriptors_frame;
	std::vector<cv::gpu::GpuMat> descriptors_images (image_number);

	for(int i = 0; i < image_number; ++i){
		detector.detectKeypoints(gpu_images[i], keypoints_images[i]);
		detector.computeDescriptors(gpu_images[i], keypoints_images[i], descriptors_images[i]);
	}

	std::vector<std::vector<cv::DMatch> > matches;
	cv::gpu::BruteForceMatcher_GPU<cv::L2< float>> matcher;
	
	std::vector<std::vector<cv::DMatch>> good_matches_vector (image_number);

	std::vector<cv::Mat> img_matches_vector (image_number);

	double dist;

	Grabber grabber(id);

	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	cv::Mat H;
	std::vector<std::vector<cv::Point2f>> obj_corners_vector (image_number);
	std::vector<cv::Point2f> scene_corners(4);

	for(int i = 0; i < image_number; ++i){
		obj_corners_vector[i].resize(4);
		obj_corners_vector[i][0] = cvPoint(0,0); 
		obj_corners_vector[i][1] = cvPoint( images[i].cols, 0 );
		obj_corners_vector[i][2] = cvPoint( images[i].cols, images[i].rows ); 
		obj_corners_vector[i][3] = cvPoint( 0, images[i].rows );
	}
	//Kinect2Grabber::Kinect2Grabber<pcl::PointXYZRGB> k2g ("../../calibration/rgb_calibration.yaml", "../../calibration/depth_calibration.yaml", "../../calibration/pose_calibration.yaml");

	int counter = 0;
	cv::Mat frame_;
	while(true){
		using namespace std::chrono;

		auto tnow = high_resolution_clock::now();
		grabber.getFrame(frame_);
		//Kinect2Grabber::CvFrame<pcl::PointXYZRGB> kinect_frame (k2g);
		//cv::Mat frame_ = kinect_frame.data_;
		cv::Mat frame_tmp;
		cv::cvtColor(frame_, frame_tmp, CV_BGR2GRAY);
		cv::gpu::GpuMat frame(frame_tmp);
		/* slower 0.0
		cv::gpu::GpuMat frame_in(frame_);
		cv::gpu::GpuMat frame;
		cv::gpu::cvtColor(frame_in, frame, CV_BGR2GRAY);
		*/

		detector.detectKeypoints(frame, keypoints_frame);
		detector.computeDescriptors(frame, keypoints_frame, descriptors_frame);

		for(int i = 0; i < image_number; ++i){
			matcher.knnMatch( descriptors_images[i], descriptors_frame, matches, 2 );
			for(int k = 0; k < cv::min(descriptors_images[i].rows-1,(int) matches.size()); ++k)   
			{  	
				if((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int) matches[k].size()<=2 && (int) matches[k].size()>0))  
				{  
					good_matches_vector[i].push_back(matches[k][0]);  
				}  
			}
			matches.clear();  
		}
		
		auto tpost = high_resolution_clock::now();
		std::cout << "time for computing " << duration_cast<duration<double>>(tpost-tnow).count()*1000 << std::endl;
		
		for(int i = 0; i < image_number; ++i){
			//cv::drawMatches( images[i], keypoints_images[i], frame_, keypoints_frame, good_matches_vector[i], img_matches_vector[i], cv::Scalar::all(-1), cv::Scalar::all(-1),
				    // std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			if (good_matches_vector[i].size() >= 10){
				for( int j = 0; j < good_matches_vector[i].size(); ++j ){
					obj.push_back( keypoints_images[i][ good_matches_vector[i][j].queryIdx ].pt );
					scene.push_back( keypoints_frame[ good_matches_vector[i][j].trainIdx ].pt );
				}
				/*
				H = cv::findHomography( obj, scene, CV_RANSAC );
				if(H.rows == 3 && H.cols == 3){
					cv::perspectiveTransform( obj_corners_vector[i], scene_corners, H);
					cv::line( img_matches_vector[i], scene_corners[0] + cv::Point2f( images[i].cols, 0), scene_corners[1] + cv::Point2f( images[i].cols, 0), cv::Scalar(0, 255, 0), 4 );
					cv::line( img_matches_vector[i], scene_corners[1] + cv::Point2f( images[i].cols, 0), scene_corners[2] + cv::Point2f( images[i].cols, 0), cv::Scalar( 0, 255, 0), 4 );
					cv::line( img_matches_vector[i], scene_corners[2] + cv::Point2f( images[i].cols, 0), scene_corners[3] + cv::Point2f( images[i].cols, 0), cv::Scalar( 0, 255, 0), 4 );
					cv::line( img_matches_vector[i], scene_corners[3] + cv::Point2f( images[i].cols, 0), scene_corners[0] + cv::Point2f( images[i].cols, 0), cv::Scalar( 0, 255, 0), 4 );
				}*/

				H = cv::findHomography( obj, scene, CV_RANSAC );
				if(H.rows == 3 && H.cols == 3){
					cv::perspectiveTransform( obj_corners_vector[i], scene_corners, H);
					if(cv::contourArea(scene_corners) > 1000){ // avoid wrong results
						cv::line(frame_, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4);
						cv::line(frame_, scene_corners[1], scene_corners[2], cv::Scalar(0, 255, 0), 4);
						cv::line(frame_, scene_corners[2], scene_corners[3], cv::Scalar(0, 255, 0), 4);
						cv::line(frame_, scene_corners[3], scene_corners[0], cv::Scalar(0, 255, 0), 4);
					}
				}
			}
			obj.clear();
			scene.clear();
			//scene_corners.resize(4);
			good_matches_vector[i].clear();

		}
		auto tpost2 = high_resolution_clock::now();
		std::cout << "time for drawing " << duration_cast<duration<double>>(tpost2-tpost).count()*1000 << std::endl;
		cv::imshow( "Good Matches", frame_ );
		//img_matches_vector.clear();
		//img_matches_vector.resize(image_number);
		char key = cv::waitKey(1);
		if(key == 27) 
			break;


	}
	return 0;
}
