/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#if FINDOBJECT_NONFREE == 0
#include <opencv2/nonfree/gpu.hpp>
#endif
#include <opencv2/gpu/gpu.hpp>
#include <chrono>


class GPUOrb 
{
public:
	GPUOrb(int nFeatures, float scaleFactor, int nLevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize, int fastThreshold, bool fastNonmaxSupression ) :
		orb_(nFeatures, scaleFactor, nLevels, edgeThreshold , firstLevel, WTA_K, scoreType, patchSize)
	{
		orb_.setFastParams(fastThreshold, fastNonmaxSupression);
	}

	virtual ~GPUOrb() {}

	void detectKeypoints(const cv::gpu::GpuMat & image, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & mask = cv::Mat())
    {
    	cv::gpu::GpuMat maskGpu(mask);
    	try
    	{
    		orb_(image, maskGpu, keypoints);
    	}
    	catch(cv::Exception &e)
		{
	  		std::cout << "error detecting keypoints" << std::endl;
		}
    }

    void computeDescriptors( const cv::gpu::GpuMat & image, std::vector<cv::KeyPoint>& keypoints, cv::gpu::GpuMat& descriptors)
	{
		try
		{
			orb_(image, cv::gpu::GpuMat(), keypoints, descriptors); 
		}
		catch(cv::Exception &e)
		{
			std::cout << "error computing descriptors" << std::endl;
		}
	}
private:
    cv::gpu::ORB_GPU orb_;
};

class Grabber {
public:

	Grabber(const unsigned int width, const unsigned int height){

		capture_ = new cv::VideoCapture(0);
		capture_->set(CV_CAP_PROP_FRAME_WIDTH, width); 
        capture_->set(CV_CAP_PROP_FRAME_HEIGHT, height); 
	}

	void getFrame(cv::Mat & frame){
		cv::Mat tmp;
       	*capture_ >> tmp;
        cv::flip(tmp, frame, 1); 
	}

private:
	cv::VideoCapture * capture_; 
};

int main(int argc, char ** argv)
{
	if( argc < 4){
		std::cout << "use: ./program image_path_file width height" << std::endl;
		return -1; 
	}
	const unsigned int width = atoi(argv[2]);
	const unsigned int height = atoi(argv[3]);


	cv::Mat image_ = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	cv::gpu::GpuMat image(image_);
	cv::Mat frame;

	if(!image.data ){ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}

	GPUOrb detector(2000, 1.2, 1, 31, 0, 2, 0, 31, 6, false);

	std::vector<cv::KeyPoint> keypoints_image, keypoints_frame;

	detector.detectKeypoints(image, keypoints_image);

	cv::gpu::GpuMat descriptors_image, descriptors_frame;

	detector.computeDescriptors(image, keypoints_image, descriptors_image);

	std::vector<std::vector<cv::DMatch> > matches;

	cv::gpu::BruteForceMatcher_GPU<cv::Hamming> matcher;
	
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

	Grabber grabber(width, height);
	cv::Mat frame_;
	while(true){
		using namespace std::chrono;

		auto tnow = high_resolution_clock::now();
		grabber.getFrame(frame_);

		cv::gpu::GpuMat frame_in(frame_);
		cv::gpu::GpuMat frame;
		cv::gpu::cvtColor(frame_in, frame, CV_BGR2GRAY);

		detector.detectKeypoints(frame, keypoints_frame);
		detector.computeDescriptors(frame, keypoints_frame, descriptors_frame);
		
		matcher.knnMatch( descriptors_image, descriptors_frame, matches, 2 );

		auto tpost = high_resolution_clock::now();
		std::cout << "time " << duration_cast<duration<double>>(tpost-tnow).count()*1000 << std::endl;
		
		for(int k = 0; k < cv::min(descriptors_image.rows-1,(int) matches.size()); k++)   
		{  
			if((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int) matches[k].size()<=2 && (int) matches[k].size()>0))  
			{  
				good_matches.push_back(matches[k][0]);  
			}  
		}  

		cv::drawMatches( image_, keypoints_image, frame_, keypoints_frame, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
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
		matches.clear();
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
