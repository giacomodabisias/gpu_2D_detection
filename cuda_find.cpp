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
#include "SiftGPU/SiftGPU.h"

class GPUFeature2D
{
public:
 	GPUFeature2D() {}
 	virtual ~GPUFeature2D() {}

 	virtual void detectKeypoints(const cv::gpu::GpuMat & image, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & mask = cv::Mat()) = 0;

 	virtual void computeDescriptors(const cv::gpu::GpuMat & image, std::vector<cv::KeyPoint> & keypoints, cv::gpu::GpuMat & descriptors) = 0;
};

class GPUSurf : public GPUFeature2D
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


class GPUSift 
{
public:
 	
 	GPUSift(): matcher_(4096)
 	{
 		sift_.CreateContextGL();
 		matcher_.SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
 		matcher_.VerifyContextGL(); 
 	}

 	virtual ~GPUSift() {}

 	void detectKeypointsAndcomputeDescriptors(const cv::Mat & image, std::vector<SiftGPU::SiftKeypoint> & keypoints, std::vector<float> & descriptors)
  	{	
  		//run SIFT on a new image given the pixel data and format/type;
		//gl_format (e.g. GL_LUMINANCE, GL_RGB) is the format of the pixel data
		//gl_type (e.g. GL_UNSIGNED_BYTE, GL_FLOAT) is the data type of the pixel data;
		//Check glTexImage2D(...format, type,...) for the accepted values
		//Using image data of GL_LUMINANCE + GL_UNSIGNED_BYTE can minimize transfer time
		//SIFTGPU_EXPORT virtual int  RunSIFT(int width, int height,	const void * data,  unsigned int gl_format, unsigned int gl_type);
  		sift_.RunSIFT(image.cols, image.rows, image.data, GL_LUMINANCE, GL_FLOAT);
  		keypoints.resize(sift_.GetFeatureNum());
  		descriptors.resize(sift_.GetFeatureNum());
		sift_.GetFeatureVector(&keypoints[0], &descriptors[0]);
  	}

  	void match(std::vector<float> & image_descriptors, std::vector<float> & frame_descriptors, std::vector< cv::DMatch > good_matches)
  	{
  		int size = image_descriptors.size();
  		matcher_.SetDescriptors(0, size, &image_descriptors[0]);
    	matcher_.SetDescriptors(1, frame_descriptors.size(), &frame_descriptors[0]); 
	    int (*match_buf)[2] = new int[size][2];
	    int num_match = matcher_.GetSiftMatch(size, match_buf);
	    std::cout << num_match << " sift matches were found;\n";

	    for(int i  = 0; i < num_match; ++i)
   		{
        	good_matches.push_back(cv::DMatch(match_buf[i][0], match_buf[i][1], 0.0));
   		}
  	}


private:
  	SiftGPU sift_;
  	SiftMatchGPU matcher_;
};


class GPUOrb : public GPUFeature2D
{
public:
	GPUOrb(int nFeatures, float scaleFactor, int nLevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize, int fastThreshold, bool fastNonmaxSupression ) :
		orb_(nFeatures, scaleFactor, nLevels, edgeThreshold , firstLevel, WTA_K, scoreType, patchSize)
	{
		orb_.setFastParams(fastThreshold, fastNonmaxSupression);
	}

	virtual ~GPUOrb() {}

	void detectKeypoints(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & mask = cv::Mat())
    {
    	cv::gpu::GpuMat imgGpu(image);
    	cv::gpu::GpuMat maskGpu(mask);
    	try
    	{
    		orb_(imgGpu, maskGpu, keypoints);
    	}
    	catch(cv::Exception &e)
		{
	  		std::cout << "error detecting keypoints" << std::endl;
		}
    }

    void computeDescriptors( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::gpu::GpuMat& descriptors)
	{
		std::vector<float> d;
		cv::gpu::GpuMat imgGpu(image);
		try
		{
			orb_(imgGpu, cv::gpu::GpuMat(), keypoints, descriptors); // No option to use provided keypoints!?
		}
		catch(cv::Exception &e)
		{
			std::cout << "error computing descriptors" << std::endl;
		}
	}
private:
    cv::gpu::ORB_GPU orb_;
};



int main( int argc, char** argv )
{
	if( argc != 2 )
		return -1; 
	

	cv::Mat image_ = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	cv::gpu::GpuMat image(image_);
	cv::Mat frame;

	if( !image.data ){ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}

	GPUSurf surf(400, 4, 2, true, 0.01, false);
	//GPUOrb surf(500, 1.2, 8, 31, 0, 2, 0, 31, 6, false);


	bool SIFT = false;

	//std::vector<SiftGPU::SiftKeypoint>  keypoints_sift;
	//std::vector<float>  descriptors_sift;
	//GPUSift sift;
	//sift.detectKeypointsAndcomputeDescriptors(image, keypoints_sift, descriptors_sift );


	std::vector<cv::KeyPoint> keypoints_image, keypoints_frame;
	//vector<SiftGPU::SiftKeypoint> keypoints_image, keypoints_frame;

	surf.detectKeypoints(image, keypoints_image);

	cv::gpu::GpuMat descriptors_image, descriptors_frame;

	surf.computeDescriptors(image, keypoints_image, descriptors_image);

	std::vector<std::vector<cv::DMatch> > matches;
	cv::gpu::BruteForceMatcher_GPU<cv::L2< float>> matcher;
	//cv::gpu::BruteForceMatcher_GPU<cv::Hamming> matcher;
	


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

	Kinect2Grabber::Kinect2Grabber<pcl::PointXYZRGB> k2g("../calibration/rgb_calibration.yaml", "../calibration/depth_calibration.yaml", "../calibration/pose_calibration.yaml");
	int counter = 0;

	while(true){
		using namespace std::chrono;

		auto tnow = high_resolution_clock::now();
		Kinect2Grabber::CvFrame<pcl::PointXYZRGB> kinect_frame (k2g);

		cv::Mat frame_ = kinect_frame.data_;
		cv::gpu::GpuMat frame_in(frame_);
		cv::gpu::GpuMat frame;
		cv::gpu::cvtColor(frame_in, frame, CV_BGR2GRAY);

		surf.detectKeypoints(frame, keypoints_frame);

		surf.computeDescriptors(frame, keypoints_frame, descriptors_frame);

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
		std::cout << " found " << good_matches.size() << " matches" << std::endl;

		std::cout << "drawing matches" << std::endl;
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

		//-- Show detected matches
		std::cout << "showing image" << std::endl;
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
