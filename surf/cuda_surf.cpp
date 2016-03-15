/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#if FINDOBJECT_NONFREE == 0
#include <opencv2/nonfree/gpu.hpp>
#endif
#include <opencv2/gpu/gpu.hpp>
#include <chrono>
#include <fstream>

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

	Grabber(const unsigned int width , const unsigned int height){

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


int main( int argc, char** argv )
{
	if( argc < 4){
		std::cout << "use: ./program image_path_file width height" << std::endl;
		return -1; 
	}

	cv::Mat frame;
	std::vector<cv::Mat> images;
	std::vector<cv::gpu::GpuMat> gpu_images;
  	std::ifstream infile(argv[1]);
  	const unsigned int width =  atoi(argv[2]);
  	const unsigned int height = atoi(argv[3]);
  
 	std::string path;
 	int image_number = 0;
 	//open the N images
 	if(infile){
	  	while (infile >> path){
	  		std::cout << "loading " << path << std::endl;
		    cv::Mat image = cv::imread( path, CV_LOAD_IMAGE_GRAYSCALE );
		    if(!image.data )                             
		    {
		        std::cout <<  "Could not open or find image " << path <<  std::endl ;
		        return -1;
		    }
		    images.push_back(image);
		    cv::gpu::GpuMat gpu_image(image);
	    	gpu_images.push_back(gpu_image);
	    	//namedWindow("Good Matches"+image_number, cv::WINDOW_NORMAL);
	    	image_number++;
		}
 	}else{
 		std::cout << "ERROR : could not open " << argv[1] << std::endl;
 	}
  	
	namedWindow("Good Matches", cv::WINDOW_NORMAL);

	GPUSurf detector(400, 4, 2, true, 0.01, false); 

	std::vector<std::vector<cv::KeyPoint>> keypoints_images(image_number);
	std::vector<cv::KeyPoint> keypoints_frame;

	cv::gpu::GpuMat descriptors_frame;
	std::vector<cv::gpu::GpuMat> descriptors_images(image_number);

	for(size_t i = 0; i < image_number; ++i){
		detector.detectKeypoints(gpu_images[i], keypoints_images[i]);
		detector.computeDescriptors(gpu_images[i], keypoints_images[i], descriptors_images[i]);
	}

	std::vector<std::vector<cv::DMatch>> matches;
	cv::gpu::BruteForceMatcher_GPU<cv::L2< float>> matcher;
	
	std::vector<std::vector<cv::DMatch>> good_matches_vector(image_number);
	std::vector<cv::Mat> img_matches_vector(image_number);

	Grabber grabber(width, height);

	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	cv::Mat homography;
	std::vector<std::vector<cv::Point2f>> obj_corners_vector (image_number);
	std::vector<cv::Point2f> scene_corners(4);

	for(int i = 0; i < image_number; ++i){
		obj_corners_vector[i].resize(4);
		obj_corners_vector[i][0] = cvPoint(0,0); 
		obj_corners_vector[i][1] = cvPoint( images[i].cols, 0 );
		obj_corners_vector[i][2] = cvPoint( images[i].cols, images[i].rows ); 
		obj_corners_vector[i][3] = cvPoint( 0, images[i].rows );
	}

	int counter = 0;
	cv::Mat frame_;
	while(true){
		using namespace std::chrono;

		auto tnow = high_resolution_clock::now();
		grabber.getFrame(frame_);
		cv::Mat frame_tmp;
		cv::cvtColor(frame_, frame_tmp, CV_BGR2GRAY);
		cv::gpu::GpuMat frame(frame_tmp);
		/* gup conversion
		cv::gpu::GpuMat frame_in(frame_);
		cv::gpu::GpuMat frame;
		cv::gpu::cvtColor(frame_in, frame, CV_BGR2GRAY);
		*/
		std::cout << "time for getting frame " << duration_cast<duration<double>>( high_resolution_clock::now()-tnow).count()*1000 << std::endl;
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
		
		std::cout << "time for computing " << duration_cast<duration<double>>( high_resolution_clock::now()-tnow).count()*1000 << std::endl;
		auto tnow2 = high_resolution_clock::now();
		for(int i = 0; i < image_number; ++i){
			if (good_matches_vector[i].size() >= 10){
				for( int j = 0; j < good_matches_vector[i].size(); ++j ){
					obj.push_back( keypoints_images[i][ good_matches_vector[i][j].queryIdx ].pt );
					scene.push_back( keypoints_frame[ good_matches_vector[i][j].trainIdx ].pt );
				}

				homography = cv::findHomography( obj, scene, CV_RANSAC );
				if(homography.rows == 3 && homography.cols == 3){
					cv::perspectiveTransform( obj_corners_vector[i], scene_corners, homography);
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
			good_matches_vector[i].clear();

		}
		std::cout << "time for drawing " << duration_cast<duration<double>>( high_resolution_clock::now()-tnow2).count()*1000 << std::endl;
		cv::imshow( "Good Matches", frame_ );
		char key = cv::waitKey(30);
		if(key == 27) 
			break;
	}
	return 0;
}
