//end point http://10.100.37.203:8080/WebSocketSample/wsocket


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
#include <thread>
#include <json/json.h>
#include "data_writer.h"
#include <opencv2/gpu/gpu.hpp>
#include "Kinect2Grabber.hpp"
#include <chrono>


boost::asio::io_service io;
struct timespec start;

void aliver(const boost::system::error_code& /*e*/)
{}

void asiothreadfx()
{
    boost::asio::deadline_timer t(io, boost::posix_time::seconds(100000));
    t.async_wait(aliver);
    io.run();
}



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

int main( int argc, char** argv )
{
	clock_gettime(CLOCK_MONOTONIC, &start);
	if( argc != 3 )
		return -1; 
	
	cv::Mat image_ = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	cv::gpu::GpuMat image(image_);
	cv::Mat frame;

	srand (time(NULL));
	DataWriter * dataWriter = new DataWriter(argv[2]); 
	double elapsed;
	struct timespec finish;
    int session = rand();
    std::thread ws_writer(asiothreadfx);


	if( !image.data ){ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}

	GPUSurf detector(400, 4, 2, true, 0.01, false);

	std::vector<cv::KeyPoint> keypoints_image, keypoints_frame;

	detector.detectKeypoints(image, keypoints_image);

	cv::gpu::GpuMat descriptors_image, descriptors_frame;

	detector.computeDescriptors(image, keypoints_image, descriptors_image);

	std::vector<std::vector<cv::DMatch> > matches;
	cv::gpu::BruteForceMatcher_GPU<cv::L2< float>> matcher;
	
	std::vector< cv::DMatch > good_matches;

	cv::Mat img_matches;
	namedWindow("Good Matches 2", cv::WINDOW_NORMAL);
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

		cv::Mat frame_ = kinect_frame.data_;
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

		if (good_matches.size() >= 15){
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
				clock_gettime(CLOCK_MONOTONIC, &finish);
	            elapsed = (finish.tv_sec - start.tv_sec);
	            elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	            Json::Value root;
	            root["obj"]["type"] = "object";
	            root["obj"]["session"] = session;
	            root["obj"]["id"] = 1;   //add object id
	            root["obj"]["x"] = (scene_corners[0].x  + scene_corners[1].x ) / 2 ;
	            root["obj"]["y"] = (scene_corners[2].y  + scene_corners[3].y ) / 2 ;
	            root["obj"]["z"] = 0;
	            root["obj"]["time"] = elapsed;

	            Json::StyledWriter writer;
	            std::string out_string = writer.write(root);
	            io.post( [=]() { //con &out_string fa schifezze
	            	   std::cout << out_string << std::endl;
	                   dataWriter->writeData(out_string);
	                 });
				}
		}

		cv::imshow( "Good Matches 2", img_matches );
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
