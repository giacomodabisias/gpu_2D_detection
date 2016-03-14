/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */

#include <iostream>
#include <chrono>
#include "sift.h"


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
	} else

  	const unsigned int width = atoi(argv[2]);
  	const unsigned int height = atoi(argv[3]);
	Grabber grabber(width, height);

	cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	if(!image.data){ 
		std::cout<< " --(!) Error reading image" << std::endl; 
		return -1; 
	}

	std::vector<cv::KeyPoint> keypoints_image;
	std::vector<float>  descriptors_image;
	std::vector<cv::KeyPoint>  keypoints_frame;
	std::vector<float>  descriptors_frame;
	SiftGPUWrapper * sift = SiftGPUWrapper::getInstance();

	sift->detect(image, keypoints_image, descriptors_image );
	std::cout << "got sift descriptors for image" << std::endl;

	std::vector<cv::DMatch> good_matches;

	cv::Mat img_matches;
	namedWindow("Good Matches", cv::WINDOW_NORMAL);

	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	cv::Mat homography;
	std::vector<cv::Point2f> obj_corners(4);
	std::vector<cv::Point2f> scene_corners(4);

	obj_corners[0] = cvPoint(0,0); 
	obj_corners[1] = cvPoint(image.cols, 0 );
	obj_corners[2] = cvPoint(image.cols, image.rows ); 
	obj_corners[3] = cvPoint(0, image.rows );


	cv::Mat outImage, frame;

	while(true){
		using namespace std::chrono;

		auto tnow = high_resolution_clock::now();
		grabber.getFrame(frame);
		
		sift->detect(frame, keypoints_frame, descriptors_frame );
		sift->match(descriptors_image, 128, descriptors_frame, 128, &good_matches);

		auto tpost = high_resolution_clock::now();
		std::cout << "time " << duration_cast<duration<double>>(tpost-tnow).count()*1000 << std::endl;
	
		cv::drawMatches( image, keypoints_image, frame, keypoints_frame, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
				     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		drawKeypoints(frame, keypoints_frame, outImage);

		if (good_matches.size() >= 10){
			for( int i = 0; i < good_matches.size(); i++ ){
				obj.push_back( keypoints_image[ good_matches[i].queryIdx ].pt );
				scene.push_back( keypoints_frame[ good_matches[i].trainIdx ].pt );
			}

			homography = cv::findHomography( obj, scene, CV_RANSAC );

			if(homography.rows == 3 && homography.cols == 3){
				cv::perspectiveTransform( obj_corners, scene_corners, homography);
				cv::line( img_matches, scene_corners[0] + cv::Point2f( image.cols, 0), scene_corners[1] + cv::Point2f( image.cols, 0), cv::Scalar(0, 255, 0), 4 );
				cv::line( img_matches, scene_corners[1] + cv::Point2f( image.cols, 0), scene_corners[2] + cv::Point2f( image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
				cv::line( img_matches, scene_corners[2] + cv::Point2f( image.cols, 0), scene_corners[3] + cv::Point2f( image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
				cv::line( img_matches, scene_corners[3] + cv::Point2f( image.cols, 0), scene_corners[0] + cv::Point2f( image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
			}
		}
		cv::imshow ("key", outImage);
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
