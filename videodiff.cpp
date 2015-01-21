// Elijah Houle
// IntegriDroid/YouTrace research
//
// Describe the differences between two videos
//  (assuming one video is derived from the other)
//
//
// Set of possible transformations:
//
//  Spatial - scaling, cropping, object modification, color adjustment, grayscale, bordering, block artifact (lossy compression)
//  Temporal - scaling, cropping
//

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

const int DISTANCE_TAU = 16; // threshold for binarizing distance matrix

struct Transformation {
  std::string type;
  cv::Point2i position; // top-left corner of local spatial transformation
  cv::Point2f degree; // "degree" (may be length or factor) of transformation along each axis
};

struct Clip {
  std::pair<int, int> endpoints; // start and end frame indices
  std::pair<int, int> parentEndpoints; // start and end in parent
  std::vector<Transformation> transformations;
};

std::vector<Clip> deltaReport;


// get next block of 64 frames (overlapping with current block)
void getBlock(cv::VideoCapture V, std::vector<cv::Mat>& block) {
  int framesToRead = 64;
  if (!block.empty()) {
    block.erase(block.begin()); // sliding window
    framesToRead = 1;
  }
  for (int i = 0; i < framesToRead; ++i) {
    cv::Mat frame;
    bool flag = V.read(frame);
    if (flag == false) {
      block = std::vector<cv::Mat>();
      return;
    }
    cv::Mat tempFrame;
    cv::cvtColor(frame, tempFrame, CV_BGR2GRAY);
    cv::resize(tempFrame, frame, cv::Size(32, 32));
    block.push_back(frame);
  }
}

std::vector<int> dctHash(const std::vector<cv::Mat>& block) {
  // 3D DCT
  std::vector<cv::Mat> dctFrames;
  for (cv::Mat frame : block) { // 2D row-column DCTs
    cv::Mat frame64F;
    frame.convertTo(frame64F, CV_64F);
    cv::Mat dctFrame;
    cv::dct(frame64F, dctFrame);
    dctFrames.push_back(dctFrame);
  }
  std::vector<std::vector<double> > results;
  for (int i = 1; i <= 4; ++i) {
    for (int j = 1; j <= 4; ++j) { // only keeping 4x4x4 coefficients
      std::vector<double> tube;
      for (auto k : dctFrames) {
	tube.push_back(k.at<double>(i, j));
      }
      std::vector<double> result;
      cv::dct(tube, result); // DCT over "tubes" of 2D DCTs
      results.push_back(result);
    }
  }

  std::vector<double> coeffs;
  for (auto v : results) {
    for (int u = 1; u <= 4; ++u) {
      coeffs.push_back(v[u]);
    }
  }
  
  std::vector<double> coeffsSorted = coeffs;
  std::sort(coeffsSorted.begin(), coeffsSorted.end());
  double median;
  if (coeffsSorted.size() % 2 == 0) {
    median = (coeffsSorted[(coeffsSorted.size()-1) / 2] + coeffsSorted[coeffsSorted.size() / 2]) / 2.0;
  } else {
    median = coeffsSorted[coeffsSorted.size() / 2];
  }

  std::vector<int> hash;
  for (auto c : coeffs) {
    if (c > median) hash.push_back(1);
    else hash.push_back(0);
  }
  return hash;
}


std::vector<std::vector<int> > hashesOfBlocks(cv::VideoCapture V) {
  std::vector<std::vector<int> > hashes;
  std::vector<cv::Mat> block;
  while (true) {
    getBlock(V, block);
    if (block.empty()) {
      break;
    }
    hashes.push_back(dctHash(block));
  }
  return hashes;
}


int distance(const std::vector<int>& blockA, const std::vector<int>& blockB) {
  int dist = 0;
  for (int i = 0; i < 64; ++i) {
    dist += blockA[i] ^ blockB[i];
  }
  return dist;
}


// return pairs of matching frames [indices] between Va and Vb for comparison
std::vector<std::pair<int, int> > align(cv::VideoCapture Va, cv::VideoCapture Vb) {
  // split into blocks of 64 frames and get 3D DCT hash of each block
  std::vector<std::vector<int> > blocksVa = hashesOfBlocks(Va);
  std::vector<std::vector<int> > blocksVb = hashesOfBlocks(Vb);

  // compare pair-wise Hamming distance between each block
  int x = blocksVa.size();
  int y = blocksVb.size();
  cv::Mat distances(y, x, CV_8U, cv::Scalar(0));
  cv::Mat distancesBinarized(y, x, CV_8U, cv::Scalar(0));
  for (int i = 0; i < x; ++i) {
    for (int j = 0; j < y; ++j) {
      distances.at<uchar>(j, i) = distance(blocksVa[i], blocksVb[j]);
      // binarize using threshold
      if (distances.at<uchar>(j, i) <= DISTANCE_TAU) {
	distancesBinarized.at<uchar>(j, i) = 1.0f;
      } else {
	distancesBinarized.at<uchar>(j, i) = 0.0f;
      }
    }
  }

  // apply morphological opening
  cv::Mat elementErosion = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
  cv::Mat elementDilation = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30), cv::Point(-1, -1));
  cv::Mat distancesBinarizedTemp;
  cv::erode(distancesBinarized, distancesBinarizedTemp, elementErosion);
  cv::dilate(distancesBinarizedTemp, distancesBinarized, elementDilation);

  // label connected components
  int currentLabel = 1;
  std::vector<std::pair<int, int> > pixelQueue;
  cv::Mat distancesLabeled(y, x, CV_32S, cv::Scalar(0));
  for (int i = 0; i < x; ++i) {
    for (int j = 0; j < y; ++j) {
      if (distancesLabeled.at<int>(j, i) != 0) continue;
      if (distancesBinarized.at<uchar>(j, i) == 1) {
	distancesLabeled.at<int>(j, i) = currentLabel;
	pixelQueue.push_back(std::make_pair(j, i));
	while (!pixelQueue.empty()) {
	  std::pair<int, int> kl = pixelQueue.back();
	  pixelQueue.pop_back();
	  int l = std::get<0>(kl);
	  int k = std::get<1>(kl);
	  for (int m = k-1; m < k+2; ++m) {
	    for (int n = l-1; n < l+2; ++n) {
	      if (m >= 0 && m < x && n >= 0 && n < y) {
		if (distancesLabeled.at<int>(n, m) == 0 && distancesBinarized.at<uchar>(n, m) == 1) {
		  distancesLabeled.at<int>(n, m) = currentLabel;
		  pixelQueue.push_back(std::make_pair(n, m));
		}
	      }
	    }
	  }
	}
	++currentLabel;
      }
    }
  }

  // find distance minima for rows and columns of components
  std::vector<std::vector<cv::Point2i> > minima;
  std::vector<std::tuple<int, int, int, int> > boundaries;
  for (int label = 1; label < currentLabel; ++label) {
    std::vector<cv::Point2i> minimaLabel;
    std::tuple<int, int, int, int> boundariesLabel = std::make_tuple(x, 0, y, 0); // left, right, upper, lower
    for (int i = 0; i < x; ++i) {
      int minColumn = DISTANCE_TAU;
      cv::Point2i minCoords(-1, -1);
      for (int j = 0; j < y; ++j) {
	if (distancesLabeled.at<int>(j, i) == label) {
	  boundariesLabel = std::make_tuple(std::min(i, std::get<0>(boundariesLabel)), std::max(i, std::get<1>(boundariesLabel)), std::min(j, std::get<2>(boundariesLabel)), std::max(j, std::get<3>(boundariesLabel)));
	  int dist = distances.at<uchar>(j, i);
	  if (dist <= minColumn) {
	    minColumn = dist;
	    minCoords = cv::Point2i(i, j);
	  }
	}
      }
      if (minCoords.x != -1 || minCoords.y != -1) {
	minimaLabel.push_back(minCoords);
      }
    }
    for (int j = 0; j < y; ++j) {
      int minRow = DISTANCE_TAU;
      cv::Point2i minCoords(-1, -1);
      for (int i = 0; i < x; ++i) {
	if (distancesLabeled.at<int>(j, i) == label) {
	  int dist = distances.at<uchar>(j, i);
	  if (dist <= minRow) {
	    minRow = dist;
	    minCoords = cv::Point2i(i, j);
	  }
	}
      }
      if (minCoords.x != -1 || minCoords.y != -1) {
	minimaLabel.push_back(minCoords);
      }
    }
    minima.push_back(minimaLabel);
    boundaries.push_back(boundariesLabel);
  }

  // fit line through minima for each component
  std::vector<std::tuple<cv::Point2i, cv::Point2i, double, double> > matchingSegments;
  for (int label = 0; label < currentLabel-1; ++label) {
    if (minima[label].empty()) continue;
    cv::Vec4f line;
    cv::fitLine(minima[label], line, CV_DIST_L2, 0, 0.01, 0.01);
    double vx = line[0];
    double vy = line[1];
    double px = line[2];
    double py = line[3];
    double m = vy/vx;
    double b = py-m*px;
    cv::Point2i upperLeft = cv::Point2i(std::get<0>(boundaries[label]), m*std::get<0>(boundaries[label])+b);
    cv::Point2i lowerRight = cv::Point2i(std::get<1>(boundaries[label]), m*std::get<1>(boundaries[label])+b);
    // recompute line parameters
    m = (double) (lowerRight.y - upperLeft.y) / (lowerRight.x - upperLeft.x);
    b = py-m*px;
    matchingSegments.push_back(std::make_tuple(upperLeft, lowerRight, m, b));
  }

  std::vector<std::pair<int, int> > framePairs;
  std::vector<std::pair<int, int> > endpointsInParent;
  for (auto segment : matchingSegments) {
    cv::Point2i upperLeft = std::get<0>(segment);
    cv::Point2i lowerRight = std::get<1>(segment);
    double m = std::get<2>(segment);
    double b = std::get<3>(segment);
    std::vector<int> frameX;
    std::vector<int> frameY;
    std::vector<float> avglumaA;
    std::vector<float> avglumaB;
    int previousPy = -1;
    for (int px = upperLeft.x; px < std::min(lowerRight.x+64, x); ++px) {
      int py = m*px+b;
      if (py < 0 || py == previousPy) continue;
      if (py >= y) break;
      previousPy = py;
      Va.set(CV_CAP_PROP_POS_FRAMES, px);
      Vb.set(CV_CAP_PROP_POS_FRAMES, py);
      cv::Mat frame1;
      bool flag1 = Va.read(frame1);
      cv::Mat frame2;
      bool flag2 = Vb.read(frame2);
      if (flag1 && flag2) {
	frameX.push_back(px);
	frameY.push_back(py);
	cv::Mat frame1Gray;
	cv::cvtColor(frame1, frame1Gray, CV_BGR2GRAY);
	int N = frame1Gray.rows*frame1Gray.cols;
	float avgluma = 0;
	for (int i = 0; i < frame1Gray.cols; ++i) {
	  for (int j = 0; j < frame1Gray.rows; ++j) {
	    avgluma += frame1Gray.at<uchar>(j, i)/(float)N;
	  }
	}
	avglumaA.push_back(avgluma);
	cv::Mat frame2Gray;
	cv::cvtColor(frame2, frame2Gray, CV_BGR2GRAY);
	N = frame2Gray.rows*frame2Gray.cols;
	avgluma = 0;
	for (int i = 0; i < frame2Gray.cols; ++i) {
	  for (int j = 0; j < frame2Gray.rows; ++j) {
	    avgluma += frame2Gray.at<uchar>(j, i)/(float)N;
	  }
	}
	avglumaB.push_back(avgluma);
      }
    }
    cv::Mat s1(1, avglumaA.size()-1, CV_32F);
    cv::Mat s2(1, avglumaB.size()-1, CV_32F);
    for (int i = 1; i < avglumaA.size(); ++i) {
      s1.at<float>(i-1) = avglumaA[i] - avglumaA[i-1];
      s2.at<float>(i-1) = avglumaB[i] - avglumaB[i-1];
    }
    cv::Mat correl1;
    cv::filter2D(s1, correl1, -1, s2);
    cv::Mat correl2;
    cv::filter2D(s2, correl2, -1, s1);
    float max = correl1.at<float>(0);
    int shiftA = 0;
    for (int i = 1; i < correl1.cols; ++i) {
      float coeff = correl1.at<float>(i);
      if (coeff > max) {
	shiftA = i;
	max = coeff;
      }
    }
    max = correl2.at<float>(0);
    int shiftB = 0;
    for (int i = 1; i < correl2.cols; ++i) {
      float coeff = correl2.at<float>(i);
      if (coeff > max) {
	shiftB = i;
	max = coeff;
      }
    }
    int shift = (shiftB - shiftA) / 2;
    if (shift != 0) {
      if (shift > 0) {
	frameY.erase(frameY.begin(), frameY.begin() + shift);
	frameX.erase(frameX.end() - shift, frameX.end());
      } else {
	frameX.erase(frameX.begin(), frameX.begin() + shift);
	frameY.erase(frameY.end() - shift, frameY.end());
      }
    }
    Clip clip;
    clip.endpoints = std::make_pair(frameY[0], frameY[frameY.size()-1]);
    clip.parentEndpoints = std::make_pair(frameX[0], frameX[frameX.size()-1]);
    Transformation temporalScaling;
    temporalScaling.type = "temporal scaling";
    temporalScaling.degree.x = m;
    clip.transformations.push_back(temporalScaling);
    deltaReport.push_back(clip);
    std::vector<int> keyframes;
    keyframes.push_back(0);
    cv::Mat images[1];
    Va.set(CV_CAP_PROP_POS_FRAMES, frameX[0]);
    cv::Mat prevHist;
    cv::Mat hist;
    int channels[] = {0, 1, 2};
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    if (Va.read(images[0])) {
      cv::calcHist(images, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange);
      cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);
    }
    for (int i = 1; i < frameX.size(); ++i) {
      prevHist = hist;
      Va.set(CV_CAP_PROP_POS_FRAMES, frameX[i]);
      if (Va.read(images[0])) {
	cv::calcHist(images, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange);
	cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);
	double correlation = cv::compareHist(prevHist, hist, CV_COMP_CORREL);
	if (correlation < 0.7) {
	  keyframes.push_back(i);
	}
      }
    }
    keyframes.push_back(frameX.size()-1);
    for (auto i : keyframes) {
      framePairs.push_back(std::make_pair(frameX[i], frameY[i]));
    }
    endpointsInParent.push_back(std::make_pair(frameX[0], frameX[frameX.size()-1]));
  }
  Clip trimBegin;
  trimBegin.parentEndpoints = std::make_pair(0, endpointsInParent[0].first);
  Transformation temporalTrimming;
  temporalTrimming.type = "temporal trimming";
  trimBegin.transformations.push_back(temporalTrimming);
  deltaReport.push_back(trimBegin);
  for (int i = 0; i < endpointsInParent.size()-1; ++i) {
    Clip clip;
    clip.parentEndpoints = std::make_pair(endpointsInParent[i].second, endpointsInParent[i+1].first);
    clip.transformations.push_back(temporalTrimming);
    deltaReport.push_back(clip);
  }
  Clip trimEnd;
  trimEnd.parentEndpoints = std::make_pair(endpointsInParent[endpointsInParent.size()-1].second, x);
  trimEnd.transformations.push_back(temporalTrimming);
  deltaReport.push_back(trimEnd);
  return framePairs;
}


int getBorderSize(cv::Mat frame, bool x) {
  cv::Vec3b color = frame.at<cv::Vec3b>(0, 0);
  int borderSize = 0;
  bool endBorder = false;
  if (x) frame = frame.t();
  for (int i = 0; i < frame.rows / 2; ++i) {
    for (int j = 0; j < frame.cols; ++j) {
      if (frame.at<cv::Vec3b>(i, j) != color
	  || frame.at<cv::Vec3b>(frame.rows-i-1, j) != color) {
	endBorder = true;
      }
    }
    if (endBorder) break;
    ++borderSize;
  }
  return borderSize;
}


void detectAndCropBorders(Clip& clip, cv::Mat& frame1, cv::Mat& frame2) {
  int borderY1 = getBorderSize(frame1, false);
  int borderY2 = getBorderSize(frame2, false);
  int borderX1 = getBorderSize(frame1, true);
  int borderX2 = getBorderSize(frame2, true);
  int borderYDelta = borderY1 - borderY2;
  int borderXDelta = borderX1 - borderX2;
  if (frame1.rows != frame2.rows || frame1.cols != frame2.cols) {
    float scalingY = (float) frame2.rows / frame1.rows;
    float scalingX = (float) frame2.cols / frame1.cols;
    if (((frame1.rows - frame2.rows) / 2 == borderYDelta)
	|| ((frame1.cols - frame2.cols) / 2 == borderXDelta)) {
      Transformation borderChange;
      borderChange.type = "border change";
      borderChange.degree = cv::Point2f(-borderXDelta, -borderYDelta);
      clip.transformations.push_back(borderChange);
    } else if (scalingY * borderY1 == borderY2
	       || scalingX * borderX1 == borderX2) {
      Transformation spatialScaling;
      spatialScaling.type = "spatial scaling";
      spatialScaling.degree = cv::Point2f(scalingX, scalingY);
      clip.transformations.push_back(spatialScaling);
    }
  } else if (borderX1 != borderX2 || borderY1 != borderY2) {
    Transformation borderChange;
    borderChange.type = "border change";
    borderChange.degree = cv::Point2f(-borderXDelta, -borderYDelta);
    clip.transformations.push_back(borderChange);
  }
  frame1 = frame1(cv::Rect(borderX1, borderY1, frame1.cols-2*borderX1, frame1.rows-2*borderY1));
  frame2 = frame2(cv::Rect(borderX2, borderY2, frame2.cols-2*borderX2, frame2.rows-2*borderY2));
}


void compareBlocks(cv::Mat& region1, cv::Mat& region2) {
  const int BLOCK_SIZE = 64;
  std::pair<cv::Rect, double> maxdiff = std::make_pair(cv::Rect(0, 0, 0, 0), 0);
  for (int i = 0; i < region1.cols / BLOCK_SIZE; ++i) {
    for (int j = 0; j < region1.rows / BLOCK_SIZE; ++j) {
      cv::Rect block = cv::Rect(i*BLOCK_SIZE, j*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
      double diff = std::abs(cv::mean(region1(block))[0] - cv::mean(region2(block))[0]);
      if (diff > maxdiff.second) {
	maxdiff = std::make_pair(block, diff);
      }
    }
  }
  std::cout << "Max diff: " << maxdiff.first.x << ", " << maxdiff.first.y << ": " << maxdiff.second << std::endl;
  cv::rectangle(region1, maxdiff.first, cv::Scalar(0, 0, 0));
  cv::rectangle(region2, maxdiff.first, cv::Scalar(0, 0, 0));
  cv::imshow("region1", region1);
  cv::imshow("region2", region2);
}


void detectKeypoints(Clip& clip, cv::Mat& frame1, cv::Mat& frame2) {
  // keypoint matching
  cv::OrbFeatureDetector detector;
  std::vector<cv::KeyPoint> kp1, kp2;
  detector.detect(frame1, kp1);
  detector.detect(frame2, kp2);
  cv::OrbDescriptorExtractor extractor;
  cv::Mat des1, des2;
  extractor.compute(frame1, kp1, des1);
  extractor.compute(frame2, kp2, des2);
  cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(6, 12, 1), new cv::flann::SearchParams(50));
  std::vector<std::vector<cv::DMatch> > matches;
  matcher.knnMatch(des1, des2, matches, 2);
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < des1.rows; i++) {
    if (matches[i].size() == 2) { // ratio test
      if (matches[i][0].distance <= 0.7*matches[i][1].distance) {
	good_matches.push_back(matches[i][0]);
      }
    }
  }

  // crop to keypoints
  cv::Point2i point1 = kp1[good_matches[0].queryIdx].pt;
  cv::Point2i point2 = kp2[good_matches[0].trainIdx].pt;
  cv::Point2i upperLeft1 = point1;
  cv::Point2i lowerRight1 = point1;
  cv::Point2i upperLeft2 = point2;
  cv::Point2i lowerRight2 = point2;
  for (auto m : good_matches) {
    point1 = kp1[m.queryIdx].pt;
    point2 = kp2[m.trainIdx].pt;
    upperLeft1 = cv::Point2i(std::min(upperLeft1.x, point1.x), std::min(upperLeft1.y, point1.y));
    lowerRight1 = cv::Point2i(std::max(lowerRight1.x, point1.x), std::max(lowerRight1.y, point1.y));
    upperLeft2 = cv::Point2i(std::min(upperLeft2.x, point2.x), std::min(upperLeft2.y, point2.y));
    lowerRight2 = cv::Point2i(std::max(lowerRight2.x, point2.x), std::max(lowerRight2.y, point2.y));
  }
  int rect1Width = lowerRight1.x - upperLeft1.x;
  int rect1Height = lowerRight1.y - upperLeft1.y;
  int rect2Width = lowerRight2.x - upperLeft2.x;
  int rect2Height = lowerRight2.y - upperLeft2.y;
  float croppingFactorX1 = (float) rect1Width / frame1.cols;
  float croppingFactorY1 = (float) rect1Height / frame1.rows;
  float croppingFactorX2 = (float) rect2Width / frame2.cols;
  float croppingFactorY2 = (float) rect2Height / frame2.rows;
  if (croppingFactorX1 != croppingFactorX2 || croppingFactorY1 != croppingFactorY2) {
    Transformation spatialCropping;
    spatialCropping.type = "spatial cropping";
    spatialCropping.degree = cv::Point2f(croppingFactorX2 / croppingFactorX1, croppingFactorY2 / croppingFactorY1);
    clip.transformations.push_back(spatialCropping);
  }
  if (rect1Width != rect2Width || rect1Height != rect2Height) {
    Transformation spatialScaling;
    spatialScaling.type = "spatial scaling";
    spatialScaling.degree = cv::Point2f((float) rect2Width / rect1Width, (float) rect2Height / rect1Height);
    clip.transformations.push_back(spatialScaling);
  }

  cv::Mat region1 = frame1(cv::Rect(upperLeft1.x, upperLeft1.y, rect1Width, rect1Height));
  cv::Mat region2 = frame2(cv::Rect(upperLeft2.x, upperLeft2.y, rect2Width, rect2Height));
  cv::cvtColor(region1, region1, CV_BGR2YCrCb);
  cv::cvtColor(region2, region2, CV_BGR2YCrCb);

  cv::Mat mean1;
  cv::Mat stddev1;
  cv::meanStdDev(region1, mean1, stddev1);
  cv::Mat mean2;
  cv::Mat stddev2;
  cv::meanStdDev(region2, mean2, stddev2);
  double numerator1 = std::max(stddev1.at<double>(1), stddev2.at<double>(1));
  double denominator1 = std::min(stddev1.at<double>(1), stddev2.at<double>(1));
  double numerator2 = std::max(stddev1.at<double>(2), stddev2.at<double>(2));
  double denominator2 = std::min(stddev1.at<double>(2), stddev2.at<double>(2));
  if (numerator1/denominator1 > 2 || numerator2/denominator2 > 2) { // hardcoded critical value
    Transformation colorAdjustment;
    colorAdjustment.type = "color adjustment";
    colorAdjustment.degree = cv::Point2f(mean2.at<double>(1) - mean1.at<double>(1), mean2.at<double>(2) - mean1.at<double>(2));
    clip.transformations.push_back(colorAdjustment);
  }

  // scale to smaller region to compare
  cv::Mat region1Scaled;
  cv::Mat region2Scaled;
  cv::Point2f scale1 = cv::Point2f(1, 1);
  cv::Point2f scale2 = cv::Point2f(1, 1);
  if (region1.cols < region2.cols || region1.rows < region2.rows) {
    region1Scaled = region1;
    cv::resize(region2, region2Scaled, cv::Size(region1.cols, region1.rows), 0, 0, CV_INTER_AREA);
    scale2 = cv::Point2f((float) region1.cols / region2.cols, (float) region1.rows / region2.rows);
  } else {
    region2Scaled = region2;
    cv::resize(region1, region1Scaled, cv::Size(region2.cols, region2.rows), 0, 0, CV_INTER_AREA);
    scale1 = cv::Point2f((float) region2.cols / region1.cols, (float) region2.rows / region1.rows);
  }
  cv::Mat region1Channels[3];
  cv::Mat region2Channels[3];
  cv::split(region1Scaled, region1Channels);
  cv::split(region2Scaled, region2Channels);
  cv::Mat region1Gray = region1Channels[0];
  cv::Mat region2Gray = region2Channels[0];
  cv::equalizeHist(region1Gray, region1Gray);
  cv::equalizeHist(region2Gray, region2Gray);
  compareBlocks(region1Gray, region2Gray);
}


void diff(cv::VideoCapture Va, cv::VideoCapture Vb, const std::vector<std::pair<int, int> >& framePairs) {
  for (int i = 0; i < framePairs.size()-1; ++i) {
    std::pair<int, int> p = framePairs[i];
    Va.set(CV_CAP_PROP_POS_FRAMES, p.first);
    Vb.set(CV_CAP_PROP_POS_FRAMES, p.second);
    cv::Mat frame1;
    bool flag1 = Va.read(frame1);
    cv::Mat frame2;
    bool flag2 = Vb.read(frame2);
    if (flag1 && flag2) {
      Clip clip;
      clip.endpoints = std::make_pair(p.second, framePairs[i+1].second);
      clip.parentEndpoints = std::make_pair(p.first, framePairs[i+1].first);
      cv::imshow("frame1", frame1);
      cv::imshow("frame2", frame2);
      cv::waitKey(0);
      detectAndCropBorders(clip, frame1, frame2);
      detectKeypoints(clip, frame1, frame2);
      cv::imshow("frame1", frame1);
      cv::imshow("frame2", frame2);
      cv::waitKey(0);
      if (!clip.transformations.empty()) deltaReport.push_back(clip);
    }
  }
}


int main(int argc, char** argv) {
  cv::VideoCapture Va(argv[1]);
  cv::VideoCapture Vb(argv[2]);
  std::vector<std::pair<int, int> > framePairs = align(Va, Vb);
  diff(Va, Vb, framePairs);
  std::cout << "delta report:" << std::endl;
  for (Clip clip : deltaReport) {
    std::cout << " clip at [" << clip.endpoints.first << ", "
	      << clip.endpoints.second << "]; in parent at ["
	      << clip.parentEndpoints.first << ", "
	      << clip.parentEndpoints.second << "]" << std::endl;
    for (Transformation transf : clip.transformations) {
      std::cout << "  " << transf.type << " at position " << transf.position
		<< " with degree " << transf.degree << std::endl;
    }
  }
}
