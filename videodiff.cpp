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

#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

const int DISTANCE_TAU = 16; // threshold for binarizing distance matrix
const int BLOCK_SIZE = 32; // standard block size to compare between frames

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
  std::time_t starttime = std::time(nullptr);
  // split into blocks of 64 frames and get 3D DCT hash of each block
  std::vector<std::vector<int> > blocksVa = hashesOfBlocks(Va);
  std::time_t endtime = std::time(nullptr);
  std::cout << "Total number of blocks in Va: " << blocksVa.size()
	    << "; time to hash: " << (endtime-starttime) << std::endl;
  std::time(&starttime);
  std::vector<std::vector<int> > blocksVb = hashesOfBlocks(Vb);
  std::time(&endtime);
  std::cout << "Total number of blocks in Vb: " << blocksVb.size()
	    << "; time to hash: " << (endtime-starttime) << std::endl;

  std::time(&starttime);
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
	distancesBinarized.at<uchar>(j, i) = 1;
      } else {
	distancesBinarized.at<uchar>(j, i) = 0;
      }
    }
  }
  std::time(&endtime);
  std::cout << "Time to compute distance matrix and binarize: " << (endtime-starttime) << std::endl;
  cv::imwrite("distances.ppm", distances);
  cv::imwrite("distancesBinarized.ppm", distancesBinarized);

  std::time(&starttime);
  // apply morphological opening
  cv::Mat elementErosion = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8), cv::Point(-1, -1));
  cv::Mat elementDilation = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50, 50), cv::Point(-1, -1));
  cv::Mat distancesBinarizedTemp;
  cv::erode(distancesBinarized, distancesBinarizedTemp, elementErosion);
  cv::dilate(distancesBinarizedTemp, distancesBinarized, elementDilation);
  std::time(&endtime);
  std::cout << "Time to apply morphological opening: " << (endtime-starttime) << std::endl;
  cv::imwrite("distancesOpened.ppm", distancesBinarized);

  std::time(&starttime);
  // grab points == 1 from binarized distance matrix
  std::vector<cv::Point2i> matchingPoints;
  for (int i = 0; i < x; ++i) {
    for (int j = 0; j < y; ++j) {
      if (distancesBinarized.at<uchar>(j, i) == 1) {
	matchingPoints.push_back(cv::Point2i(i, j));
      }
    }
  }
  // load into matrix for clustering
  cv::Mat matchingPointsMat = cv::Mat(matchingPoints.size(), 2, CV_32F);
  for (unsigned int i = 0; i < matchingPoints.size(); ++i) {
    matchingPointsMat.at<float>(i, 0) = matchingPoints[i].x;
    matchingPointsMat.at<float>(i, 1) = matchingPoints[i].y;
  }
  cv::Mat clusterLabels;
  cv::Mat clusterCenters;
  double bestCompactness = -1;
  cv::Mat bestLabels;
  int numClusters;
  for (int K = 5; K > 0; --K) {
    double compactness = cv::kmeans(matchingPointsMat, K, clusterLabels,
				    cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 1.0), 3, cv::KMEANS_RANDOM_CENTERS);
    if (compactness < bestCompactness || bestCompactness == -1) {
      bestCompactness = compactness;
      bestLabels = clusterLabels.clone();
      numClusters = K;
    }
  }
  std::time(&endtime);
  std::cout << "Total number of clusters: " << numClusters
	    << "; time to discover: " << (endtime-starttime) << std::endl;

  std::time(&starttime);
  // find distance minima for rows and columns of components
  bool *initialized = new bool[numClusters];
  std::vector<cv::Point2i> *minima = new std::vector<cv::Point2i>[numClusters];
  std::pair<int, int> *boundaries = new std::pair<int, int>[numClusters];
  std::map<int, int> *minRowCoords = new std::map<int, int>[numClusters];
  std::map<int, int> *minRowDists = new std::map<int, int>[numClusters];
  std::map<int, int> *minColumnCoords = new std::map<int, int>[numClusters];
  std::map<int, int> *minColumnDists = new std::map<int, int>[numClusters];
  for (unsigned int i = 0; i < matchingPoints.size(); ++i) {
    int label = bestLabels.at<int>(i);
    std::map<int, int> *minRowCoord = &minRowCoords[label];
    std::map<int, int> *minRowDist = &minRowDists[label];
    std::map<int, int> *minColumnCoord = &minColumnCoords[label];
    std::map<int, int> *minColumnDist = &minColumnDists[label];
    int px = matchingPoints[i].x;
    int py = matchingPoints[i].y;
    int distance = distances.at<uchar>(py, px);
    std::map<int, int>::iterator yit = minRowDist->find(py);
    std::map<int, int>::iterator xit = minColumnDist->find(px);
    if (yit == minRowDist->end() || distance < yit->second) {
      (*minRowCoord)[py] = px;
      (*minRowDist)[py] = distance;
    }
    if (xit == minColumnDist->end() || distance < xit->second) {
      (*minColumnCoord)[px] = py;
      (*minColumnDist)[px] = distance;
    }
    if (!initialized[label]) {
      boundaries[label] = std::make_pair(px, px);
      initialized[label] = true;
    } else {
      boundaries[label] = std::make_pair(std::min(px, boundaries[label].first), std::max(px, boundaries[label].second)); // min x, max x
    }
  }
  for (int i = 0; i < numClusters; ++i) {
    for (std::map<int, int>::iterator it = minRowCoords[i].begin(); it != minRowCoords[i].end(); ++it) {
      minima[i].push_back(cv::Point2i(it->second, it->first));
    }
    for (std::map<int, int>::iterator it = minColumnCoords[i].begin(); it != minColumnCoords[i].end(); ++it) {
      minima[i].push_back(cv::Point2i(it->first, it->second));
    }
  }
  std::time(&endtime);
  std::cout << "Time to find distance minima for all components: " << (endtime-starttime) << std::endl;

  std::time(&starttime);
  // fit line through minima for each component
  std::vector<std::tuple<cv::Point2i, cv::Point2i, double, double> > matchingSegments;
  for (int label = 0; label < numClusters; ++label) {
    if (minima[label].empty()) continue;
    cv::Vec4f line;
    cv::fitLine(minima[label], line, CV_DIST_L2, 0, 0.01, 0.01);
    double vx = line[0];
    double vy = line[1];
    double px = line[2];
    double py = line[3];
    double m = vy/vx;
    double b = py-m*px;
    cv::Point2i upperLeft = cv::Point2i(boundaries[label].first, m*boundaries[label].first+b);
    cv::Point2i lowerRight = cv::Point2i(boundaries[label].second, m*boundaries[label].second+b);
    // recompute line parameters
    m = (double) (lowerRight.y - upperLeft.y) / (lowerRight.x - upperLeft.x);
    b = py-m*px;
    matchingSegments.push_back(std::make_tuple(upperLeft, lowerRight, m, b));
  }
  std::time(&endtime);
  std::cout << "Time to fit lines for all components: " << (endtime-starttime) << std::endl;
  std::cout << "# matching segments: " << matchingSegments.size() << std::endl;
  for (auto segment : matchingSegments) {
    std::cout << "(" << std::get<0>(segment) << ") -> (" << std::get<1>(segment) << "): y = " << std::get<2>(segment) << "x + " << std::get<3>(segment) << std::endl;
  }

  delete[] initialized;
  delete[] minima;
  delete[] boundaries;
  delete[] minRowCoords;
  delete[] minRowDists;
  delete[] minColumnCoords;
  delete[] minColumnDists;

  std::time(&starttime);
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
	avglumaA.push_back((float)cv::mean(frame1Gray)[0]);
	cv::Mat frame2Gray;
	cv::cvtColor(frame2, frame2Gray, CV_BGR2GRAY);
	avglumaB.push_back((float)cv::mean(frame2Gray)[0]);
      }
    }
    cv::Mat s1(1, avglumaA.size()-1, CV_32F);
    cv::Mat s2(1, avglumaB.size()-1, CV_32F);
    for (unsigned int i = 1; i < avglumaA.size(); ++i) {
      s1.at<float>(i-1) = avglumaA[i] - avglumaA[i-1];
      s2.at<float>(i-1) = avglumaB[i] - avglumaB[i-1];
    }
    cv::Mat correl1;
    cv::filter2D(s1, correl1, -1, s2, cv::Point2i(0, 0));
    cv::Mat correl2;
    cv::filter2D(s2, correl2, -1, s1, cv::Point2i(0, 0));
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
    unsigned int shift = shiftA + 1 - correl1.cols - shiftB;
    if (shift != 0 && frameX.size() > shift) {
      if (shift > 0) {
	frameY.erase(frameY.begin(), frameY.begin() + shift);
	frameX.erase(frameX.end() - shift, frameX.end());
      } else {
	frameX.erase(frameX.begin(), frameX.begin() - shift);
	frameY.erase(frameY.end() + shift, frameY.end());
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
    for (unsigned int i = 1; i < frameX.size()-1; ++i) {
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
  for (unsigned int i = 0; i < endpointsInParent.size()-1; ++i) {
    if (endpointsInParent[i+1].first > endpointsInParent[i].second) {
      Clip clip;
      clip.parentEndpoints = std::make_pair(endpointsInParent[i].second, endpointsInParent[i+1].first);
      clip.transformations.push_back(temporalTrimming);
      deltaReport.push_back(clip);
    }
  }
  if (endpointsInParent[endpointsInParent.size()-1].second != x-1) {
    Clip trimEnd;
    trimEnd.parentEndpoints = std::make_pair(endpointsInParent[endpointsInParent.size()-1].second, x-1);
    trimEnd.transformations.push_back(temporalTrimming);
    deltaReport.push_back(trimEnd);
  }
  std::time(&endtime);
  std::cout << "Time to correlate sequences and match frames: " << (endtime-starttime) << std::endl;
  return framePairs;
}


int getBorderSize(cv::Mat frame, bool x) {
  cv::Vec3b color = frame.at<cv::Vec3b>(0, 0);
  int borderSize = 0;
  bool endBorder = false;
  if (x) frame = frame.t();
  for (int i = 0; i < frame.rows / 2; ++i) {
    for (int j = 0; j < frame.cols; ++j) {
      cv::Vec3b diff1 = frame.at<cv::Vec3b>(i, j) - color;
      cv::Vec3b diff2 = frame.at<cv::Vec3b>(frame.rows-i-1, j) - color;
      if (std::abs(diff1.val[0]) > 5 || std::abs(diff1.val[1]) > 5
	  || std::abs(diff1.val[2]) > 5 || std::abs(diff2.val[0]) > 5
	  || std::abs(diff2.val[1]) > 5 || std::abs(diff2.val[2]) > 5) {
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


std::vector<Transformation> compareBlocks(cv::Mat& region1, cv::Mat& region2, const int blockSize, const float ssimThreshold) {
  int BLOCK_AREA = blockSize*blockSize;
  const double c_1 = 6.5025;  // (k1*L)^2 = (0.01*255)^2 -- k's are default constants
  const double c_2 = 58.5225; // (k2*L)^2 = (0.03*255)^2 -- L is dynamic range
  std::vector<Transformation> blockModifications;
  int i, j;
  for (i = 0; i < region1.cols / blockSize; ++i) {
    for (j = 0; j < region1.rows / blockSize; ++j) {
      int subblockSize = 0.88*blockSize;
      int gap = blockSize-subblockSize;
      subblockSize -= gap;
      cv::Rect block = cv::Rect(i*blockSize+gap, j*blockSize+gap, subblockSize, subblockSize); // center subblock
      cv::Mat block1 = region1(block);
      // account for imprecision in keypoint matching
      // compare to all other subblocks within block, get max SSIM
      double max_ssim = -1.0;
      for (int block_shift_x = 0; block_shift_x < gap; ++block_shift_x) {
	for (int block_shift_y = 0; block_shift_y < gap; ++block_shift_y) {
	  cv::Rect subblock = cv::Rect(i*blockSize + block_shift_x, j*blockSize + block_shift_y, subblockSize, subblockSize);
	  cv::Mat block2 = region2(subblock);
	  double mean_x = 0;
	  double mean_y = 0;
	  double variance_x = 0;
	  double variance_y = 0;
	  double covariance = 0;
	  for (int k = 0; k < subblockSize; ++k) {
	    for (int l = 0; l < subblockSize; ++l) {
	      double x = block1.at<uchar>(l, k);
	      double y = block2.at<uchar>(l, k);
	      mean_x += x / BLOCK_AREA;
	      mean_y += y / BLOCK_AREA;
	      variance_x += x*x / BLOCK_AREA;
	      variance_y += y*y / BLOCK_AREA;
	      covariance += x*y / BLOCK_AREA;
	    }
	  }
	  double meansquare_x = mean_x*mean_x;
	  double meansquare_y = mean_y*mean_y;
	  variance_x -= meansquare_x;
	  variance_y -= meansquare_y;
	  covariance -= mean_x*mean_y;
	  double ssim = ((2*mean_x*mean_y + c_1)*(2*covariance + c_2)) / ((meansquare_x + meansquare_y + c_1)*(variance_x + variance_y + c_2));
	  if (ssim > max_ssim) max_ssim = ssim;
	}
      }
      if (max_ssim < ssimThreshold) {
	Transformation blockModification;
	blockModification.type = "block modification";
	blockModification.position = cv::Point2i(i*blockSize, j*blockSize);
	blockModification.degree = cv::Point2f(blockSize, max_ssim);
	blockModifications.push_back(blockModification);
      }
    }
  }
  if (blockModifications.size() < (i*j)*0.4) { // hardcoded threshold
    // -- detected modifications must make up less than 40% of total blocks
    return blockModifications;
  } else {
    return std::vector<Transformation>();
  }
}


void detectKeypoints(Clip& clip, cv::Mat& frame1, cv::Mat& frame2) {
  // keypoint matching
  cv::SurfFeatureDetector detector;
  std::vector<cv::KeyPoint> kp1, kp2;
  detector.detect(frame1, kp1);
  detector.detect(frame2, kp2);
  cv::SurfDescriptorExtractor extractor;
  cv::Mat des1, des2;
  extractor.compute(frame1, kp1, des1);
  extractor.compute(frame2, kp2, des2);
  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;
  matcher.match(des1, des2, matches);
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < des1.rows; i++) {
    if (matches[i].distance < 0.1) { // hardcoded threshold
      good_matches.push_back(matches[i]);
    }
  }
  
  if (good_matches.size() < 50) return; // hardcoded threshold -- can't proceed with comparison if frames don't match

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
    int size1 = kp1[m.queryIdx].size / 2;
    int size2 = kp2[m.trainIdx].size / 2;
    upperLeft1 = cv::Point2i(std::max(std::min(upperLeft1.x, point1.x - size1), 0), std::max(std::min(upperLeft1.y, point1.y - size1), 0));
    lowerRight1 = cv::Point2i(std::min(frame1.cols-1, std::max(lowerRight1.x, point1.x + size1)), std::min(frame1.rows-1, std::max(lowerRight1.y, point1.y + size1)));
    upperLeft2 = cv::Point2i(std::max(std::min(upperLeft2.x, point2.x - size2), 0), std::max(std::min(upperLeft2.y, point2.y - size2), 0));
    lowerRight2 = cv::Point2i(std::min(frame2.cols-1, std::max(lowerRight2.x, point2.x + size2)), std::min(frame2.rows-1, std::max(lowerRight2.y, point2.y + size2)));
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

  // resize region1 to region2 to compare
  cv::Mat region1Scaled;
  cv::resize(region1, region1Scaled, cv::Size(region2.cols, region2.rows), 0, 0, CV_INTER_AREA);
  cv::Mat region1Channels[3];
  cv::Mat region2Channels[3];
  cv::split(region1Scaled, region1Channels);
  cv::split(region2, region2Channels);
  cv::Mat region1Gray = region1Channels[0];
  cv::Mat region2Gray = region2Channels[0];
  std::vector<Transformation> checkFramesMatch = compareBlocks(region1Gray, region2Gray, std::min(region1Gray.cols, region1Gray.rows), 0.9);
  if (!checkFramesMatch.empty()) return;
  std::vector<Transformation> blockModifications = compareBlocks(region1Gray, region2Gray, BLOCK_SIZE, 0.5);
  clip.transformations.insert(clip.transformations.end(), blockModifications.begin(), blockModifications.end());
}


void diff(cv::VideoCapture Va, cv::VideoCapture Vb, const std::vector<std::pair<int, int> >& framePairs) {
  for (unsigned int i = 0; i < framePairs.size()-1; ++i) {
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
      detectAndCropBorders(clip, frame1, frame2);
      detectKeypoints(clip, frame1, frame2);
      if (!clip.transformations.empty()) deltaReport.push_back(clip);
    }
  }
}


int main(int argc, char** argv) {
  cv::VideoCapture Va(argv[1]);
  cv::VideoCapture Vb(argv[2]);
  std::vector<std::pair<int, int> > framePairs = align(Va, Vb);
  std::time_t starttime = std::time(nullptr);
  diff(Va, Vb, framePairs);
  std::time_t endtime = std::time(nullptr);
  std::cout << "Total number of frames: " << framePairs.size()
	    << "; time to compare: " << (endtime-starttime) << std::endl;
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
