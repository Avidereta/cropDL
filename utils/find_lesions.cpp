#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <unistd.h>

using namespace cv;
using std::ifstream;
using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::min;
using std::max;

typedef pair<Point, Point> line_t;

vector<line_t> parseAnnotations(string annotations_file) {
  vector<line_t> results;
  FILE *fin = fopen(annotations_file.c_str(), "r");
  if (!fin) {
    cout << "Cannot open file \"" << annotations_file << "\"";
    return results;
  }

  fscanf(fin, "%*[^\n]\n");
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  while(fscanf(fin, "%*d,%*[a-zA-Z_0-9],%d,%d,%d,%d,%*[^\n]\n", &x1, &y1, &x2, &y2) == 4)
    results.emplace_back(Point(x1, y1), Point(x2, y2));

  fclose(fin);
  return results;
}

double getMean(vector<int> values) {
  double sum = 0;
  for (int a : values)
    sum += a;
  return sum / values.size();
}

double getStd(vector<int> values, double mean) {
  double std_sum = 0;
  for (int a : values)
    std_sum += (a - mean) * (a - mean);
  return sqrt(std_sum / values.size());
}

void getLineParameters(Mat mat, line_t line, int *pmean, int *pstd, const size_t steps_count=10000) {
  size_t sum = 0;
  vector<int> all_points; 
  for (double multiplier = 0.1; multiplier < 1; multiplier += 0.8 / steps_count) {
    Point current = line.first + (line.second - line.first) * multiplier;
    all_points.push_back(mat.at<uchar>(current));
  }
  
  double mean = getMean(all_points);
  double std = getStd(all_points, mean);
   
  vector<int> filtered_points;
  for (int val : all_points) {
    if (mean - std < val && val < mean + std)
      filtered_points.push_back(val);
  }

  *pmean = getMean(filtered_points);
  *pstd = getStd(filtered_points, mean);
}

Mat getMask(Mat source, line_t line) {
  Mat mask;
  blur(source, mask, Size(4,4));

  int mean = 0, std = 0;
  getLineParameters(source, line, &mean, &std);
  inRange(mask, mean - 2*std, mean + 2*std, mask);
  
  Mat ellipse_mask = Mat::zeros(source.rows, source.cols, CV_8U);
  Point center = (line.first + line.second) / 2;
  int seglen = sqrt((line.first - line.second).dot(line.first - line.second)) / 2;
  double angle = atan2(line.first.y - line.second.y, line.first.x - line.second.x) * 180 / M_PI;
  ellipse(ellipse_mask, center, {seglen, seglen / 6}, angle, 0, 360, {255}, -1);

  bitwise_and(mask, ellipse_mask, mask);

  return mask;
}

void processOneImage(string source, string annotations_file, string output, string output_masks) {
  Mat src = imread(source);
  if (!src.data) {
    cerr << "Cannot open file \"" << source << "\"";
    return;
  }

  Mat masks = Mat::zeros(src.rows, src.cols, CV_8UC1);
  vector<line_t> annotation_lines = parseAnnotations(annotations_file);

  for(auto& l : annotation_lines) {
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    Mat planes[3];
    split(hsv, planes);

    Mat mask = getMask(planes[0], l);
    bitwise_or(masks, mask, masks);
    bitwise_not(mask, mask);
  }
    
  for(auto& l : annotation_lines)
    line(src, l.first, l.second, {0, 0, 255}, 1);
  for(auto& l : annotation_lines) {
    Point center = (l.first + l.second) / 2;
    int seglen = sqrt((l.first - l.second).dot(l.first - l.second)) / 2;
    double angle = atan2(l.first.y - l.second.y, l.first.x - l.second.x) * 180 / M_PI;
    ellipse(src, center, {seglen, seglen / 6}, angle, 0, 360, {0, 255, 0}, 1);
  }

  imwrite(output, src);
  imwrite(output_masks, masks);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "USAGE: ./find_lesions input_files" << endl;
    return -1;
  }
  ifstream fin(argv[1], std::ifstream::in);
  if (!fin.is_open()) {
    cout << "Cannot open file \"" << argv[1] << "\"" << endl;
    return -1;
  }

  while (!fin.eof()) {
    string picture_file; fin >> picture_file;
    if (picture_file.size() == 0)
      break;
    string annotations_file; fin >> annotations_file;
    string output_annotated; fin >> output_annotated;
    string masks; fin >> masks;

    processOneImage(picture_file, annotations_file, output_annotated, masks);
  }

  return 0;
  }

