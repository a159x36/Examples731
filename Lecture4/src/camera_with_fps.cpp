#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace chrono;

class Menu {
    private:
    string menutitle;
    vector<string> menuitems;
    Mat_<Vec3b> image;
    int width=180;
    int highlighted=-1;
    int selected=-1;
    Vec3b bg=Vec3b(0xbf,0xbe,0xbb);
    Vec3b grey=Vec3b(0xef,0xf0,0xf1);
    Vec3b blue=Vec3b(0xe9,0xae,0x3d);
    Vec3b black=Vec3b(0,0,0);
    Vec3b white=Vec3b(0xff,0xff,0xff);
    public:
    Menu(string title,vector<string> items) {
        menutitle=title;
        menuitems=items;
        image=Mat_<Vec3b>(menuitems.size()*32,width);
        namedWindow(menutitle);
        setMouseCallback(menutitle,onmouse,(void *)this);
        drawmenu();
    }
    static void onmouse(int event, int , int y, int , void *userdata) {
        Menu *m=(Menu *)userdata;
        if(event==0) m->sethighlighted(y/32);
        if(event==1) m->setselected(y/32);
        m->drawmenu();
    }
    void setselected(int i) {
        selected=i;
        if(menuitems[i]=="Exit") exit(1);
    }
    string getselected() {
        if(selected==-1) return "";
        return menuitems[selected];
    }
    void sethighlighted(int i) {
        highlighted=i;
    }
    void drawmenu() {
        image=bg;
        for(int i=0;i<(int)menuitems.size();i++) {
            Vec3b bgcol=grey;
            Vec3b fgcol=black;
            if(i==highlighted) {
                bgcol=blue;
                fgcol=white;
            }
            if(i==selected) {
                bgcol=black;
                fgcol=white;
            }
            rectangle(image,Rect(1,i*32,width-2,32),bgcol,FILLED);
            putText(image, menuitems[i], Vec2i(12, 20 + i*32), FONT_HERSHEY_PLAIN, 1,
                fgcol,1,LINE_AA);
        }
        imshow(menutitle,image);
    }
};

int p1,p2;
void gaussian(Mat_<uint8_t> im) {
  GaussianBlur(im,im,Size(0,0),8.0);
}

void laplacian(Mat_<uint8_t> im) {
  Mat_<int16_t>lap(im.rows,im.cols);
  Laplacian(im,lap,CV_16S,7,0.02,128);
  convertScaleAbs(lap,im);
}

void hough_lines(Mat_<Vec3b> im) {
  Mat_<uint8_t> proc(im.rows,im.cols);
  Canny(im,proc,50,100);    
  vector<Vec2f> lines;
  HoughLines(proc,lines,1,CV_PI/180,150);
  for(size_t i=0;i<lines.size();i++) {
      float rho = lines[i][0], theta = lines[i][1];
      Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      pt1.x = cvRound(x0 + 1000*(-b));
      pt1.y = cvRound(y0 + 1000*(a));
      pt2.x = cvRound(x0 - 1000*(-b));
      pt2.y = cvRound(y0 - 1000*(a));
      line( im, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
  }
}

void hough_prob(Mat_<Vec3b> im) {
  Mat_<uint8_t> proc(im.rows,im.cols);
  Canny(im,proc,1,100); 
  vector<Vec4i> linesP; 
  HoughLinesP(proc,linesP,1,CV_PI/180,50,50,10);
  for(size_t i=0;i<linesP.size();i++) {
      Vec4i l = linesP[i];
      line(im, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,0), 3, LINE_AA);
  }
}
void hough_circles(Mat_<Vec3b> im, Mat_<uint8_t> grey) {
  vector<Vec3f> circles;

  GaussianBlur( grey, grey, Size(9, 9), 2, 2 );
  HoughCircles(grey, circles, HOUGH_GRADIENT, 2, grey.rows/4, 200, 100 );
  for( size_t i = 0; i < circles.size(); i++ ) {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // draw the circle center
      circle( im, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // draw the circle outlin e
      circle( im, center, radius, Scalar(0,0,255), 3, 8, 0 );
  }
}

void contours(Mat_<Vec3b> im, Mat_<uint8_t> grey) {
  vector<vector<Point> > contours;
  threshold(grey,grey,128,255,THRESH_OTSU);
  findContours(grey,contours,RETR_TREE,CHAIN_APPROX_SIMPLE);
  drawContours(im,contours,-1,Scalar(0,255,0));
}

Vec3b randomcolour() {
  return Vec3b(rand()%220+35,rand()%220+35,rand()%220+35);
}

void hierarchy(Mat_<Vec3b> im, Mat_<uint8_t> grey) {
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  threshold(grey,grey,128,255,THRESH_BINARY);
  findContours(grey,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE);
  int components=0, idx=0;
  Mat_<Vec3b> im1(im.rows,im.cols);
  im1=0;
  for( ; idx >= 0; idx = hierarchy[idx][0], components++ ) {
    srand(components);
    drawContours(im1,contours,idx,im(contours[idx][0].y,contours[idx][0].x),-1,8,hierarchy);
  }
  im1.copyTo(im);
}

void dowatershed(Mat_<Vec3b> im, Mat_<uint8_t> grey) {
  Mat_<int> markers(grey.rows,grey.cols);
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  threshold(grey,grey,128,255,THRESH_OTSU);
  findContours(grey,contours,hierarchy,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
  int components=0, idx=0;
  markers=-1;
  for( ; idx >= 0; idx = hierarchy[idx][0], components++ ) {
    drawContours(markers,contours,idx,components,-1,8,hierarchy);
  }
  watershed( im, markers );
  vector<Vec3b> colourTab(components);
//  colourTab.resize(components);
  for(int i = 0; i < components; i++ )
      colourTab.push_back(Vec3b(0,0,0));///randomcolour());

  for(int i = 0; i < markers.rows; i++ )
    for(int j = 0; j < markers.cols; j++ ) {
      int idx=markers(i,j);
      if(idx>=0) {
        if(colourTab[idx]==Vec3b(0,0,0))
          colourTab[idx]=im(i,j);
        im(i,j)=colourTab.at(idx);
      }
    }
}

int main(int argc, char** argv) {
    (void)argv[argc - 1];
    bool COLOUR=false;
    bool has_camera=true;
    VideoCapture cap;
    Mat_<Vec3b> frame,image;
    cap.open(0);
    if (!cap.isOpened()) {
        cout << "Failed to open camera, using static image" << endl;
        image=imread("../../../Images/baboon.jpg");
        image.copyTo(frame);
        has_camera=false;
    } else
    cout << "Opened camera" << endl;
    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Processed", WINDOW_NORMAL);
    namedWindow("Parameters",WINDOW_AUTOSIZE);
    createTrackbar("Param1","Parameters",&p1,255);
    createTrackbar("Param2","Parameters",&p2,255);
    resizeWindow("Parameters",512,128);
    if(has_camera) {
      cap.set(CAP_PROP_FRAME_WIDTH, 640);
      //   cap.set(CAP_PROP_FRAME_WIDTH, 960);
      //   cap.set(CAP_PROP_FRAME_WIDTH, 1600);
      //   cap.set(CAP_PROP_FRAME_HEIGHT, 480);
      //   cap.set(CAP_PROP_FRAME_HEIGHT, 720);
      //   cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
      cap >> frame;
    }
    Mat_<uint8_t> grey(frame.rows,frame.cols);
    Mat_<Vec3b> colour(frame.rows,frame.cols);
    Mat_<int16_t> lap(frame.rows,frame.cols);
    printf("frame size %d %d\n", frame.rows, frame.cols);
    resizeWindow("Original",frame.cols,frame.rows);
    resizeWindow("Processed",frame.cols,frame.rows);
    int key = 0;
    float fps = 0.0;
    Menu m=Menu("Menu",vector<string>({"Gaussian","Laplacian","Canny", "Hough Lines", "Hough Segments", "Hough Circles", "Contours", "Hierarchy", "Watershed", "Exit"}));
    while (1) {
        system_clock::time_point start = system_clock::now();
        if(has_camera)
          cap >> frame;
          else image.copyTo(frame);
        if (frame.empty()) break;
        ostringstream ss;
        ss<<setw(4)<<setprecision(4)<<setfill('0')<<fps<<"fps";
        string str=ss.str();
        putText(frame, str, Vec2i(12, 32), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 0), 2, 8);
        putText(frame, str, Vec2i(10, 30), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 255), 2, 8);
        Mat channels[3];
        if(COLOUR) {
            cvtColor(frame,colour,COLOR_BGR2HSV);    
            split(colour,channels);
            grey=channels[2];
        } else cvtColor(frame,grey,COLOR_BGR2GRAY);

      
        string selection=m.getselected();
        if(selection=="Gaussian") gaussian(grey);
        if(selection=="Laplacian") laplacian(grey);
        if(selection=="Hough Lines") hough_lines(frame);
        if(selection=="Hough Segments") hough_prob(frame);
        if(selection=="Hough Circles") hough_circles(frame,grey);
        if(selection=="Contours") contours(frame,grey);
        if(selection=="Hierarchy") hierarchy(frame,grey);
        if(selection=="Watershed") dowatershed(frame,grey);
        if(selection=="Canny") Canny(frame,grey,p1,p2);
        imshow("Original",frame);
        if(COLOUR) {
            channels[2]=grey;
            merge(channels,3,colour);
            cvtColor(colour,colour,COLOR_HSV2BGR);
            imshow("Processed", colour);
        } else {
            imshow("Processed", grey);
        }
        key = waitKey(1);
        if (key == 113 || key == 27 ) {return 0; } // either esc or 'q'
        system_clock::time_point end = system_clock::now();
        float mseconds = (end - start) / 1ms;
        fps = 1000.0f / mseconds;
    }
}
