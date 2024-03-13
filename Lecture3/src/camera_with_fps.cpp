#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace chrono;

/*
int thresh=128;

void mouse(int event, int x, int y, int flags, void *userdata) {
    cout<<event<<":"<<x<<","<<y<<endl;
}

int main(int argc, char** argv) {
    (void)argv[argc - 1];
    
    Mat_<int> m(10,10);
    m=-1;
    cout<<m<<endl;
    float a[2][3]={{1,2,3},{4,5,6}};
    Mat_<float> ma(2,3,(float *)a);
    cout<<ma<<endl;
    Mat_<float> identity(2,2);
    identity<<1,0,0,1;
    cout<<identity<<endl;
    
    float a[2][3]= { {1 , 2 , 3 } , { 4 , 5 , 6 } } ;
    float b[2][3]= { {7 , 8 , 9 } , { 0 , 1 , 2 } } ;
    float c[3][2]= { {7 , 8 } , { 9 , 0 } , { 1 , 2 } } ;
    float d[2][2]= { {1 , 2 }, { 2 , 5 } } ;
    Mat_<float> A(2,3,(float *)a);
    Mat_<float> B(2,3,(float *)b);
    Mat_<float> C(3,2,(float *)c);
    Mat_<float> D(2,2,(float *)d);

    Mat r;

    r=A+B;
    cout<<r<<endl;
    r=A*C;
    cout<<r<<endl;
    transpose(A,r);
    cout<<r<<endl;
    float det=determinant(D);
    cout<<det<<endl;
    Mat INV;
    invert(D,INV);
    cout<<INV<<endl;
    r=D*INV;
    cout<<r<<endl;
    exit(0);
    
    Mat_<Vec3b> imagecolour;
    Mat_<uint8_t> imagegrey;
    namedWindow("Threshold");
    createTrackbar("Threshold","Threshold",&thresh,255);
    imagecolour=imread("../../baboon.jpg");
    imshow("Colour",imagecolour);
    cvtColor(imagecolour,imagegrey,COLOR_BGR2GRAY);
    imshow("Grey",imagegrey);
    Mat_<uint8_t> th(imagegrey.rows,imagegrey.cols);
    setMouseCallback("Threshold",mouse);
    while(1) {
        for(int i=0;i<imagegrey.rows;i++) {
            for(int j=0;j<imagegrey.cols;j++) {
                if(imagegrey(i,j)>thresh) th(i,j)=255; else th(i,j)=0;
            }
        }
        imshow("Threshold",th);
        waitKey(1);
    }
 
}
*/

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
        //cout<<event<<","<<y<<","<<flags<<endl;
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

Mat_<uint8_t> negative(Mat_<uint8_t> im) {
  return 255-im;
}

Mat_<uint8_t> thresh(Mat_<uint8_t> im,int v) {
  Mat thr;
//  inRange(im,{v},{255},thr);
  threshold(im,thr,v,255,THRESH_OTSU);
  return thr;
}

uint8_t im_cl(Mat_<uint8_t> im, int r,int c) {
  return im(clamp(r,0,im.rows-1),clamp(c,0,im.cols-1));
}

Mat_<uint8_t> enhance(Mat_<uint8_t> im) {
  Mat_<uint8_t> im1(im.rows,im.cols);
  for(int r=0;r<im.rows;r++) {
    for(int c=0;c<im.cols;c++) {
      im1(r,c)=clamp(im(r,c)*5-im_cl(im,r-1,c)-im_cl(im,r+1,c)-
        im_cl(im,r,c-1)-im_cl(im,r,c+1),0,255);
    }
  }
  return im1;
}

Mat_<uint8_t> stretch(Mat_<uint8_t> im) {
  float mean=0,var=0,sd,c_str=1.5;
  for(int r=0;r<im.rows;r++) {
    for(int c=0;c<im.cols;c++) {
      mean+=im(r,c);
      var+=im(r,c)*im(r,c);
    }
  }
  int n=im.rows*im.cols;
  mean=mean/n;
  var=var/n-(mean*mean);
  sd=sqrt(var);
  Mat_<uint8_t> im1(im.rows,im.cols);
  for(int r=0;r<im.rows;r++) {
    for(int c=0;c<im.cols;c++) {
      float p=(255.0*(im(r,c)-mean+c_str*sd))/(2*c_str*sd);
      if(p>255) p=255;
      if(p<0) p=0;
      im1(r,c)=p;
    }
  }
  return im1;
}

Mat_<uint8_t> convolve(Mat_<uint8_t> im) {
  Mat kernel = (Mat_<float>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
 //Mat kernel = (Mat_<float>(3,3) << 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
 //   1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
 //   1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f);
  transpose(kernel,kernel);
  Mat_<uint8_t> im1(im.rows,im.cols);
  filter2D(im,im1,-1,kernel,Point(-1,-1),0);
  return im1;
}

Mat_<uint8_t> sobel(Mat_<uint8_t> im) {
  Mat kernel = (Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
  Mat_<uint8_t> im1(im.rows,im.cols);
  filter2D(im,im1,-1,kernel,Point(-1,-1),0);
  return im1;
}

Mat_<uint8_t> zc(Mat_<int16_t> im) {
    int T=100000;
    Mat_<uint8_t> im1(im.rows,im.cols);
    im1=0;
    for(int i=0;i<im.rows-1;i++) {
        for(int j=0;j<im.cols-1;j++) {
            int a=im(i,j);
            int b=im(i,j+1);
            int c=im(i+1,j+1);
            int d=im(i+1,j);
            if(a*b<=-T || a*d<=-T || a*c<=-T)
                im1(i,j)=255;
        }
    }
    return im1;
}

int main(int argc, char** argv) {
    (void)argv[argc - 1];
    bool COLOUR=false;
    VideoCapture cap;
    Mat_<Vec3b> frame;
    cap.open(0);
    if (!cap.isOpened()) {
        cout << "Failed to open camera" << endl;
        return 0;
    }
    cout << "Opened camera" << endl;
    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Processed", WINDOW_NORMAL);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    //   cap.set(CAP_PROP_FRAME_WIDTH, 960);
    //   cap.set(CAP_PROP_FRAME_WIDTH, 1600);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
    cap >> frame;
    Mat_<uint8_t> grey(frame.rows,frame.cols);
    Mat_<uint8_t> grey1(frame.rows,frame.cols);
    Mat_<int16_t> lap(frame.rows,frame.cols);
    printf("frame size %d %d\n", frame.rows, frame.cols);
    resizeWindow("Original",frame.cols,frame.rows);
    resizeWindow("Processed",frame.cols,frame.rows);
    int key = 0;
    float fps = 0.0;
    cv::Ptr<cv::CLAHE> clahe=createCLAHE(2.0,Size(4,4));
    Menu m=Menu("Menu",vector<string>({"Negative", "Threshold", "Stretch", "Enhance", "Sobel", "Equalize", "CLAHE", "Edges", "Rotate", "Exit"}));
    while (1) {
        system_clock::time_point start = system_clock::now();
        cap >> frame;
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
            cvtColor(frame,frame,COLOR_BGR2HSV);    
            split(frame,channels);
            grey=channels[2];
        } else cvtColor(frame,grey,COLOR_BGR2GRAY);

        imshow("Original", grey);
        string selection=m.getselected();
        if(selection=="Negative") grey=negative(grey);
        if(selection=="Threshold") grey=thresh(grey,128);
        if(selection=="Stretch") grey=stretch(grey);
        if(selection=="Enhance") grey=enhance(grey);
        if(selection=="Sobel") grey=sobel(grey);
        if(selection=="Equalize") equalizeHist(grey,grey);
        if(selection=="CLAHE") clahe->apply(grey,grey);
        if(selection=="Edges") {
            float sz=5;
            Laplacian(grey,lap,CV_16S,sz, 1);
            grey=zc(lap);
        }
        if(selection=="Rotate") warpAffine(grey,grey,getRotationMatrix2D(Point(grey.cols/2,grey.rows/2),-30,1),grey.size());
        if(COLOUR) {
            channels[2]=grey;
            merge(channels,3,frame);
            cvtColor(frame,frame,COLOR_HSV2BGR);
            imshow("Processed", frame);
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
