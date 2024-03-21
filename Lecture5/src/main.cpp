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

void showinwindow(string windowname,Mat im) {
    namedWindow(windowname, WINDOW_NORMAL);
    imshow(windowname,im);
    if(im.rows>1024)
        resizeWindow(windowname,im.cols/2,im.rows/2);
    else
        resizeWindow(windowname,im.cols,im.rows);
}

void hdr_process(string normalname, string undername,string overname) {
    
    Mat_<Vec3b> normal=imread("../../"+normalname);
    Mat_<Vec3b> over=imread("../../"+overname);
    Mat_<Vec3b> under=imread("../../"+undername);
    

    cout<<"Images Read"<<endl;

    Mat_<Vec3f> merged;//(normal.rows,normal.cols);
    Mat_<Vec3f> toned;//(normal.rows,normal.cols);

    vector<Mat> images({normal,over,under});

    // you can use this for simple alignment but it doesn't work for rotated images
    //Ptr<AlignMTB> align_mtb=createAlignMTB();
    //vector<Mat> aligned;
    //align_mtb->process(images,images);

    Ptr<SIFT> orb=SIFT::create(5000);
    Ptr<DescriptorMatcher> bfm=BFMatcher::create(NORM_L2,true);
    Mat desc0, desc1, desc2;
    vector<KeyPoint> keypoints0,keypoints1,keypoints2;
    orb->detectAndCompute(images[0],noArray(),keypoints0,desc0);
    orb->detectAndCompute(images[1],noArray(),keypoints1,desc1);
    orb->detectAndCompute(images[2],noArray(),keypoints2,desc2);
    vector<DMatch> dm;
    bfm->match(desc1,desc0,dm);
    vector<Point2f> kpq,kpt;
    for( DMatch d:dm ) { 
        kpq.push_back(keypoints1[ d.queryIdx ].pt);
        kpt.push_back(keypoints0[ d.trainIdx ].pt);
    }
    Mat H=findHomography(kpq,kpt,RANSAC,9.0,noArray(),5000);
    Mat warped1,warped2;
    warpPerspective(images[1],warped1,H,Size(normal.cols,normal.rows),WARP_FILL_OUTLIERS,BORDER_CONSTANT,Vec3b(255,255,255));
    dm.clear();
    bfm->match(desc2,desc0,dm);

    vector<Point2f> kpq1,kpt1;
    for( DMatch d:dm ) { 
        kpq1.push_back(keypoints2[ d.queryIdx ].pt);
        kpt1.push_back(keypoints0[ d.trainIdx ].pt);
    }
    Mat H1=findHomography(kpq1,kpt1,RANSAC,9.0,noArray(),5000);

    warpPerspective(images[2],warped2,H1,Size(normal.cols,normal.rows),WARP_FILL_OUTLIERS,BORDER_CONSTANT,Vec3b(255,255,255));
    cout<<H<<endl;
    cout<<H1<<endl;

    showinwindow("Under",under);
    showinwindow("Over",over);
    showinwindow("Normal",normal);

    images={normal,warped1,warped2};
    
    Mat response;
 //   vector<float> times({1.0,2.0,0.5});
 //   Ptr<CalibrateRobertson> CalibrateRobertson = createCalibrateRobertson();
 //   CalibrateRobertson->process(images, response, times);
 //   Ptr<MergeDebevec> merge=createMergeDebevec();
 //   merge->process(images,merged,times,response);
   
    Ptr<MergeMertens> merge_mer=createMergeMertens();
    merge_mer->process(images,merged);

    Ptr<Tonemap> tone=createTonemap(0.6);
    tone->process(merged,toned);

    showinwindow("Merged",merged);
    showinwindow("Toned",toned);
}


int main(int argc, char** argv) {
    (void)argv[argc - 1];
    Menu m("HDR",{"Bridge","Colour Card","Church","Massey","Garden","Exit"});
    string done="";
    while(1) {
        string selection=m.getselected();
        if(done!=selection) {
            if(selection=="Bridge") hdr_process("br_normal.png","br_under.png","br_over.png");
            if(selection=="Colour Card") hdr_process("cc_normal.png","cc_under.png","cc_over.png");
            if(selection=="Church") hdr_process("ch_normal.png","ch_under.png","ch_over.png");
            if(selection=="Massey") hdr_process("cp_normal.png","cp_under.png","cp_over.png");
            if(selection=="Garden") hdr_process("yu_normal.png","yu_under.png","yu_over.png");

            done=selection;
        }
        int key=waitKey(1);
        if (key == 113 || key == 27 ) {return 0; } // either esc or 'q'
    }
}
