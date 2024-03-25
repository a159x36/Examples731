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
    static const int ROWHEIGHT=32;
    const Vec3b BG=Vec3b(0xbf,0xbe,0xbb);
    const Vec3b GREY=Vec3b(0xef,0xf0,0xf1);
    const Vec3b BLUE=Vec3b(0xe9,0xae,0x3d);
    const Vec3b BLACK=Vec3b(0,0,0);
    const Vec3b WHITE=Vec3b(0xff,0xff,0xff);
    public:
    Menu(string title,vector<string> items) {
        menutitle=title;
        menuitems=items;
        image=Mat_<Vec3b>(menuitems.size()*ROWHEIGHT,width);
        namedWindow(menutitle);
        setMouseCallback(menutitle,onmouse,(void *)this);
        drawmenu();
    }
    static void onmouse(int event, int , int y, int , void *userdata) {
        Menu *m=(Menu *)userdata;
        if(event==0) m->sethighlighted(y/ROWHEIGHT);
        if(event==1) m->setselected(y/ROWHEIGHT);
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
        image=BG;
        for(int i=0;i<(int)menuitems.size();i++) {
            Vec3b bgcol=GREY;
            Vec3b fgcol=BLACK;
            if(i==highlighted) {
                bgcol=BLUE;
                fgcol=WHITE;
            }
            if(i==selected) {
                bgcol=BLACK;
                fgcol=WHITE;
            }
            rectangle(image,Rect(1,i*ROWHEIGHT,width-2,ROWHEIGHT),bgcol,FILLED);
            putText(image, menuitems[i], Vec2i(12, 20 + i*ROWHEIGHT), FONT_HERSHEY_PLAIN, 1,
                fgcol,1,LINE_AA);
        }
        imshow(menutitle,image);
    }
};

