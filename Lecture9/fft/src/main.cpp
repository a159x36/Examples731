#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "menu.hpp"

using namespace cv;
using namespace std;
using namespace chrono;

const int MAXFWHT_SIZE=1024;

// reverse the bits in n
int bit_reverse( int n, unsigned nbits ) {
    int r=0;
    for(;nbits--;n=n>>1) { 
        r=r<<1 | (n&1);
    }
    return r;
}

// work out sequency_permutation for fwht
void fwht_sequency_permutation(int perm[], unsigned order ) {
	int size=1<<order;
	for ( int i=0;i<size;i++) 
        perm[i] = bit_reverse((i >> 1) ^ i,order);
}

// A 1D Fast Walsh Hadamard Transform, n must be  power of 2
// l2n is log base 2 n, perm is an array of sequency permutations
void fwht1d(float *data, const int n, int l2n, int stride, int *perm, int scale) {
    float copy[MAXFWHT_SIZE];
    for(int i=0;i<n;i++) {
        copy[i]=data[i*stride];
    }
	for (int i = 0; i < l2n; ++i ) {
        int o=(1<<i);
		for (int j = 0; j < n; j += o*2) {
            for (int k = 0; k < o; ++k ) {
                float t=copy[j+k];
                float t1=copy[j+k+o];
                copy[j+k]=t+t1;
                copy[j+k+o]=t-t1;
            }
        }
	}
    for(int i=0;i<n;i++) {
        data[i*stride]=copy[perm[i]]/scale;
    }
}
// returns log base 2 the smallest power of 2 >= x 
int log2(int x) {
    int i=1;
    int n=0;
    while(i<x) {
        i=i<<1;
        n++;
    }
    return n;
}

// a 2d Fast Walsh Hadamard Transform
void fwht(Mat_<float> in, bool forward) {
    int ln2c=log2(in.cols);
    int ln2r=log2(in.rows);
    int cols=1<<ln2c;
    int rows=1<<ln2r;
    Mat_<float> out;
    copyMakeBorder(in, out, 0, rows-in.rows, 0, cols-in.cols, BORDER_CONSTANT,0);
    int perm[MAXFWHT_SIZE];
    fwht_sequency_permutation(perm,ln2c);
    for(int i=0;i<out.rows;i++) {
        fwht1d(&out(i,0), out.cols, ln2c, 1,perm, forward?out.cols:1);
    }
    fwht_sequency_permutation(perm,ln2r);
    for(int i=0;i<out.cols;i++) {
        fwht1d(&out(0,i), out.rows, ln2r, out.cols ,perm, forward?out.rows:1);
    }
    for(int i=0;i<in.rows;i++) 
        for(int j=0;j<in.cols;j++)
            in(i,j)=out(i,j); 
}
// main, uses webcam to get a colour image, encoded it using the dct, quantizes this
// then dequantizes it and decodes it using an idct. 
int main(int argc, char** argv) {
    bool COLOUR=true;
    bool WEBCAM=true;
    bool WEINER=true;
    bool HADAMARD=false;
    bool SHOW_FFT=false;


    (void)argv[argc - 1];
    VideoCapture cap;
    Mat_<Vec3b> frame;
    Mat_<Vec3b> image;
    image=imread("../../../../Images/clown.tif");
   // image=imread("../../../../Images/baboon.jpg");
    frame=image.clone();
    if(WEBCAM) {
#if __linux__
        cap.open(0,CAP_V4L2);
#else
        cap.open(0);
#endif
        if (!cap.isOpened()) {
            cout << "Failed to open camera" << endl;
            return 0;
        }
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        //   cap.set(CAP_PROP_FRAME_WIDTH, 960);
        //   cap.set(CAP_PROP_FRAME_WIDTH, 1600);
        //   cap.set(CAP_PROP_FRAME_HEIGHT, 480);
       //    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
        //   cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
        cap >> frame;
    }
    
    Menu m("Deconvolution",{"Weiner Filter","Pseudoinverse","Colour","Greyscale","Webcam","Static Image","FWHT","FFT","Exit"});
    namedWindow("WebCam", WINDOW_FREERATIO);
    namedWindow("Result", WINDOW_FREERATIO);

    printf("frame size %d %d\n", frame.rows, frame.cols);
    resizeWindow("WebCam",frame.cols,frame.rows);
    resizeWindow("Result",frame.cols,frame.rows);

    int key = 0;
    float fps = 0.0;

    Mat_<Vec2f> complex1;
    Mat_<Vec2f> fft;
    Mat_<Vec3b> colour;
    Mat_<uint8_t> channels[3];
    Mat_<float> planes[2];

    int noise=108;
    int cutoff=138;//frame.cols/4;
    int focus=36;
    createTrackbar( "cutoff", "WebCam", &cutoff, (frame.rows+frame.cols)*2);
    createTrackbar( "noise", "WebCam", &noise, 1000);
    createTrackbar( "focus", "WebCam", &focus, 255);
    while (1) {
        system_clock::time_point start = system_clock::now();
        if(WEBCAM) {
            cap.set(CAP_PROP_FOCUS,focus);
            cap >> frame;
        } else {
            frame=image.clone();
        }
        if (frame.empty()) break;

        if(COLOUR) {
            split(frame,channels);
        } else {
            cvtColor(frame,colour,COLOR_BGR2YUV);    
            split(colour,channels);
        }
        
        
        int nchannels=1;
        if(COLOUR) nchannels=3;

        if(HADAMARD) {
            Mat_<float> hadamard(frame.size());
            for(int ch=0;ch<nchannels;ch++) {
            channels[ch].convertTo(hadamard,CV_32FC1);
            float rad=(cutoff/100.0)/5;
            fwht(hadamard,true);
            for(int i=0;i<hadamard.rows;i++) {
                for(int j=0;j<hadamard.cols;j++) {
                    float d;
                    //d=rad*min(i,j)/10;
                    d=rad*(hypot(hadamard.rows,hadamard.cols)-hypot(hadamard.rows-i,hadamard.cols-j))/100;
                    float G;
                    G=exp(-d*d);
                    hadamard(i,j)*=G;
                }
            }
            imshow("Hadamard",hadamard+0.5);
            fwht(hadamard,false);
            hadamard.convertTo(channels[ch],CV_8UC1);
            }
        } else { // use the FFT
            for(int ch=0;ch<nchannels;ch++) {
                planes[0]=Mat_<float>(channels[ch].clone());
                planes[1]=Mat::zeros(frame.size(),CV_32F);
                merge(planes, 2, complex1);
                dft(complex1,fft);
                if(ch==0 && SHOW_FFT) {
                    split(fft,planes);
                    imshow("FFT",planes[0]/(256.0*256.0));
                }
                float rad=(cutoff/100.0+1)/5;
                for(int i=0;i<frame.rows;i++) {
                    for(int j=0;j<frame.cols;j++) {

                        int i1=i<frame.rows/2?i:frame.rows-i;
                        int j1=j<frame.cols/2?j:frame.cols-j;
                        float d=rad*hypot(i1,j1)/10;
                        float H,G;
                        if(d!=0)
                            H=sin(d)/(d);      
                        else
                            H=1;  
                        if(WEINER) {
                            //  G = H / ((|H|^2)+N)
                            float denom=H*H+noise/10000.0;
                            G=H/denom;
                        } else {
                            if(H>noise/1000.0)
                                G=1/H;
                            else
                                G=0;
                        }
                        fft(i,j)=fft(i,j)*G;
                    }
                }
                
                idft(fft,complex1,DFT_SCALE);

                split(complex1,planes);
                planes[0].convertTo(channels[ch],CV_8UC1);
            }
        }
    
        ostringstream ss;
        ss<<setw(4)<<setprecision(4)<<setfill('0')<<fps<<"fps ";
        string str=ss.str();
        putText(frame, str, Vec2i(12, 32), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 0), 2, 8);
        putText(frame, str, Vec2i(10, 30), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 255), 2, 8);
        imshow("WebCam", frame);
        if(!COLOUR) {
            imshow("Result",channels[0]);
        } else {
            merge(channels,3,colour);
            imshow("Result",colour);
        }
        
        
        key = waitKey(1);
        if (key == 113 || key == 27) return 0;  // either esc or 'q'
        string entry=m.getselected();
        if(entry=="Weiner Filter") {
            WEINER=true;
        }
        if(entry=="Pseudoinverse") {
            WEINER=false;
        }
        if(entry=="Colour") {
            COLOUR=true;
        }
        if(entry=="Greyscale") {
            COLOUR=false;
        }
        if(entry=="Webcam") {
            WEBCAM=true;
        }
        if(entry=="Static Image") {
            WEBCAM=false;
        }
        if(entry=="FWHT") {
            HADAMARD=true;
        }
        if(entry=="FFT") {
            HADAMARD=false;
        }
        system_clock::time_point end = system_clock::now();
        float mseconds = (end - start) / 1ms;
        fps = 1000.0f / mseconds;
    }
}
