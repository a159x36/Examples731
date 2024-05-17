

#include <chrono>
#include <ctime>
#include <iostream>

#include "halide_dens_step.h"
#include "halide_vel_step.h"
#include "halide_bitmap.h"
#include "HalideBuffer.h"

#include <opencv2/opencv.hpp>

using namespace Halide::Runtime;
using namespace cv;
using namespace std;



class FluidSim {

private:
	int width, height;
	float dt, diff, visc;
	float force, source;
	int flames=3;

	Buffer<float> u, v, u0, v0, dens, dens0;

public: 
	FluidSim( int width, int height, float dt, float diff, float visc, float force, float source) : 
		width(width), height(height), dt(dt), diff(diff), visc(visc), force(force), source(source) {
		int r=height,c=width;

		u=Buffer<float>(c,r);
		u0=Buffer<float>(c,r);
		v=Buffer<float>(c,r);
		v0=Buffer<float>(c,r);
		dens=Buffer<float>(c,r);
		dens0=Buffer<float>(c,r);	
	}

	~FluidSim ( void ) {
			u.deallocate();
			v.deallocate();
			u0.deallocate();
			v0.deallocate();
			dens.deallocate();
			dens0.deallocate();
	}
	void step(void) {
		for(int i=0;i<flames;i++) {
			int xp = width / (flames + 1) * (i + 1);
			dens0(xp,height-10)=source;
			v0(xp,height-10) = 0;
			u0(xp,height-10) = -force;
		}
		u0.set_host_dirty();
		v0.set_host_dirty();
		dens0.set_host_dirty();
		halide_vel_step(u, v, u0, v0, visc, dt, u, v);
		halide_dens_step(dens,dens0,u,v,diff,dt, dens);
	}
	void get_bitmap(Buffer<unsigned> rgb_h) {
		halide_bitmap(dens,u,v,rgb_h);
		rgb_h.copy_to_host();
	}
};



double get_time() {
	struct timespec time;
	clock_gettime(CLOCK_REALTIME,&time);
	return time.tv_sec+time.tv_nsec/1e9;
}


int main ( int argc, char ** argv )
{
	double fps;
	int frames=0;
	double lasttime=0,lasttime1=0;
	Mat_<Vec4b> rgb(512,512);
	FluidSim fsim(rgb.cols,rgb.rows,0.1f,0,0,5.0f,100.0f);
	
	namedWindow("Fluid",WINDOW_FREERATIO);
	resizeWindow("Fluid",rgb.cols,rgb.rows);

	Buffer<uint32_t> rgb_h=Buffer<uint32_t>((uint32_t *)rgb.data, rgb.cols, rgb.rows);
	while(1) {
	//	cout<<"frame:"<<frames<<endl;
		double time,time1,time2,newtime;
		time=get_time();
		fsim.step();
		time2=get_time();
		frames++;
		if(time-lasttime1>1) {
			
			lasttime1=time;
			newtime=get_time();
			fps=1.0 / (newtime-time);
			printf("Fps:%f %f %d\n", 1.0 / (newtime-time), 1.0/(time2-time),frames);
			frames=0;
		}
		if(time-lasttime>1/60.0) {
			fsim.get_bitmap(rgb_h);
			ostringstream ss;
        	ss<<setprecision(4)<<fps<<"fps ";
			putText(rgb,ss.str(),Point(16,16),FONT_HERSHEY_PLAIN, 1, Vec3i(255, 255, 255));
			imshow("Fluid",rgb);
			lasttime=time;
		}
		int k=waitKey(1);
		if(k=='q') {
			exit(1);
		}
	}

	exit ( 0 );
}
