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
	float diff, visc;
	float force, source;
	int flames=3;
	bool changed=true;
	Buffer<float> u, v, u0, v0, dens, dens0;

public: 
	FluidSim( int width, int height, float diff, float visc, float force, float source) : 
		width(width), height(height), diff(diff), visc(visc), force(force), source(source) {
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
	void step(double dt) {
		if(changed) {
			for(int i=0;i<flames;i++) {
				int xp = width / (flames + 1) * (i + 1);
				dens0(xp,height-10)=source;
				v0(xp,height-10) = 0;
				u0(xp,height-10) = -force;
			}
			u0.set_host_dirty();
			v0.set_host_dirty();
			dens0.set_host_dirty();
			changed=false;
		}
		halide_vel_step(u, v, u0, v0, visc, dt, u, v);
		halide_dens_step(dens,dens0,u,v,diff,dt, dens);
	}
	int clamp(int i, int min, int max) {
		if(i<min) return min;
		return (i>max)?max:i;
	}
	void add_dens(int x,int y) {
		dens.copy_to_host();
		for(int i=x-1;i<=x+1;i++)
			for(int j=y-1;j<=y+1;j++) {
				int ii=clamp(i,0,width-1);
				int jj=clamp(j,0,height-1);
				dens(ii,jj)=500;
			}
		dens.set_host_dirty();
	}
	void add_vel(int x,int y, int dx, int dy) {
		u.copy_to_host();
		v.copy_to_host();
		for(int i=x-1;i<=x+1;i++)
			for(int j=y-1;j<=y+1;j++) {
				int ii=clamp(i,0,width-1);
				int jj=clamp(j,0,height-1);
				u(ii,jj)=dy*force;
				v(ii,jj)=dx*force;
			}
		u.set_host_dirty();
		v.set_host_dirty();
	}

	void get_bitmap(Buffer<> rgb_h) {
		halide_bitmap(dens,u,v,rgb_h);
		rgb_h.copy_to_host();
	}
	static void setDiffusion(int pos, void *data) {
		FluidSim *fsim=(FluidSim *)data;
		fsim->diff=pos/10000000.0;
	}
	static void setViscosity(int pos, void *data) {
		FluidSim *fsim=(FluidSim *)data;
		fsim->visc=pos/10000000.0;
	}
	static void setForce(int pos, void *data) {
		FluidSim *fsim=(FluidSim *)data;
		fsim->force=pos*0.5f;
		fsim->changed=true;
	}
};



double get_time() {
	struct timespec time;
	clock_gettime(CLOCK_REALTIME,&time);
	return time.tv_sec+time.tv_nsec/1e9;
}

static void onmouse(int event, int x, int y, int , void *userdata) {
	FluidSim *fs=(FluidSim *)userdata;
	static int lx,ly;
	static bool ld=false;
	static bool rd=false;
	if(event==EVENT_RBUTTONDOWN) rd=true;
	if(event==EVENT_RBUTTONUP) rd=false;
	if(event==EVENT_LBUTTONDOWN) ld=true;
	if(event==EVENT_LBUTTONUP) ld=false;

	if(rd)
		fs->add_dens(x,y);
	if(ld)
		fs->add_vel(x,y,x-lx,y-ly);
	lx=x;
	ly=y;
}
int main ( int argc, char ** argv ) {
	(void)argc,(void)argv;
	double fps;
	int frames=0;
	double lasttime=0,lasttime1=0;
	Mat_<Vec4b> rgb(512,512);
	bool FIXED_TIMESTEP=true;
	double timestep=0.1;
	FluidSim fsim(rgb.cols,rgb.rows,0.0000023f,0.0000001,10.0f,200.0f);
	

	namedWindow("Fluid",WINDOW_NORMAL | WINDOW_GUI_NORMAL);
	resizeWindow("Fluid",rgb.cols,rgb.rows + 100);
	setMouseCallback("Fluid",onmouse,(void *)&fsim);
	createTrackbar("Diffusion","Fluid",NULL,100,fsim.setDiffusion, &fsim);
	createTrackbar("Viscosity","Fluid",NULL,100,fsim.setViscosity, &fsim);
	createTrackbar("Force","Fluid",NULL,100,fsim.setForce, &fsim);

	Buffer<uint32_t> rgb_h=Buffer<uint32_t>((uint32_t *)rgb.data, rgb.cols, rgb.rows);
	double time,newtime,oldtime=get_time();
	while(1) {
		time=get_time();
		fsim.step(FIXED_TIMESTEP?timestep:(time-oldtime)*10);
		oldtime=time;
		frames++;
		if(time-lasttime1>1) {
			lasttime1=time;
			newtime=get_time();
			fps=1.0 / (newtime-time);
			cout<<"Fps:"<< 1.0 / (newtime-time) <<","<<frames<<endl;
			frames=0;
		}
		if(time-lasttime>1/30.0) {
			lasttime=time;
			fsim.get_bitmap(rgb_h);
			ostringstream ss;
        	ss<<setprecision(4)<<fps<<"fps ";
			putText(rgb,ss.str(),Point(16,16),FONT_HERSHEY_PLAIN, 1, Vec3i(255, 255, 255));
			imshow("Fluid",rgb);
			int k=pollKey();
			if(k=='q') {
				exit(1);
			}
		}
		
	}

	exit ( 0 );
}
