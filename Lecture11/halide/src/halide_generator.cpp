#include <Halide.h>
using namespace Halide;
using namespace std;
bool gpu=false;
bool auto_sch=false;

Var x("x"), y("y"),xi("xi"), yi("yi"), yo("yo"),xo("xo");

void good_schedule(vector <Func> v) {
    if(auto_sch)
        return;
    for(Func f:v) {
        if (gpu)
            f.compute_root().gpu_tile(x,y, xo,yo,xi, yi, 16,16);
        else
            f.compute_root().tile(x, y, xi, yi, 8, 8).parallel(y).vectorize(xi,8);
    }
}

FuncRef lin_solve(Func in, Func x0, Expr a, Expr c, Expr w, Expr h, int num_steps= 4) {
    Func x1=BoundaryConditions::constant_exterior(x0, 0, 1, w-2, 1, h-2);
    Func fin=BoundaryConditions::constant_exterior(in, 0, 1, w-2, 1, h-2);
    Expr invc=1.0f/c;

    vector<Func> f(num_steps+1);
    f[0]=fin;
    // uses Jacobi rather than Gauss-Seidel so converges a bit slower
    for(int k=1 ; k<=num_steps ; k++) {
        f[k](x, y) = invc*(x1(x, y) + a *(f[k - 1](x - 1, y) + f[k - 1](x + 1, y)
                                        + f[k - 1](x, y - 1) + f[k - 1](x, y + 1)));
        if(k%2==0 || k==num_steps) {
            if(!auto_sch) {
                if (gpu) {
                    good_schedule({f[k]});//,f[k - 1]});
                } else {
                    f[k].tile(x, y, xi, yi, 8, 8);
                    f[k].compute_root().parallel(y).vectorize(xi, 8);
                    f[k - 1].compute_at(f[k], y).vectorize(x, 8);
                }
            }
        }
    }
    return f[num_steps](x,y);
}

FuncRef diffuse(Func dens, Func dens0, Expr diff, Expr dt, Expr w, Expr h ) {
    Expr a = dt*diff*w*h;
    return lin_solve(dens, dens0, a, 1.0f + 4.0f * a, w, h, 4);
}

FuncRef advect (Func d0, Func u, Func v, Expr dt, Expr w, Expr h) {
    Func advected{"advected"};
    Expr dt0 = dt*200;
    Expr xx = clamp(x-dt0*v(x,y), 0.5f, w-0.5f);
    Expr yy = clamp(y-dt0*u(x,y), 0.5f, h-0.5f);
    Expr i0=cast<int>(xx);
    Expr j0=cast<int>(yy);
    Expr i1=i0+1;
    Expr j1=j0+1;
    Expr s1 = xx-i0;
    Expr t1 = yy-j0;
    advected(x,y) = lerp(lerp(d0(i0,j0),d0(i0,j1),t1),lerp(d0(i1,j0),d0(i1,j1),t1),s1); 
    return advected(x,y);
}

void project (Func u, Func v, Func uu, Func vv, Expr w, Expr h ) {
    Func div("div"),p{"p"},zero;
    Expr m=-1.0f/h;
    div(x,y) = m*(v(x+1,y)-v(x-1,y)+u(x,y+1)-u(x,y-1));
    zero(x,y) = 0.f;
    p(x,y)= lin_solve(zero, div, 1.0f, 4.0f, w, h, 4);
    vv(x,y) = v(x,y) - 0.5f*w*(p(x+1,y)-p(x-1,y));
    uu(x,y) = u(x,y) - 0.5f*h*(p(x,y+1)-p(x,y-1));
    good_schedule({div});
}

class DensStepGenerator : public Halide::Generator<DensStepGenerator> {
public:
    Input <Buffer<float>> dens{"dens", 2};
    Input <Buffer<float>> dens_prev{"dens_prev", 2};
    Input <Buffer<float>> u{"u", 2};
    Input <Buffer<float>> v{"v", 2};
    Input <float> diff{"diff"};
    Input <float> dt{"dt"};
    Output <Buffer<float>> output{"output", 2};

    Func src_added{"src_added"};
    Func diffused{"diffused"};
    void generate() {
        gpu = get_target().has_gpu_feature() || get_target().has_feature(Target::OpenGLCompute);
        auto_sch = using_autoscheduler();
        Expr w = dens.width();
        Expr h = dens.height();
        src_added(x,y) = dens(x,y) + dens_prev(x,y) * dt;
        diffused(x,y) = diffuse(dens_prev, src_added, diff, dt, w, h);
        output(x, y) = advect(diffused, u, v, dt, w, h);
    }

    void schedule() {
       // int parallel_task_size = 8;
       // int vector_width = 4;
        if (using_autoscheduler()) {
            u.set_estimates({{0, 512},{0, 786}});
            v.set_estimates({{0, 512},{0, 786}});
            dens.set_estimates({{0, 512},{0, 786}});
            dens_prev.set_estimates({{0, 512},{0, 786}});
            output.set_estimates({{0, 512},{0, 786}});
            dt.set_estimate(0.1f);
            diff.set_estimate(0.00001f);
        } else {
            good_schedule({output});
        }
    }

};

class VelStepGenerator : public Halide::Generator<VelStepGenerator> {
public:
    Input <Buffer<float>> u{"u", 2};
    Input <Buffer<float>> v{"v", 2};
    Input <Buffer<float>> u0{"u0", 2};
    Input <Buffer<float>> v0{"v0", 2};
    Input<float> visc{"visc"};
    Input<float> dt{"dt"};
    Output<Buffer<float>> outputu{"outputu", 2};
    Output<Buffer<float>> outputv{"outputv", 2};

    Func uu{"uu"}, vv{"vv"}, diffU{"diffU"}, diffV{"diffV"};
    Func au0{"au0"}, av0{"av0"}, adU{"adU"}, adV{"adV"};

    void generate() {
        gpu = get_target().has_gpu_feature() || get_target().has_feature(Target::OpenGLCompute)|| get_target().has_feature(Target::Vulkan) ;
        auto_sch = using_autoscheduler();
        Expr w = u.width();
        Expr h = u.height();
        uu(x,y) = u(x,y) + u0(x,y) * dt;
        vv(x,y) = v(x,y) + v0(x,y) * dt;
        diffU(x,y) = diffuse(u0, uu, visc, dt, w, h);
        diffV(x,y) = diffuse(v0, vv, visc, dt, w, h);
        project(diffU, diffV, au0, av0, w, h);
        adU(x,y) = advect(au0, au0, av0, dt, w, h);
        adV(x,y) = advect(av0, au0, av0, dt, w, h);
        project(adU, adV, outputu, outputv, w, h);
    }

    void schedule() {
        if (using_autoscheduler()) {
            u.set_estimates({{0, 512},{0, 786}});
            v.set_estimates({{0, 512},{0, 786}});
            u0.set_estimates({{0, 512},{0, 786}});
            v0.set_estimates({{0, 512},{0, 786}});
            outputu.set_estimates({{0, 512},{0, 786}});
            outputv.set_estimates({{0, 512},{0, 786}});
            dt.set_estimate(0.1);
            visc.set_estimate(0.00001);
        } else {
            good_schedule({ uu, vv, au0, av0, adU, adV, outputu, outputv});
            if(!gpu) {
                adU.compute_with(adV, x);
                uu.compute_with(vv, x);
                au0.compute_with(av0, x);
                outputu.compute_with(outputv, x);
            } else {
                adU.compute_with(adV, xo);
                uu.compute_with(vv, xo);
                au0.compute_with(av0, xo);
                outputu.compute_with(outputv, xo);
            }
        }
    }
};


class BitmapGenerator : public Halide::Generator<BitmapGenerator> {
public:
    Input <Buffer<float>> d{"d", 2};
    Input <Buffer<float>> u{"u", 2};
    Input <Buffer<float>> v{"v", 2};
    Output<Buffer<unsigned>> output{"output", 2};

    void generate() {
        gpu = get_target().has_gpu_feature() || get_target().has_feature(Target::OpenGLCompute);
        Expr r =cast<unsigned>(clamp((d(x , y ) * 255), 0, 255));
        Expr g = cast<unsigned>(clamp((u(x , y ) * -2000) + 128, 0, 255));
        Expr b = cast<unsigned>(clamp((v(x , y ) * 2000) + 128, 0, 255));
        output(x, y) = cast<unsigned>((r<<16) | (g << 8) | (b ) | Expr(0xff000000));
    }

    void schedule() {
        if (using_autoscheduler()) {
          d.set_estimates({{0, 512},{0, 786}});
          u.set_estimates({{0, 512},{0, 786}});
          v.set_estimates({{0, 512},{0, 786}});
          output.set_estimates({{0, 512},{0, 786}});
        } else
            good_schedule({output});
    }
};
HALIDE_REGISTER_GENERATOR(DensStepGenerator,halide_dens_step)
HALIDE_REGISTER_GENERATOR(VelStepGenerator,halide_vel_step)
HALIDE_REGISTER_GENERATOR(BitmapGenerator,halide_bitmap)


