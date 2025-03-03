#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ctime>

//#include <unordered_set>

using namespace cv;
using namespace std;
using namespace chrono;

// class to hold a list of equivalent labels
class ConnectedComponents {
  private:
  vector<set<int>> equivs; // list of sets of equivalents
  vector<int> labels; // final resolved equivalent for each label
  vector<Vec3b> colours; // colours for the labels
  int nlabels=0;

  public:
  void addequiv(int s1,int s2) {
    uint m=max(s1,s2);
    if(m>=equivs.size())
      equivs.resize(m+1);
    equivs.at(s1).insert(s2); 
    equivs.at(s2).insert(s1);
  }

  void resolveequvs(int i) { // find all the equivalent labels to label i and set them in the labels array
    if(labels[i]!=-1) return;
    labels[i]=nlabels;
    for(const int &eq : equivs[i]) { // recursively find all equivalents
      resolveequvs(eq);
      labels[eq]=nlabels;
    }
  }
  
  void findlabels() { // get equivalents for all the labels
    labels.resize(equivs.size(),-1);
    for(unsigned i=0;i<equivs.size();i++) {
      if(labels[i]==-1) {
        resolveequvs(i);
        colours.push_back(Vec3b(rand()%220+35,rand()%220+35,rand()%220+35));
        nlabels++;
      }
    }
  }
  int getlabel(int n) {
    return labels[n];
  }
  Vec3b getcolour(int n) {
    return colours[n];
  }
};

int main(int argc, char** argv) {
    (void)argv[argc - 1];
    srand(100);
    Mat_<uint8_t> im_grey=imread("../../Binary1.jpg",IMREAD_GRAYSCALE);
    
    // use the following for a more difficult example
    // Mat_<uint8_t> im_grey=imread("../../Hilbert.png",IMREAD_GRAYSCALE);

    //  uncomment the following for a random image instead
    /*
    Mat_<uint8_t> im_grey(1000,1000);
    for(int y=0;y<im_grey.rows;y++) {
      for(int x=0;x<im_grey.cols;x++) {
        im_grey(y,x)=min(rand()%256+24,255);
      }
    }
    */
  
    //uncomment to invert
    //im_grey=255-im_grey;
    
    imshow("input", im_grey);
    int counter=0;
    int s1,s2;
    // holds the labels
    Mat_<int> A(im_grey.rows,im_grey.cols);
    A=-1;
    ConnectedComponents cc;
    system_clock::time_point start = system_clock::now();
    system_clock::time_point end;
    float mseconds;
    
    // Pass1: assign labels and record equivalants
    for(int y=0;y<A.rows;y++) {
      for(int x=0;x<A.cols;x++) {
        if(im_grey(y,x)>128) { // if pixel is set (white)
          int g1=0;
          int g2=0;
          if(y-1>=0) g1=im_grey(y-1,x); // pixel above
          if(x-1>=0) g2=im_grey(y,x-1); // pixel to the left
          if (g1>128 || g2>128) { // if either is set
            int y1=max(y-1,0);
            int x1=max(x-1,0);
            s1=A(y1,x); // label in previous row
            s2=A(y,x1); // label in previous column
            if(s1!=-1) { // set the current label to one of them
              A(y,x)=s1;
            } else {
              if(s2!=-1)
                A(y,x)=s2;
            }
            // if the labels are both different then add an equivalent for them.
            if(s1!=s2 && s1!=-1 && s2!=-1) {
              cc.addequiv(s1,s2);
            }
          } else { // otherwise create a new label
            A(y,x)=counter++;
          }
        }
      }
    }
    end = system_clock::now();
    mseconds = (end - start) / 1ms;
    cout<<"Pass1:"<<mseconds<<"ms"<<endl;
    // resolve all the labels
    cc.findlabels();
    end = system_clock::now();
    mseconds = (end - start) / 1ms;
    cout<<"Labels:"<<mseconds<<"ms"<<endl;
    // Pass 2: replace all the lables with their equivalent
    for(int y=0;y<A.rows;y++) {
      for(int x=0;x<A.cols;x++) {
        if(A(y,x)!=-1) {
          int l=cc.getlabel(A(y,x));
          if(l!=-1)
            A(y,x)=l;
        }
      }
    }
    end = system_clock::now();
    mseconds = (end - start) / 1ms;
    cout<<"Pass2:"<<mseconds<<"ms"<<endl;
    cout<<counter<<endl;
    imwrite("op.png",A);
    Mat_<Vec3b> c(im_grey.rows,im_grey.cols);
    c=0;
    // make output image with colours for the labels
    for(int y=0;y<im_grey.rows;y++) {
      for(int x=0;x<im_grey.cols;x++)
        if(A(y,x)!=-1)
          c(y,x)=cc.getcolour(A(y,x));
    }
    imwrite("opc.png",c);
    imshow("output",c);
    waitKey(0);
}
