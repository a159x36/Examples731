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

// class for a node in the huffman tree
// used for encoding and decoding
class HuffmanNode {
public:
    int data;
    int freq;
    HuffmanNode* left;
    HuffmanNode* right;

    HuffmanNode(int character,int frequency) {
        data = character;
        freq = frequency;
        left = right = NULL;
    }
};

// compare class for the priority queue
class Compare {
public:
    bool operator()(HuffmanNode* a, HuffmanNode* b) {
        return a->freq > b->freq;
    }
};

 
class Huffman {
private:
    // holds a histogram of the data
    unordered_map<int, int> histogram;
    // maps data to codes
    unordered_map<int, string> map;
public:
    // make the map from the huffman tree
    void makeMap(HuffmanNode* root, string s) {
        // Add 0 to string and recurse
        if (root->left) {
            makeMap(root->left,s+"0");
        }
        // Add 1 to string and recurse
        if (root->right) {
            makeMap(root->right,s+"1");
        }
        // If this is a leaf node,
        if (!root->left && !root->right) {
            map[root->data]=s;
        }
    }

    // save the map to a file
    void savemap(ofstream &fout) {
        fout<<map.size()<<endl;
        for(pair<int,string> p:map) {
            fout<<p.first<<" "<<p.second<<endl;
        }
    }
    // load the map from a file
    void loadmap(ifstream &fin) {
        map.clear();
        int size;
        fin>>size;
        int code;
        string s;
        for(int i=0;i<size;i++) {
            fin>>code>>s;
            map[code]=s;
        }
    }
 
    // put a value into the tree at the node given by traversing it using string s
    void treefromstring(HuffmanNode *root, int val, string s) {
        for(char c:s) {
            if(c=='0') {
                if(root->left==NULL) {
                    root->left=new HuffmanNode(0,0);
                }
                root=root->left;
            }
            if(c=='1') {
                if(root->right==NULL) {
                    root->right=new HuffmanNode(0,0);
                }
                root=root->right;
            }
        }
        root->data=val;
    }

    // decode a string of bits ('0' and '1') using the huffman map
    vector<int> decode(string s) {
        // make a tree using the map
        HuffmanNode *root=new HuffmanNode(0,0);
        HuffmanNode *h;
        // add all the map entries to the tree
        for(pair<int,string> p:map) {
            treefromstring(root,p.first,p.second);
        }
        h=root;
        // use the tree to decode the string into a vector of ints
        vector<int> v;
        for(char c:s) {
            if(c=='0') h=h->left;
            if(c=='1') h=h->right;
            if(h->left==NULL && h->right==NULL) {
                v.push_back(h->data);
                h=root;
            }
        }
        deleteTree(root);
        return v;
    }
    // tree is only used temporarily so delete after use
    void deleteTree(HuffmanNode *root) {
        if(root->left) deleteTree(root->left);
        if(root->right) deleteTree(root->right);
        delete root;
    }

    // make the huffman map from some data
    void makeHuffmanMap(vector<int> data) {
        histogram.clear();
        // build a hostogram of the data
        for(int d:data) {
            histogram[d]++;
        }
        // use a priority queue to hold the data ordered by frequency
        priority_queue<HuffmanNode*, vector<HuffmanNode*>, Compare > pq;
        for(pair<int, int>p:histogram) {
            pq.push(new HuffmanNode(p.first, p.second));
        }
        // combine nodes into the huffman tree
        while (pq.size() >1) {
            HuffmanNode* left = pq.top();
            pq.pop();
            // Node which has least
            // frequency
            HuffmanNode* right = pq.top();
            // Remove from
            // Priority Queue
            pq.pop();
            // A new node is formed
            // with frequency left->freq + right->freq
            HuffmanNode* node = new HuffmanNode(0,left->freq + right->freq);
            node->left = left;
            node->right = right;
            // Push node to
            // Priority Queue
            pq.push(node);
        }
        HuffmanNode* root = pq.top();
        // make the map from the tree
        makeMap(root,"");
        deleteTree(root);
    }

    // get the code for a symbol
    string getcode(int i) {
        return map[i];
    }

    // calculate entropy from the histogram
    void entropy(void) {
        unsigned long int totalelem=0;
        for(pair<int, int>p:histogram) {
            totalelem+=p.second;
        }
        float entropy=0.0;
        for(pair<int, int>p:histogram) {
            entropy=entropy-(((float)p.second)/totalelem)*(log(((float)p.second)/totalelem)/log(2));
        }
        cout<<"Entropy is:" <<entropy<<","<<totalelem<<endl;
    }
};

// get the changes to i and j to move in a zigzag pattern over a Mat
void getd(int i,int j,int maxi, int maxj, int &id, int &jd) {
    id=0;
    jd=0;
    if(i==maxi && j==maxj) return;
    if((i+j)%2) {
        if(i==maxi) {
            jd=1;
        } else if(j==0) {
            id=1;
        } else {
            jd=-1;
            id=1;
        }
    } else {
        if(j==maxj) {
            id=1;
        } else if(i==0) {
            jd=1;
        } else {
            jd=1;
            id=-1;
        }
    }
}

// encode m into a vector of ints using zigzag traversal and run length encoding of zeros 
vector<int> zigzag_encode(Mat_<int> m) {
    vector<int> vals;
    int i=0,j=0;
    int zeros=0;
    int id,jd;
    do {
        int v=m(i,j);
        if(v==0) {
            zeros++; // count runs of zeros
        } else {
            if(zeros) { // end of a run of zeros
                // add a run of zeros as 0 followed by the number of them
                vals.push_back(0);
                vals.push_back(zeros);
                zeros=0;
            }
            vals.push_back(v); // add a non zero value
        }
        // move in a zigag pattern
        getd(i,j,m.rows-1,m.cols-1,id,jd);
        i+=id;
        j+=jd;
    } while(!(id==0 && jd==0)); // until we don't move any more
// not necessary to add the last run of zeros
 //   if(zeros) {
 //               vals.push_back(0);
 //               vals.push_back(zeros);
 //   }
    return vals;
} 

// decode vals into a Mat, reverse of encoding, zigzag and run length encoding of zeros
void zigzag_decode(Mat_<int> m, vector<int> vals) {
    int i=0,j=0;
    int zeros=0;
    int id,jd;
    m=0; // initialise to all 0
    for(int v:vals) {
        if(v==0) {
            zeros=1; // wait for next v to see how many zeros
        } else {
            if(zeros) {
                // skip over all the zeros
                for(int k=0;k<v;k++) {
                    getd(i,j,m.rows-1,m.cols-1,id,jd);
                    i+=id;
                    j+=jd;
                }
                zeros=0;
            } else {
                // put a single value in the Mat
                m(i,j)=v;
                getd(i,j,m.rows-1,m.cols-1,id,jd);
                i+=id;
                j+=jd;
            }
        }
    } 
    cout<<"zigzagdecode:"<<i<<","<<j<<endl;
}

// quantise by dividing by a coeficient determined by the distance from (0,0) and a factor qf
// then rounding to an int
void quantize(Mat_<float> transform, Mat_<int>trint, float qf) {
    for(int i=0;i<transform.rows;i++) {
        for(int j=0;j<transform.cols;j++) {
            float qc=hypot(i,j)*qf*2+1;
            trint(i,j)=round(transform(i,j)/qc);
        }
    }
}

void dequantize(Mat_<float> transform, Mat_<int>trint, float qf) {
    for(int i=0;i<transform.rows;i++) {
        for(int j=0;j<transform.cols;j++) {
            float qc=hypot(i,j)*qf*2+1;
            transform(i,j)=trint(i,j)*qc;
        }
    }
}

// calculate mean square error between two images
float mse(Mat_<uint8_t> m1, Mat_<uint8_t> m2) {
    float sumsq=0;
    float d;
    for(int i=0;i<m1.rows;i++) {
        for(int j=0;j<m1.cols;j++) {
            d=m1(i,j)-m2(i,j);
            sumsq+=d*d;
        }
    }
    return sumsq/(m1.rows*m1.cols);
}

// main, uses webcam to get a colour image, encoded it using the dct, quantizes this
// then dequantizes it and decodes it using an idct. 
int main(int argc, char** argv) {
    bool COLOUR=true;
    bool WEBCAM=true;

    (void)argv[argc - 1];
    VideoCapture cap;
    Mat_<Vec3b> frame;
    Mat_<Vec3b> image;
    if(!WEBCAM) {
        image=imread("../../../../Images/baboon.jpg");
        frame=image.clone();
    } else {
        cap.open(0);
        if (!cap.isOpened()) {
            cout << "Failed to open camera" << endl;
            return 0;
        }
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        //   cap.set(CAP_PROP_FRAME_WIDTH, 960);
        //   cap.set(CAP_PROP_FRAME_WIDTH, 1600);
        //   cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        //   cap.set(CAP_PROP_FRAME_HEIGHT, 720);
        //   cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
        cap >> frame;
    }
    
    Menu m("Image Coding",{"Save","Load","Exit"});
    namedWindow("WebCam", WINDOW_FREERATIO);
    namedWindow("Result", WINDOW_AUTOSIZE);
    namedWindow("Transform", WINDOW_FREERATIO);
    namedWindow("Colour", WINDOW_FREERATIO);

    
    printf("frame size %d %d\n", frame.rows, frame.cols);
    resizeWindow("WebCam",frame.cols,frame.rows);
    resizeWindow("Result",frame.cols,frame.rows);
    resizeWindow("Transform",frame.cols,frame.rows);
    resizeWindow("Colour",frame.cols,frame.rows);
    int key = 0;
    float fps = 0.0;
    Mat_<uint8_t> proc(frame.rows,frame.cols);
    Mat_<float> grey(frame.rows,frame.cols);
    Mat_<float> u_im(frame.rows,frame.cols);
    Mat_<float> v_im(frame.rows,frame.cols);
    Mat_<float> transform(frame.rows,frame.cols);
    Mat_<Vec3b> colour(frame.rows,frame.cols);
    Mat_<int> trint(frame.rows,frame.cols);
    Mat_<uint8_t> channels[3];

    int cutoff=frame.cols/4;
    createTrackbar( "cutoff", "Result", &cutoff, hypot(frame.cols,frame.rows));
    while (1) {
        system_clock::time_point start = system_clock::now();
        if(WEBCAM) {
            cap >> frame;
        } else {
            frame=image.clone();
        }
        if (frame.empty()) break;

        Mat proc;
        cvtColor(frame,colour,COLOR_BGR2YUV);    
        split(colour,channels);
        channels[0].convertTo(grey,CV_32FC1,1);
        dct(grey,transform);
      
        float qf=cutoff/1000.0;
        quantize(transform,trint,qf);
        vector<int> rle_zz=zigzag_encode(trint);
        Huffman h;
        h.makeHuffmanMap(rle_zz);
        int encoded_size=0;
        for(int d:rle_zz) {
            encoded_size+=h.getcode(d).length();
        }
        encoded_size/=8;
        
        //h.entropy();
        dequantize(transform,trint,qf);
        idct(transform,grey);
        grey.convertTo(proc,CV_8UC1);
        float err=mse(channels[0],proc);

        imshow("Transform",transform/256+0.5);

        if(COLOUR) {
            Mat down;
            Mat_<float> transform(frame.rows/2,frame.cols/2);
            resize(channels[1],down,Size(frame.cols/2,frame.rows/2));
            down.convertTo(u_im,CV_32FC1,1);
            dct(u_im,transform);
            quantize(transform,trint,qf);
            dequantize(transform,trint,qf);
            idct(transform,grey);
            grey.convertTo(down,CV_8UC1);
            resize(down,channels[1],Size(frame.cols,frame.rows));

            resize(channels[2],down,Size(frame.cols/2,frame.rows/2));
            down.convertTo(v_im,CV_32FC1,1);
            dct(v_im,transform);
            quantize(transform,trint,qf);
            dequantize(transform,trint,qf);
            idct(transform,grey);
            grey.convertTo(down,CV_8UC1);
            resize(down,channels[2],Size(frame.cols,frame.rows));
            proc.copyTo(channels[0]);
            merge(channels,3,colour);
            cvtColor(colour,colour,COLOR_YUV2BGR);
            imshow("Colour",colour);
        }

        cout<<"Encoded size:"<<encoded_size<<" bytes MSE="<<err<<endl;
        ostringstream ss;
        ss<<setw(4)<<setprecision(4)<<setfill('0')<<encoded_size<<"bytes "<<fps<<"fps ";
        string str=ss.str();
        putText(frame, str, Vec2i(12, 32), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 0), 2, 8);
        putText(frame, str, Vec2i(10, 30), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 255), 2, 8);
        imshow("WebCam", frame);
        imshow("Result",proc);
        
        
        key = waitKey(1);
        if (key == 113 || key == 27) return 0;  // either esc or 'q'
        if(m.getselected()=="Save") {
            m.setselected(-1);
            stringstream huffman_encoded;
            for(int d:rle_zz) {
                huffman_encoded<<h.getcode(d);
            }
            ofstream opfile("../../image.txt");
            opfile<<qf<<endl;
            h.savemap(opfile);
            opfile<<huffman_encoded.str();
            opfile.close();
        }
        if(m.getselected()=="Load") {
            Mat_<float> transform(frame.rows,frame.cols);
            m.setselected(-1);
            ifstream ipfile("../../image.txt");
            ipfile>>qf;
            h.loadmap(ipfile);
            string s;
            ipfile>>s;
            ipfile.close();
            vector<int> rle=h.decode(s);
            zigzag_decode(trint,rle);
            dequantize(transform,trint,qf);
            idct(transform,grey);
            imshow("Loaded",grey/256);
        }
        system_clock::time_point end = system_clock::now();
        float mseconds = (end - start) / 1ms;
        fps = 1000.0f / mseconds;
    }
}
