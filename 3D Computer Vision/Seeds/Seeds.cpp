// Imagine++ project
// Project:  Seeds
// Author:   Pascal Monasse

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <queue>
#include <iostream>
#include <cmath>
#include <algorithm>
using namespace Imagine;
using namespace std;

/// Min and max disparities
static const float dMin=-30, dMax=-7;

/// Min NCC for a seed
static const float nccSeed=0.97;

/// Radius of patch for correlation
static const int win=(9-1)/2;
/// To avoid division by 0 for constant patch
static const float EPS=0.1f;

/// A seed
struct Seed {
    Seed(int x0, int y0, int d0, float ncc0)
    : x(x0), y(y0), d(d0), ncc(ncc0) {}
    int x,y, d;
    float ncc;
};

/// Order by NCC
bool operator<(const Seed& s1, const Seed& s2) {
    return (s1.ncc<s2.ncc);
}

/// 4-neighbors
static const int dx[]={+1,  0, -1,  0};
static const int dy[]={ 0, -1,  0, +1};

/*
/// 8-neighbors
static const int dx[]={+1, +1,  0, -1, -1, -1,  0, +1};
static const int dy[]={ 0, -1, -1, -1,  0, +1, +1, +1};
*/

/// Display disparity map
static void displayDisp(const Image<int> disp, Window W, int subW) {
    Image<Color> im(disp.width(), disp.height());
    for(int j=0; j<disp.height(); j++)
        for(int i=0; i<disp.width(); i++) {
            if(disp(i,j)<dMin || disp(i,j)>dMax)
                im(i,j)=Color(0,255,255);
            else {
                int g = 255*(disp(i,j)-dMin)/(dMax-dMin);
                im(i,j)= Color(g,g,g);
            }
        }
    setActiveWindow(W,subW);
    display(im);
    showWindow(W,subW);
}

/// Show 3D window
static void show3D(const Image<Color> im, const Image<int> disp) {
#ifdef IMAGINE_OPENGL // Imagine++ must have been built with OpenGL support...
    // Intrinsic parameters given by Middlebury website
    const float f=3740;
    const float d0=-200; // Doll images cropped by this amount
    const float zoom=2; // Half-size images, should double measured disparity
    const float B=0.160; // Baseline in m
    FMatrix<float,3,3> K(0.0f);
    K(0,0)=-f/zoom; K(0,2)=disp.width()/2;
    K(1,1)= f/zoom; K(1,2)=disp.height()/2;
    K(2,2)=1.0f;
    K = inverse(K);
    K /= K(2,2);
    std::vector<FloatPoint3> pts;
    std::vector<Color> col;
    for(int j=0; j<disp.height(); j++)
        for(int i=0; i<disp.width(); i++)
            if(dMin<=disp(i,j) && disp(i,j)<=dMax) {
                float z = B*f/(zoom*disp(i,j)+d0);
                FloatPoint3 pt((float)i,(float)j,1.0f);
                pts.push_back(K*pt*z);
                col.push_back(im(i,j));
            }
    Mesh mesh(&pts[0], pts.size(), 0,0,0,0,VERTEX_COLOR);
    mesh.setColors(VERTEX, &col[0]);
    Window W = openWindow3D(512,512,"3D");
    setActiveWindow(W);
    showMesh(mesh);
#else
    std::cout << "No 3D: Imagine++ not built with OpenGL support" << std::endl;
#endif
}


/// Sum of pixel values in patch centered on (i,j).
static float sum(const Image<byte>& im, int i, int j)
{    
    float s=0.0f;
    int x,y;

    for (int q=-win;q<win+1;q++){
        for (int p=-win;p<win+1;p++){

            // Sum with border repeat
            if (p+i<0)
                x = 0;
            else if (p+i>=im.width())
                x = im.width()-1;
            else
                x = p+i;
            if (q+j<0)
                y = 0;
            else if (q+j>=im.height())
                y = im.height()-1;
            else
                y = q+j;

            s += im(x,y);
        }
    }

    return s;
}

/// Correlation between patches centered on (i1,j1) and (i2,j2). The values
/// m1 or m2 are subtracted from each pixel value.
static float correl(const Image<byte>& im1, int i1,int j1,float m1,
                    const Image<byte>& im2, int i2,int j2,float m2)
{

    float dist = 0.0f;
    float sum2_1 = 0.0f;
    float sum2_2 = 0.0f;
    float sum_num = 0.0f;
    float im1_term = 0.0f;
    float im2_term = 0.0f;
    int x1,y1,x2,y2;

    for (int q=-win;q<win+1;q++){
        for (int p=-win;p<win+1;p++){

            // Correlation with border repeat
            if (p+i1<0)
                x1 = 0;
            else if (p+i1>=im1.width())
                x1 = im1.width()-1;
            else
                x1 = p+i1;
            if (q+j1<0)
                y1 = 0;
            else if (q+j1>=im1.height())
                y1 = im1.height()-1;
            else
                y1 = q+j1;
            if (p+i2<0)
                x2 = 0;
            else if (p+i2>=im2.width())
                x2 = im2.width()-1;
            else
                x2 = p+i2;
            if (q+j2<0)
                y2 = 0;
            else if (q+j2>=im2.height())
                y2 = im2.height()-1;
            else
                y2 = q+j2;


            im1_term = im1(x1,y1)-m1;
            im2_term = im2(x2,y2)-m2;
            sum_num += im1_term*im2_term;

            sum2_1 += im1_term*im1_term;
            sum2_2 += im2_term*im2_term;
        }
    }
    dist = sum_num/(sqrt(sum2_1*sum2_2+EPS));

    return dist;
}

/// Centered correlation of patches of size 2*win+1.
static float ccorrel(const Image<byte>& im1,int i1,int j1,
                     const Image<byte>& im2,int i2,int j2)
{
    float m1 = sum(im1,i1,j1);
    float m2 = sum(im2,i2,j2);
    int w = 2*win+1;
    return correl(im1,i1,j1,m1/(w*w), im2,i2,j2,m2/(w*w));
}

/// Compute disparity map from im1 to im2, but only at points where NCC is
/// above nccSeed. Set to true the seeds and put them in Q.
static void find_seeds(Image<byte> im1, Image<byte> im2,
                       float nccSeed,
                       Image<int>& disp, Image<bool>& seeds,
                       std::priority_queue<Seed>& Q) {
    disp.fill(dMin-1);
    seeds.fill(false);
    while(! Q.empty())
        Q.pop();

    for(int y=win; y+win<im1.height() && y+win<im2.height(); y++){
        for(int x=win; x+win<im1.width(); x++) {
            float ncc_max=0;
            int d_max=0;
            int d_start = 0;
            d_start = std::max((float)(-x+win),dMin);

            // scan image 2 in the horizontal direction to get best match patch in terms of ncc
            for (int d=d_start; x+win+d<im2.width() && d<=dMax; d++){
                float ncc_temp = ccorrel(im1,x,y,im2,x+d,y);
                if (ncc_temp>ncc_max){
                    ncc_max=ncc_temp; // takes the maximal value of disparity along horizontal line
                    d_max = d;
                }
            }
            if (dMin<=d_max && d_max<=dMax && ncc_max>nccSeed){
                seeds(x,y)=true; // only keeps seeds with disparities in acceptable range and ncc higher than nccSeed
                Seed s(x,y,d_max,ncc_max);
                disp(x,y) = d_max; // keeps the disparity associated with the best NCC distance
                Q.push(s);
            }

        }
    }
}

/// Propagate seeds
static void propagate(Image<byte> im1, Image<byte> im2,
                      Image<int>& disp, Image<bool>& seeds,
                      std::priority_queue<Seed>& Q) {
    while(! Q.empty()) {
        Seed s=Q.top();
        Q.pop();
        for(int i=0; i<4; i++) {
            float x=s.x+dx[i], y=s.y+dy[i];
            if(0<=x-win && 0<=y-win &&
               x+win<im2.width() && y+win<im2.height() &&
               ! seeds(x,y)) {
                Seed q(x,y,0,0.0f);
                int dQ=s.d;
                for (int k=0;k<3;k++){
                    float ncc_temp = ccorrel(im1,x,y,im2,x+s.d-1+k,y);

                    if (ncc_temp >= q.ncc && dMin<=s.d-1+k && s.d-1+k<=dMax){
                        q.ncc = ncc_temp;
                        dQ = s.d-1+k;
                    }
                }
                disp(x,y) = dQ;
                q.d = dQ;
                seeds(x,y) = true;
                Q.push(q);
            }
        }
    }
}

int main()
{
    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, srcPath("im1.jpg")) ||
        ! load(I2, srcPath("im2.jpg")) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    std::string names[5]={"image 1","image 2","dense","seeds","propagation"};
    Window W = openComplexWindow(I1.width(), I1.height(), "Seeds propagation",
                                 5, names);
    setActiveWindow(W,0);
    display(I1,0,0);
    setActiveWindow(W,1);
    display(I2,0,0);

    Image<int> disp(I1.width(), I1.height());
    Image<bool> seeds(I1.width(), I1.height());
    std::priority_queue<Seed> Q;

    // Dense disparity
    cout << "============= Computing the disparity map ... =============" ;
    find_seeds(I1, I2, -1.0f, disp, seeds, Q);
    cout << " Done. "<< endl;
    displayDisp(disp,W,2);

    // Only seeds
    cout << "============= Computing the seeds map ... =================" ;
    find_seeds(I1, I2, nccSeed, disp, seeds, Q);
    cout << " Done. "<< endl;
    displayDisp(disp,W,3);

    // Propagation of seeds
    cout << "============= Propagating the seeds ... ===================" ;
    propagate(I1, I2, disp, seeds, Q);
    cout << " Done. "<< endl;
    displayDisp(disp,W,4);

    // Show 3D (use shift click to animate)
    show3D(I1,disp);

    endGraphics();
    return 0;
}
