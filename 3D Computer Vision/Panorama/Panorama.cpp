//---- Original project

// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

//---- Version modified for Practical Session 1
// Author:   Michaël Ramamonjisoa
// Date:     2017/10/10


#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    int button=0;
    IntPoint2 current_point;

    pts1.clear();
    pts2.clear();

    cout << "Click on matching points in the two images (at least 4 pairs). Right-click for the last pair to move to the next step." <<endl;

    int i = 0;
    int count = 4;
    do {
        showWindow(w1);
        setActiveWindow(w1);
        button = getMouse(current_point);
        pts1.push_back(current_point);
        drawEllipse(current_point-IntPoint2(10,10),20,20,WHITE,3,true);

        showWindow(w2);
        setActiveWindow(w2);
        button = getMouse(current_point);
        pts2.push_back(current_point);
        drawEllipse(current_point-IntPoint2(10,10),20,20,WHITE,3,true);

        --count;
        if (count>0) {
            cout << "Please choose "<< count << " more pairs of points." << endl;
        } else cout << "Right click to move to the next step." << endl;

        i++;
    } while(i<100 && button!=3);
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    A.fill(0);
    Vector<double> B(2*n);
    B.fill(0);

    for (int i=0; i<n; i++) {
        A(2*i,0)=pts1[i].x(); A(2*i,1)=pts1[i].y(); A(2*i,2)=1; A(2*i,6)=-pts2[i].x()*pts1[i].x(); A(2*i,7)=-pts2[i].x()*pts1[i].y();
        A(2*i+1,3)=pts1[i].x(); A(2*i+1,4)=pts1[i].y(); A(2*i+1,5)=1; A(2*i+1,6)=-pts2[i].y()*pts1[i].x(); A(2*i+1,7)=-pts2[i].y()*pts1[i].y();

        B[2*i]=pts2[i].x(); B[2*i+1]=pts2[i].y();
    }

    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    /*
    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    */
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;
}

// Panorama construction-----using the pull method
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    Matrix<float> G = inverse(H);

    float x0=0, y0=0, x1=I1.width(), y1=I1.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=G*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I2.width(); v[1]=0; v[2]=1;
    v=G*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I2.width(); v[1]=I2.height(); v[2]=1;
    v=G*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I2.height(); v[2]=1;
    v=G*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);

    int x; int y;

    //---- Fill I with I2----

    for (int x=0; x<I2.width(); x++) {
        for (int y=0; y<I2.height(); y++) {
            I((int)(I.width()-I2.width()+x),(int)(I.height()-I2.height()+y)) = I2(x,y);
        }
    }

    //-------Apply Homography------ we use the pull method
    for (int x_im=0; x_im<I.width(); x_im++) {
        for (int y_im=0; y_im<I.height(); y_im++) {

            v[0]=x_im-I.width()+I2.width(); v[1]=y_im-I.height()+I2.height(); v[2]=1;
            Vector<float> v_inv = G*v;
            v_inv /= v_inv[2];

            //x=(int)(v_inv[0]+I.width()-x1); y=(int)(v_inv[1]+I.height()-y1);
            x=(int)v_inv[0]; y=(int)v_inv[1];

            if (x>=0 && x<I1.width() && y>=0 && y<I1.height()) {
                if (x_im>=I.width()-I2.width() && x_im<x1-x0 && y>=I.height()-I2.height() && y_im<y1-y0) {
                    I(x_im,y_im) = I1(x,y)/((byte)2)+I(x_im,y_im)/((byte)2);
                }
                else I(x_im,y_im) = I1(x,y);
            }
        }
    }

    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
