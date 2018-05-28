// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Date:     2013/10/08

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure
static const float LOGBETA = logf(BETA);
static const double MAXITER = 100000.0;

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    float fNiter = 100000.0f; //Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    vector<int> currentInliers;
    FMatrix<float,9,9> A;

    int num_inliers=0;
    int iter = 0;

    int num_samples = matches.size();
    vector<Match> startMatches;
    vector<int> startIndex;
    int i;
    srand(time(0));


    //RANSAC-------------
    cout<<"RANSAC algorithm starting... "<< endl;
     while (iter < Niter && iter < MAXITER) {

         startMatches.clear();
         startIndex.clear();
         num_inliers = 0;
         i=0;

         while (i<8){
             int r = intRandom(0,matches.size());
             if (find(startIndex.begin(), startIndex.end(), r) == startIndex.end()){
                 startMatches.push_back(matches[r]);
                 startIndex.push_back(r);
                 i++;
             }
         }

        //Compute F over those samples////////////////////

        // Normalize points

        FMatrix<float,3,3> N;
        N.fill(0.0f);
        N(0,0) = 0.001f; N(1,1) = 0.001f; N(2,2) = 1.0f;

        vector<Match> normalizedMatches;
        FVector<float,3> xy;

        for (size_t i=0; i<8; i++){
            Match m;
            xy[0] = startMatches[i].x1;
            xy[1] = startMatches[i].y1;
            xy[2] = 1.0f;

            xy = N*xy;
            m.x1 = xy[0];
            m.y1 = xy[1];

            xy[0] = startMatches[i].x2;
            xy[1] = startMatches[i].y2;

            xy = N*xy;
            m.x2 = xy[0];
            m.y2 = xy[1];

            normalizedMatches.push_back(m);
        }

        A.fill(0.0f);

        for (size_t i=0; i<8; i++){
            A(i,0) = normalizedMatches[i].x1*normalizedMatches[i].x2;
            A(i,1) = normalizedMatches[i].x1*normalizedMatches[i].y2;
            A(i,2) = normalizedMatches[i].x1;
            A(i,3) = normalizedMatches[i].x2*normalizedMatches[i].y1;
            A(i,4) = normalizedMatches[i].y1*normalizedMatches[i].y2;
            A(i,5) = normalizedMatches[i].y1;
            A(i,6) = normalizedMatches[i].x2;
            A(i,7) = normalizedMatches[i].y2;
            A(i,8) = 1;
        }

        //cout<< "A" << A <<endl;

        FVector<float,9> S;                      // Singular value decomposition:
        FMatrix<float,9,9> U, Vt;
        svd(A,U,S,Vt,false);

        FMatrix<float,3,3> F;
        F(0,0) = Vt(8,0);
        F(0,1) = Vt(8,1);
        F(0,2) = Vt(8,2);
        F(1,0) = Vt(8,3);
        F(1,1) = Vt(8,4);
        F(1,2) = Vt(8,5);
        F(2,0) = Vt(8,6);
        F(2,1) = Vt(8,7);
        F(2,2) = Vt(8,8);

        // Constrain F to rank 2

        FVector<float,3> S_vec;                      // Singular value decomposition:
        FMatrix<float,3,3> U2, Vt2;

        svd(F,U2,S_vec,Vt2);
        FMatrix<float,3,3> S2;
        S2.fill(0);
        S2(0,0) = S_vec[0];
        S2(1,1) = S_vec[1];

        F = U2*S2*Vt2;

        // De-normalize F
        F = N*F*N;

        //cout << "F : " << F << endl;

        float dist;

        currentInliers.clear();

        // discriminate inliers and outliers
        for (size_t i=0;i<num_samples;i++){
            const Match& m = matches[i];
            FVector<float, 3> X1(m.x1,m.y1,1);
            FVector<float, 3> X2(m.x2,m.y2,1);

            FVector<float, 3> D = transpose(F) * X1;

            dist = abs((X2*D)/(sqrt(pow(D[0],2)+pow(D[1],2))));

            if (dist < distMax){
                currentInliers.push_back(i);
                ++num_inliers;
            }
        }

        if (num_inliers > bestInliers.size()){
            bestInliers = currentInliers;
            bestF = F;
            float ratio = 1.0f*num_inliers/(1.0f*num_samples);
            fNiter = (LOGBETA/logf(1-pow(ratio,8)));

            if (fNiter*logf(1-pow(ratio,8))<LOGBETA){
                Niter = (int)fNiter;}
        }

        ++iter;

        //cout<<"Niter"<<Niter << endl;

    }

    cout<< "Niter: " << Niter <<endl;
    cout<< "F_RANSAC_8pts: "<< bestF <<endl;

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);


    //Least squares refinement of F

    Matrix<float> A_LS(bestInliers.size(),9);

    FVector<float,3> xy;

    for (size_t i=0; i<matches.size(); i++){
        Match m;
        xy[0] = matches[i].x1;
        xy[1] = matches[i].y1;
        xy[2] = 1.0f;

        xy[0] /= 1000.0f;
        xy[1] /= 1000.0f;
        m.x1 = xy[0];
        m.y1 = xy[1];

        xy[0] = matches[i].x2;
        xy[1] = matches[i].y2;

        xy[0] /= 1000.0f;
        xy[1] /= 1000.0f;
        m.x2 = xy[0];
        m.y2 = xy[1];

        A_LS(i,0) = m.x1*m.x2;
        A_LS(i,1) = m.x1*m.y2;
        A_LS(i,2) = m.x1;
        A_LS(i,3) = m.x2*m.y1;
        A_LS(i,4) = m.y1*m.y2;
        A_LS(i,5) = m.y1;
        A_LS(i,6) = m.x2;
        A_LS(i,7) = m.y2;
        A_LS(i,8) = 1;
    }

    Vector<float> S(9);                      // Singular value decomposition:
    Matrix<float> U(matches.size(),matches.size());
    Matrix<float> Vt(9,9);
    svd(A_LS,U,S,Vt,false);

    bestF(0,0) = Vt(8,0);
    bestF(0,1) = Vt(8,1);
    bestF(0,2) = Vt(8,2);
    bestF(1,0) = Vt(8,3);
    bestF(1,1) = Vt(8,4);
    bestF(1,2) = Vt(8,5);
    bestF(2,0) = Vt(8,6);
    bestF(2,1) = Vt(8,7);
    bestF(2,2) = Vt(8,8);

    //cout<<"F_LS before rank 2: "<<bestF<<endl;
    // Constrain F to rank 2

    FVector<float, 3> S_vec;                      // Singular value decomposition:
    FMatrix<float,3,3> U2, Vt2;

    svd(bestF,U2,S_vec,Vt2);
    FMatrix<float,3,3> S2;
    S2.fill(0);
    S2(0,0) = S_vec[0];
    S2(1,1) = S_vec[1];

    bestF = U2*S2*Vt2;

    // De-normalize F
    FMatrix<float,3,3> N;
    N.fill(0.0f);
    N(0,0) = 0.001f; N(1,1) = 0.001f; N(2,2) = 1.0f;

    bestF = N*bestF*N;

    cout << "inliers: " << bestInliers.size() << endl;

    return bestF;

}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {

    cout << "click on a point in left or right image to display the associated epipolar line" << endl;
    cout << "right click when finished" << endl;

    float a;
    float b;
    while(true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;
        int x1, y1, x2, y2;

        FVector<double,3> X;

        X[0] = x;
        X[1] = y;
        X[2] = 1;

        if (x<I1.width()){
            X = transpose(F)*X; // get the epipolar line of X
            a = -X[0]/X[1];
            b = -X[2]/X[1];
            x1 = 0;
            x2 = I2.width();

            y1 = a*x1 + b;
            y2 = a*x2 + b;
            x1 += I1.width();
            x2 += I1.width();

        }
        else if(x>I1.width()){
            X[0] = x - I1.width();
            X = F*X;
            a = -X[0]/X[1];
            b = -X[2]/X[1];
            x1 = 0;
            x2 = I1.width()-1;

            y1 = a*x1 + b;
            y2 = a*x2 + b;
        }

        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(x, y, 2, c);
        drawLine(x1,y1,x2,y2,c);

    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    cout << " matches: " << matches.size() << endl;
    click();

    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);
    }

    cout << "click on the image to move to the next stage" << endl;
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
